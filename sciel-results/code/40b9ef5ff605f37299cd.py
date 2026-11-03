"""Fresh design: smooth soft-min surrogate + L-BFGS-B for 26-circle packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_TIME_LIMIT = 110.0  # safety margin under 120 s wall-clock


def _check_valid(arr):
    """Return (value, valid). value = sum of radii if valid else -inf."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r <= 0):
        return -np.inf, False
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return -np.inf, False
    if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return -np.inf, False
    if np.any(x + r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return -np.inf, False
    if np.any(y + r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return -np.inf, False
    if np.any(x - r < -1e-7) or np.any(x - r > 1 + 1e-7):
        return -np.inf, False
    if np.any(y - r < -1e-7) or np.any(y - r > 1 + 1e-7):
        return -np.inf, False
    # pairwise non-overlap (strictly disjoint interiors)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-9:
                return -np.inf, False
    return float(np.sum(r)), True


def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side))
    return np.array(cells[:N], dtype=float)


def _init_cluster():
    """Clustered initial guess: one big circle per corner + a central cluster."""
    arr = np.empty((N, 3))
    corners = [(0.12, 0.12, 0.12), (0.88, 0.12, 0.12),
               (0.12, 0.88, 0.12), (0.88, 0.88, 0.12)]
    arr[:4, 0] = [c[0] for c in corners]
    arr[:4, 1] = [c[1] for c in corners]
    arr[:4, 2] = [c[2] for c in corners]
    # remaining circles clustered near center in a ring
    m = N - 4
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
    rad = 0.20
    arr[4:, 0] = 0.5 + rad * np.cos(ang)
    arr[4:, 1] = 0.5 + rad * np.sin(ang)
    arr[4:, 2] = 0.08
    return arr


def _softmin_neg(a, b):
    """Smooth approximation of -min(a,b). Returns (value, grad_a, grad_b)."""
    t = 20.0
    m = np.minimum(a, b)
    d = a - b
    z = t * d
    # stable: -min - logaddexp(t*a, t*b) = m - logaddexp(t*a, t*b)
    val = m - np.logaddexp(t * a, t * b)
    e_a = np.exp(t * a - np.logaddexp(t * a, t * b))
    e_b = np.exp(t * b - np.logaddexp(t * a, t * b))
    return val, -e_a, -e_b


def _objective_and_grad(p):
    """p: (3N,) vector [x0,y0,r0, x1,y1,r1, ...]. Minimize -sum(r) + penalty."""
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]

    obj = -np.sum(r)
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)

    # soft boundary penalty: circle must stay inside [0,1]^2
    # constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    t = 30.0
    def softpen(v):
        # v should be >= 0; penalty when v < 0: smooth max(0, -v)
        # use softplus(-v) scaled
        return np.logaddexp(0, -t * v) / t
    def softpen_grad(v):
        # d/dv softplus(-t v)/t = -1/(1+exp(t v))
        return -1.0 / (1.0 + np.exp(t * v))
    for idx in range(N):
        v1 = x[idx] - r[idx]
        v2 = 1.0 - x[idx] - r[idx]
        v3 = y[idx] - r[idx]
        v4 = 1.0 - y[idx] - r[idx]
        obj += softpen(v1) + softpen(v2) + softpen(v3) + softpen(v4)
        g1 = softpen_grad(v1)
        g2 = softpen_grad(v2)
        g3 = softpen_grad(v3)
        g4 = softpen_grad(v4)
        gx[idx] += g1 - g2
        gy[idx] += g3 - g4
        gr[idx] += -(g1 + g2 + g3 + g4)

    # soft non-overlap penalty: for pair (i,j), constraint d - (ri+rj) >= 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < 1e-8:
                d = 1e-8
            v = d - (r[i] + r[j])
            obj += softpen(v)
            gv = softpen_grad(v)
            gx[i] += gv * dx / d
            gx[j] -= gv * dx / d
            gy[i] += gv * dy / d
            gy[j] -= gv * dy / d
            gr[i] += -gv
            gr[j] += -gv

    grad = np.empty(3 * N)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    return obj, grad


def _polish(arr, deadline):
    """High-precision polish: minimize sum of constraint violations, then maximize radii."""
    best = arr.copy()
    best_val, best_valid = _check_valid(best)

    # Phase 1: drive constraint violations to ~0 while keeping radii
    def viol_obj(p):
        x = p[0::3]; y = p[1::3]; r = p[2::3]
        v = 0.0
        for idx in range(N):
            v += max(0.0, -(x[idx] - r[idx]))**2
            v += max(0.0, -(1.0 - x[idx] - r[idx]))**2
            v += max(0.0, -(y[idx] - r[idx]))**2
            v += max(0.0, -(1.0 - y[idx] - r[idx]))**2
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if d < 1e-8:
                    d = 1e-8
                v += max(0.0, (r[i]+r[j]) - d)**2
        return v

    if time.time() < deadline:
        res = minimize(viol_obj, best.ravel(), method='L-BFGS-B',
                       options={'maxiter': 2000, 'maxfun': 4000, 'ftol': 1e-14, 'gtol': 1e-12})
        cand = res.x.reshape(N, 3)
        if viol_obj(cand.ravel()) < viol_obj(best.ravel()) + 1e-12:
            best = cand

    # Phase 2: maximize sum of radii under (soft) constraints
    if time.time() < deadline:
        def final_obj(p):
            x = p[0::3]; y = p[1::3]; r = p[2::3]
            pen = 0.0
            W = 50.0
            for idx in range(N):
                pen += W * max(0.0, -(x[idx]-r[idx]))**2
                pen += W * max(0.0, -(1.0-x[idx]-r[idx]))**2
                pen += W * max(0.0, -(y[idx]-r[idx]))**2
                pen += W * max(0.0, -(1.0-y[idx]-r[idx]))**2
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    if d < 1e-8:
                        d = 1e-8
                    pen += W * max(0.0, (r[i]+r[j])-d)**2
            return -np.sum(r) + pen
        res = minimize(final_obj, best.ravel(), method='L-BFGS-B',
                       options={'maxiter': 4000, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-12})
        cand = res.x.reshape(N, 3)
        cv, cvv = _check_valid(cand)
        bv, bvv = _check_valid(best)
        if cvv and cv > bv:
            best = cand
    return best


def construct():
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT
    best = None
    best_val = -np.inf

    inits = [_init_grid(), _init_cluster()]
    # add a few perturbed inits
    rng = np.random.default_rng(0)
    for k in range(2):
        base = inits[k % 2].copy()
        base[:, :2] += rng.normal(0, 0.02, size=(N, 2))
        base[:, :2] = np.clip(base[:, :2], 0.05, 0.95)
        inits.append(base)

    for init in inits:
        if time.time() > deadline - 20.0:
            break
        p0 = init.ravel()
        # clip radii to a sane range
        p0[2::3] = np.clip(p0[2::3], 0.02, 0.45)
        res = minimize(_objective_and_grad, p0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 3000, 'maxfun': 6000, 'ftol': 1e-12, 'gtol': 1e-10})
        cand = res.x.reshape(N, 3)
        # clip tiny negatives
        cand[:, 2] = np.clip(cand[:, 2], 1e-4, 0.5)
        val, valid = _check_valid(cand)
        if valid and val > best_val:
            best_val = val
            best = cand.copy()
        if time.time() > deadline - 25.0:
            break

    if best is None:
        # fallback to grid (valid)
        g = _init_grid()
        g[:, 2] -= 1e-3
        return g

    # polish best
    if time.time() < deadline - 10.0:
        polished = _polish(best, deadline - 5.0)
        pv, pvalid = _check_valid(polished)
        if pvalid and pv >= best_val - 1e-9:
            best = polished
            best_val = pv

    # final safety: ensure valid; if not, fall back to grid
    fv, fvalid = _check_valid(best)
    if not fvalid:
        g = _init_grid()
        g[:, 2] -= 1e-3
        return g
    return best

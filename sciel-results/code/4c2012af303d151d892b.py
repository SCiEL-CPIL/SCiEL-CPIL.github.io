"""
Circle packing in perimeter-4 rectangle:
Combined contact-graph SLSQP (current) + unified free-aspect-ratio formulation (inspiration).

Structural change: the rectangle width W is a continuous optimization variable
inside the NLP, and the optimizer can simultaneously adjust circle positions,
radii, AND the aspect ratio.  The contact-graph machinery from the current
program is retained as a refinement layer that runs *after* the unified
optimization, using the best aspect ratio found.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 138.0  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _clamp_valid(arr, W, H):
    """Clamp a (N,3) array to be valid in rectangle W×H."""
    cx = np.clip(arr[:, 0], 0.0, W)
    cy = np.clip(arr[:, 1], 0.0, H)
    r = np.maximum(arr[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    return np.column_stack([cx, cy, r])


def _grid_layout(kx, ky, W, H, shrink=0.999):
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


def _random_layout(W, H, seed=None):
    rng = random.Random(seed)
    pts = []
    for _ in range(N):
        cx = rng.uniform(0.05 * W, 0.95 * W)
        cy = rng.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)


# ---------------------------------------------------------------------------
# Contact-graph machinery (from current program)
# ---------------------------------------------------------------------------

def _build_contact_graph(x, W, H, threshold=1.05):
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d <= threshold * (r[i] + r[j]):
                A[i, j] = 1
                A[j, i] = 1
    return A


def _perturb_contact_graph(A, n, n_perturb=3, rng=None):
    if rng is None:
        rng = random
    A_new = A.copy()
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A_new[i, j]]
    for _ in range(n_perturb):
        if edges:
            i, j = rng.choice(edges)
            A_new[i, j] = 0
            A_new[j, i] = 0
            edges.remove((i, j))
    non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not A_new[i, j]]
    for _ in range(n_perturb):
        if non_edges:
            i, j = rng.choice(non_edges)
            A_new[i, j] = 1
            A_new[j, i] = 1
            non_edges.remove((i, j))
    return A_new


def _objective_contact(x, W, H, A, mu=1e3, mu_contact=5e3):
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)

    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)
    for i in range(n):
        if viol_b[i] <= 0:
            continue
        d_left = cx[i]
        d_bottom = cy[i]
        d_right = W - cx[i]
        d_top = H - cy[i]
        min_d = lo[i]
        g_cx = 0.0
        g_cy = 0.0
        eps = 1e-9
        cnt = 0
        if d_left <= min_d + eps:
            g_cx += 1.0; cnt += 1
        if d_right <= min_d + eps:
            g_cx += -1.0; cnt += 1
        if d_bottom <= min_d + eps:
            g_cy += 1.0; cnt += 1
        if d_top <= min_d + eps:
            g_cy += -1.0; cnt += 1
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
        dv = 2.0 * mu * viol_b[i]
        grad[i, 0] += dv * (-g_cx)
        grad[i, 1] += dv * (-g_cy)
        grad[i, 2] += dv * 1.0

    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            grad[i, 2] += 2.0 * mu * viol_r[i] * (-1.0)

    mu_c = mu * 0.1
    viol_cx_lo = np.maximum(-cx, 0.0)
    viol_cx_hi = np.maximum(cx - W, 0.0)
    viol_cy_lo = np.maximum(-cy, 0.0)
    viol_cy_hi = np.maximum(cy - H, 0.0)
    val += mu_c * (np.sum(viol_cx_lo**2) + np.sum(viol_cx_hi**2) +
                   np.sum(viol_cy_lo**2) + np.sum(viol_cy_hi**2))
    for i in range(n):
        if viol_cx_lo[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_lo[i] * (-1.0)
        if viol_cx_hi[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_hi[i] * 1.0
        if viol_cy_lo[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_lo[i] * (-1.0)
        if viol_cy_hi[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_hi[i] * 1.0

    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu * viol * 1.0
                grad[j, 0] += 2.0 * mu * viol * (-1.0)
                grad[i, 2] += 2.0 * mu * viol * 1.0
                grad[j, 2] += 2.0 * mu * viol * 1.0
                continue
            dv = 2.0 * mu * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                continue
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            target = r[i] + r[j]
            diff = d - target
            if abs(diff) < 1e-10:
                continue
            val += mu_contact * diff ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu_contact * diff * 1.0
                grad[j, 0] += 2.0 * mu_contact * diff * (-1.0)
                grad[i, 2] += 2.0 * mu_contact * diff * (-1.0)
                grad[j, 2] += 2.0 * mu_contact * diff * (-1.0)
                continue
            dv = 2.0 * mu_contact * diff
            grad[i, 0] += dv * (dx / d)
            grad[i, 1] += dv * (dy / d)
            grad[j, 0] += dv * (-dx / d)
            grad[j, 1] += dv * (-dy / d)
            grad[i, 2] += dv * (-1.0)
            grad[j, 2] += dv * (-1.0)

    return val, grad


def _force_precondition(x, W, H, steps=100, lr=0.005):
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    for step in range(steps):
        fcx = np.zeros(n)
        fcy = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req and d > 1e-12:
                    f = (req - d) / d
                    fcx[i] += f * dx
                    fcy[i] += f * dy
                    fcx[j] -= f * dx
                    fcy[j] -= f * dy
                elif d <= 1e-12:
                    angle = random.uniform(0, 2 * np.pi)
                    fcx[i] += 0.1 * np.cos(angle)
                    fcy[i] += 0.1 * np.sin(angle)
                    fcx[j] -= 0.1 * np.cos(angle)
                    fcy[j] -= 0.1 * np.sin(angle)
        cx += lr * fcx
        cy += lr * fcy
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r * 1.0005 + 0.00005, lo)
        r = np.maximum(r, 1e-6)
        lr *= 0.995
    return np.column_stack([cx, cy, r])


def _optimize_contact(x0, W, H, A=None, max_iter=200):
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0
    if A is None:
        A = _build_contact_graph(x0, W, H, threshold=1.02)

    def fun(x):
        v, _ = _objective_contact(x, W, H, A)
        return v

    def grad(x):
        _, g = _objective_contact(x, W, H, A)
        return g

    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass

    if time.time() < DEADLINE - 8:
        A2 = _build_contact_graph(best_x, W, H, threshold=1.02)

        def fun2(x):
            v, _ = _objective_contact(x, W, H, A2)
            return v

        def grad2(x):
            _, g = _objective_contact(x, W, H, A2)
            return g

        try:
            res2 = minimize(fun2, best_x, jac=grad2, method='SLSQP',
                            options={'maxiter': 150, 'ftol': 1e-12, 'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass

    return best_x, best_sum


# ---------------------------------------------------------------------------
# Unified free-aspect-ratio NLP (inspired by inspiration 1)
# ---------------------------------------------------------------------------

def _unified_objective(z, mu=500.0):
    """
    z = [x1, y1, r1, ..., xN, yN, rN, w]  (3N+1 elements)
    h = 2 - w
    Returns scalar penalty objective (lower is better; -sum(r) + penalties).
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3:
        return 1e6

    obj = -np.sum(r)

    # Overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            req = r[i] + r[j]
            if d < req:
                obj += mu * (req - d) ** 2

    # Boundary penalty
    for i in range(N):
        if x[i] < r[i]:
            obj += mu * (r[i] - x[i]) ** 2
        elif x[i] > w - r[i]:
            obj += mu * (x[i] - (w - r[i])) ** 2
        if y[i] < r[i]:
            obj += mu * (r[i] - y[i]) ** 2
        elif y[i] > h - r[i]:
            obj += mu * (y[i] - (h - r[i])) ** 2

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += mu * (0.5 - w) ** 2
    elif w > 1.5:
        obj += mu * (w - 1.5) ** 2

    return obj


def _unified_grad(z, mu=500.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3:
        return np.zeros_like(z)

    grad = np.zeros_like(z)
    # d(-sum(r))/dr_i = -1
    grad[2::3][:N] = -1.0

    # Overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            dv = 2.0 * mu * viol
            if d < 1e-12:
                grad[3 * i] += dv * 1.0
                grad[3 * j] += dv * (-1.0)
                grad[3 * i + 2] += dv * 1.0
                grad[3 * j + 2] += dv * 1.0
                continue
            grad[3 * i] += dv * (-dx / d)
            grad[3 * i + 1] += dv * (-dy / d)
            grad[3 * j] += dv * (dx / d)
            grad[3 * j + 1] += dv * (dy / d)
            grad[3 * i + 2] += dv * 1.0
            grad[3 * j + 2] += dv * 1.0

    # Boundary
    for i in range(N):
        if x[i] < r[i]:
            v = r[i] - x[i]
            dv = 2.0 * mu * v
            grad[3 * i] += dv * (-1.0)
            grad[3 * i + 2] += dv * 1.0
        elif x[i] > w - r[i]:
            v = x[i] - (w - r[i])
            dv = 2.0 * mu * v
            grad[3 * i] += dv * 1.0
            grad[3 * i + 2] += dv * (-1.0)
            grad[-1] += dv * (-1.0)  # d(w-r_i)/dw = 1, so d(x - (w-r_i))/dw = -1
        if y[i] < r[i]:
            v = r[i] - y[i]
            dv = 2.0 * mu * v
            grad[3 * i + 1] += dv * (-1.0)
            grad[3 * i + 2] += dv * 1.0
        elif y[i] > h - r[i]:
            v = y[i] - (h - r[i])
            dv = 2.0 * mu * v
            grad[3 * i + 1] += dv * 1.0
            grad[3 * i + 2] += dv * (-1.0)
            # h = 2 - w, so d(h - r_i)/dw = -1, d(y - (h-r_i))/dw = +1
            grad[-1] += dv * 1.0

    # Aspect ratio penalty
    if w < 0.5:
        v = 0.5 - w
        dv = 2.0 * mu * v
        grad[-1] += dv * (-1.0)
    elif w > 1.5:
        v = w - 1.5
        dv = 2.0 * mu * v
        grad[-1] += dv * 1.0

    return grad


def _unified_valid(z, tol=1e-8):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            rs = r[i] + r[j]
            if d2 < rs * rs - tol:
                return False, 0.0
    return True, float(np.sum(r))


def _unified_force_relax(z, max_iter=200, dt=0.003):
    z = z.copy().astype(float)
    n = N
    for _ in range(max_iter):
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                md = r[i] + r[j]
                if d < md and d > 1e-12:
                    ov = md - d
                    fx[i] += (dx / d) * ov * 10
                    fy[i] += (dy / d) * ov * 10
                    fx[j] -= (dx / d) * ov * 10
                    fy[j] -= (dy / d) * ov * 10
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], max(r[i], 0.0), max(w - r[i], 0.01))
            z[3 * i + 1] = np.clip(z[3 * i + 1], max(r[i], 0.0), max(h - r[i], 0.01))
    return z


def _unified_optimize(z0, max_iter=300):
    """Run SLSQP on the unified parameter vector."""
    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    try:
        res = minimize(_unified_objective, z0, jac=_unified_grad,
                       method='SLSQP', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'maxls': 50, 'disp': 0})
        return res.x
    except Exception:
        return z0


def _unified_polish(z, time_budget):
    """Force-directed radius growth on a valid unified solution."""
    z = z.copy().astype(float)
    n = N
    dt = 0.001
    start = time.time()
    for _ in range(500):
        if time.time() - start > time_budget:
            break
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                md = r[i] + r[j]
                if d < md and d > 1e-12:
                    ov = md - d
                    fx[i] += (dx / d) * ov * 10
                    fy[i] += (dy / d) * ov * 10
                    fx[j] -= (dx / d) * ov * 10
                    fy[j] -= (dy / d) * ov * 10
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])
            lo = min(z[3 * i], z[3 * i + 1], w - z[3 * i], h - z[3 * i + 1])
            z[3 * i + 2] = min(z[3 * i + 2] * 1.0003 + 0.00002, lo)
            z[3 * i + 2] = max(z[3 * i + 2], 1e-6)
    return z


# ---------------------------------------------------------------------------
# Contact-graph topology search (from current program)
# ---------------------------------------------------------------------------

def _topology_search(x0, W, H, n_topologies=4, max_iter=150, rng=None):
    if rng is None:
        rng = random
    base_A = _build_contact_graph(x0, W, H, threshold=1.02)
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    x_opt, s_opt = _optimize_contact(x0, W, H, A=base_A, max_iter=max_iter)
    if s_opt > best_sum:
        best_sum = s_opt
        best_x = x_opt

    for _ in range(n_topologies):
        if time.time() > DEADLINE - 10:
            break
        A_pert = _perturb_contact_graph(base_A, N, n_perturb=2, rng=rng)
        x_opt, s_opt = _optimize_contact(x0, W, H, A=A_pert, max_iter=max_iter)
        if s_opt > best_sum:
            best_sum = s_opt
            best_x = x_opt

    return best_x, best_sum


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    def time_ok():
        return time.time() < DEADLINE

    random.seed(42)
    np.random.seed(42)

    # ---- Phase 1: Unified free-aspect-ratio SLSQP multi-start ----
    # Build diverse initial guesses in unified space
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
    side = int(np.ceil(np.sqrt(N)))
    unified_guesses = []

    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.45
            z.extend([cx, cy, r0])
        z.append(w0)
        unified_guesses.append(np.array(z))

    rng = np.random.default_rng(42)
    for _ in range(6):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = []
        for i in range(N):
            cx = rng.uniform(0.1, w0 - 0.1)
            cy = rng.uniform(0.1, h0 - 0.1)
            r0 = rng.uniform(0.05, 0.12)
            z.extend([cx, cy, r0])
        z.append(w0)
        unified_guesses.append(np.array(z))

    for idx, z0 in enumerate(unified_guesses):
        if not time_ok():
            break
        # Force-directed relaxation
        z0r = _unified_force_relax(z0, max_iter=150)

        # SLSQP
        remaining = DEADLINE - time.time()
        maxiter = int(min(250, remaining * 25))
        if maxiter < 50:
            break
        z_opt = _unified_optimize(z0r, max_iter=maxiter)

        valid, val = _unified_valid(z_opt)
        if valid and val > best_sum:
            best_sum = val
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            W = z_opt[-1]
            H = 2.0 - W
            best_arr = np.column_stack([x, y, r])
            best_W = W
            best_H = H
        elif not valid:
            # Try repair: shrink radii
            z_repair = z_opt.copy()
            r_idx = np.arange(2, len(z_repair) - 1, 3)
            z_repair[r_idx] *= 0.95
            valid2, val2 = _unified_valid(z_repair)
            if valid2 and val2 > best_sum:
                best_sum = val2
                x = z_repair[0::3][:N]
                y = z_repair[1::3][:N]
                r = z_repair[2::3][:N]
                W = z_repair[-1]
                H = 2.0 - W
                best_arr = np.column_stack([x, y, r])
                best_W = W
                best_H = H

    # ---- Phase 2: Contact-graph refinement on best unified result ----
    if best_arr is not None and time_ok():
        # Force pre-condition
        if time.time() < DEADLINE - 15:
            best_arr = _force_precondition(best_arr, best_W, best_H, steps=60, lr=0.004)
            s = _evaluate(best_arr, best_W, best_H)
            if s is not None:
                best_sum = s
            else:
                best_sum = 0.0

        # Topology search
        x_opt, s_opt = _topology_search(best_arr, best_W, best_H,
                                        n_topologies=4, max_iter=150)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt

    # ---- Phase 3: Additional unified restarts from best aspect ratio ----
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(4, int(remaining // 15))
        for _ in range(n_extra):
            if not time_ok():
                break
            # Perturb best in unified space
            z_best = np.concatenate([best_arr.ravel(), [best_W]])
            z_pert = z_best + np.random.randn(*z_best.shape) * 0.015
            z_pert[-1] = np.clip(z_pert[-1], 0.4, 1.6)
            z_pert = _unified_force_relax(z_pert, max_iter=80)

            maxiter = int(min(200, (DEADLINE - time.time()) * 20))
            if maxiter < 50:
                break
            z_opt = _unified_optimize(z_pert, max_iter=maxiter)
            valid, val = _unified_valid(z_opt)
            if valid and val > best_sum:
                best_sum = val
                x = z_opt[0::3][:N]
                y = z_opt[1::3][:N]
                r = z_opt[2::3][:N]
                W = z_opt[-1]
                H = 2.0 - W
                best_arr = np.column_stack([x, y, r])
                best_W = W
                best_H = H

    # ---- Phase 4: Polish ----
    if best_arr is not None and time_ok():
        remaining = DEADLINE - time.time()
        if remaining > 10:
            z_best = np.concatenate([best_arr.ravel(), [best_W]])
            z_polished = _unified_polish(z_best, time_budget=remaining - 5)
            valid, val = _unified_valid(z_polished)
            if valid and val > best_sum:
                best_sum = val
                x = z_polished[0::3][:N]
                y = z_polished[1::3][:N]
                r = z_polished[2::3][:N]
                W = z_polished[-1]
                H = 2.0 - W
                best_arr = np.column_stack([x, y, r])
                best_W = W
                best_H = H

    # ---- Final clamping & validation ----
    if best_arr is None:
        W, H = 1.0, 1.0
        best_arr = _grid_layout(5, 5, W, H, shrink=0.98)
        best_W, best_H = W, H

    best_arr = _clamp_valid(best_arr, best_W, best_H)

    s_final = _evaluate(best_arr, best_W, best_H)
    if s_final is None:
        # Fallback grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety shrink
    r = best_arr[:, 2] * (1 - 1e-7)
    best_arr = np.column_stack([best_arr[:, 0], best_arr[:, 1], r])
    best_arr = _clamp_valid(best_arr, best_W, best_H)

    return best_arr

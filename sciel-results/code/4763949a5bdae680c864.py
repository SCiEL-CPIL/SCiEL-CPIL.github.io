import time
import numpy as np
from scipy.optimize import minimize

N = 32

# ---- Deterministic hexagonal seed: per-row column counts so every circle
# ---- fits inside [0,1]^2. 6 rows x ~5-6 cols = 32 circles. ----
def _hex_seed():
    rows = [(0, 6), (1, 5), (2, 6), (3, 5), (4, 6), (5, 5)]
    xs, ys, rs = [], [], []
    for r, (y, n) in enumerate(rows):
        for c in range(n):
            x = (c + 0.5) / n
            yv = (y + 0.5) / 6.0
            xs.append(x); ys.append(yv)
            # gap to each neighbour
            gx = min(x, 1 - x) * 2                      # to vertical wall
            gy = min(yv, 1 - yv) * 2                    # to horizontal wall
            dx = 1.0 / n                                # horiz neighbour spacing
            dy = 1.0 / 6.0                              # vert neighbour spacing
            off = (0.5 / n) if (r % 2 == 1) else 0.0    # hex offset
            ddiag = np.hypot(dx + off, dy)
            gap = min(gx, gy, dx, ddiag)
            rs.append(0.5 * gap - 1e-4)
    x = np.array(xs); y = np.array(ys); r = np.array(rs)
    return x, y, r


def _feasible(x, y, r, eps=1e-9):
    if np.any(r < -eps):
        return False
    if np.any(x - r < -eps) or np.any(1 - x - r < -eps):
        return False
    if np.any(y - r < -eps) or np.any(1 - y - r < -eps):
        return False
    i, j = np.triu_indices(N, 1)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    if np.any(d < r[i] + r[j] - eps):
        return False
    return True


def _score(x, y, r):
    return float(np.sum(r))


def _min_norm_step(x, y, r, max_iter=200, tol=1e-11):
    """Solve the minimum-norm correction so the packing becomes feasible
    while keeping radii fixed.  Returns (x, y) and success flag."""
    n = N
    # boundary: 4 per circle (dir, coef)
    Bdir = np.zeros((4 * n, 2))
    Bcoef = np.zeros(4 * n)
    Bb = np.zeros(4 * n)
    for i in range(n):
        Bdir[4 * i]     = (1.0, 0.0);  Bcoef[4 * i]     = 1.0;  Bb[4 * i]     = -r[i]
        Bdir[4 * i + 1] = (-1.0, 0.0); Bcoef[4 * i + 1] = -1.0; Bb[4 * i + 1] = 1 - r[i]
        Bdir[4 * i + 2] = (0.0, 1.0);  Bcoef[4 * i + 2] = 1.0;  Bb[4 * i + 2] = -r[i]
        Bdir[4 * i + 3] = (0.0, -1.0); Bcoef[4 * i + 3] = -1.0; Bb[4 * i + 3] = 1 - r[i]
    # overlap pairs
    pi, pj = np.triu_indices(n, 1)
    Pcoef = np.zeros((n * (n - 1) // 2, 2 * n))
    Pdir = np.zeros((n * (n - 1) // 2, 2))
    Pb = np.zeros(n * (n - 1) // 2)
    for k in range(len(pi)):
        a, b = pi[k], pj[k]
        Pcoef[k, 2 * a] = 1.0; Pcoef[k, 2 * b] = -1.0
        Pdir[k] = (1.0, 0.0)
        Pb[k] = r[a] + r[b]
    A = np.concatenate([Bdir, Pdir], axis=0)
    coef = np.concatenate([Bcoef, Pcoef])
    b = np.concatenate([Bb, Pb])
    x0 = np.concatenate([x, y])
    lam = np.zeros(len(A))
    for _ in range(max_iter):
        g = A.T @ lam
        Ax = coef @ x0 + A @ g
        res = b - Ax
        # active set: boundary violations (res>0) and overlaps (res>0)
        active = res > 1e-12
        if not np.any(active):
            return x0[:n].copy(), x0[n:].copy(), True
        Aa = A[active]
        ca = coef[active]
        ba = b[active]
        # solve min ||g||  s.t. Aa g + ca x0 = ba  ->  g = Aa^T (Aa Aa^T)^-1 (ba - ca x0)
        try:
            rhs = ba - ca @ x0
            M = Aa @ Aa.T
            w = np.linalg.solve(M + 1e-12 * np.eye(len(Aa)), rhs)
            g = Aa.T @ w
        except np.linalg.LinAlgError:
            g = np.zeros(2 * n)
        lam = np.maximum(lam + g, 0.0)
        if np.linalg.norm(g) < tol:
            break
    x0 = x0 + A.T @ lam
    return x0[:n].copy(), x0[n:].copy(), False


def _grow(x, y, r, budget):
    """Iteratively grow radii, fixing feasibility with min-norm steps."""
    base = r.copy()
    while time.time() < budget:
        r_try = r * 1.004
        # per-circle cap from walls
        cap = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r_try = np.minimum(r_try, cap)
        # feasibility repair
        nx, ny, ok = _min_norm_step(x, y, r_try, max_iter=300, tol=1e-12)
        if ok and _feasible(nx, ny, r_try):
            x, y, r = nx, ny, r_try
        else:
            # try a smaller growth factor
            grown = False
            for f in (1.002, 1.001, 1.0):
                r2 = np.minimum(base * f, cap)
                nx2, ny2, ok2 = _min_norm_step(x, y, r2, max_iter=300, tol=1e-12)
                if ok2 and _feasible(nx2, ny2, r2):
                    x, y, r = nx2, ny2, r2
                    grown = True
                    break
            if not grown:
                break
    return x, y, r


def _polish(x, y, r, budget):
    """Final local polish: maximize sum of radii subject to soft constraints."""
    def obj(p):
        xx = p[:N]; yy = p[N:2*N]; rr = p[2*N:]
        pen = 0.0
        pen += np.sum(np.maximum(0, rr - xx))**2
        pen += np.sum(np.maximum(0, rr - (1 - xx)))**2
        pen += np.sum(np.maximum(0, rr - yy))**2
        pen += np.sum(np.maximum(0, rr - (1 - yy)))**2
        i, j = np.triu_indices(N, 1)
        d = np.hypot(xx[i] - xx[j], yy[i] - yy[j])
        pen += np.sum(np.maximum(0, rr[i] + rr[j] - d))**2
        return -np.sum(rr) + 500.0 * pen
    p0 = np.concatenate([x, y, r])
    lo = np.concatenate([0.0, 0.0, 0.0])
    hi = np.concatenate([1.0, 1.0, 0.5])
    try:
        res = minimize(obj, p0, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                       options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10})
        p = res.x
        nx, ny, nr = p[:N], p[N:2*N], p[2*N:]
        if _feasible(nx, ny, nr) and _score(nx, ny, nr) > _score(x, y, r):
            return nx, ny, nr
    except Exception:
        pass
    return x, y, r


def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    x, y, r = _hex_seed()

    # sanity: seed must be feasible
    if not _feasible(x, y, r):
        # shrink until feasible
        while not _feasible(x, y, r) and r.max() > 1e-6:
            r *= 0.95
        if not _feasible(x, y, r):
            s = 0.5 / 6.0
            gx = np.tile(np.arange(6) / 6.0 + 0.5 / 6.0, 6)
            gy = np.repeat(np.arange(6) / 6.0 + 0.5 / 6.0, 6)
            return np.column_stack([gx[:N], gy[:N], np.full(N, 0.5 / 6.0 - 1e-3)])

    best = _score(x, y, r)
    bx, by, br = x.copy(), y.copy(), r.copy()

    # phase 1: grow
    x, y, r = _grow(x, y, r, deadline - 40.0)
    s = _score(x, y, r)
    if s > best:
        best, bx, by, br = s, x.copy(), y.copy(), r.copy()

    # phase 2: polish
    if time.time() < deadline - 20.0:
        x2, y2, r2 = _polish(bx, by, br, deadline - 20.0)
        s2 = _score(x2, y2, r2)
        if s2 > best:
            best, bx, by, br = s2, x2.copy(), y2.copy(), r2.copy()

    # final safety clamp
    r = np.clip(br, 0.0, 0.5)
    bx = np.clip(bx, r, 1 - r)
    by = np.clip(by, r, 1 - r)
    if _feasible(bx, by, r):
        return np.column_stack([bx, by, r])

    # fallback grid
    s = 0.5 / 6.0
    gx = np.tile(np.arange(6) / 6.0 + 0.5 / 6.0, 6)
    gy = np.repeat(np.arange(6) / 6.0 + 0.5 / 6.0, 6)
    return np.column_stack([gx[:N], gy[:N], np.full(N, 0.5 / 6.0 - 1e-3)])

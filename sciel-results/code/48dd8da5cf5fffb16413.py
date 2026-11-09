"""Fresh design: smooth surrogate (soft-min) + KDTree-accelerated NN + L-BFGS-B + polish."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
_TIME_LIMIT = 150.0
_MARGIN = 8.0  # safety margin in seconds

# ---- helpers ----

def _softmin(x, t):
    """Soft-min of array x with temperature t, stable via log-sum-exp on -x."""
    # soft-min ≈ -t * log(sum(exp(-x/t)))
    z = -x / t
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    return -(zmax * t + t * np.log(s))

def _nearest_min_dist(pts):
    """Min nearest-neighbor distance using KDTree (O(N log N))."""
    tree = cKDTree(pts)
    d, i = tree.query(pts, k=2)  # k=2: self + nearest
    return float(np.min(d[:, 1]))

def _container_margin(pts):
    """min_i (1 - |c_i|)."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    return float(np.min(1.0 - r))

def _true_score(pts):
    """The actual score: min(min_pair_dist/2, min_container_margin)."""
    d_nn = _nearest_min_dist(pts)
    m = _container_margin(pts)
    return min(d_nn / 2.0, m)

def _hex_lattice_init(n):
    """Hexagonal lattice points inside unit disk, return n closest-to-origin."""
    # choose spacing so we get enough points
    for spacing in np.linspace(0.05, 0.12, 200):
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) >= n:
            # keep those well inside (margin) and pick n closest to origin
            r = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[r <= 1.0 - spacing]
            if len(pts) >= n:
                idx = np.argsort(r, kind="stable")[:n]
                return pts[idx]
    # fallback: grid
    g = np.linspace(-0.95, 0.95, 20)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r <= 0.95]
    idx = np.argsort(r, kind="stable")[:n]
    return pts[idx]

# ---- objective ----

def _surrogate_neg(pts, t):
    """Negative soft-min of (pair nearest dists /2) and container margins.
    We approximate the pair term by the nearest-neighbor distance per point,
    which is a lower bound proxy for the global min pair distance and is
    O(N log N) via KDTree. We combine with container margins via soft-min."""
    pts = np.asarray(pts).reshape(-1, 2)
    tree = cKDTree(pts)
    d, i = tree.query(pts, k=2)
    nn = d[:, 1]  # nearest neighbor distance per point
    half = nn / 2.0
    r = np.hypot(pts[:, 0], pts[:, 1])
    margin = 1.0 - r
    # combine both sets into one vector for soft-min
    all_vals = np.concatenate([half, margin])
    sm = _softmin(all_vals, t)
    return -sm

# ---- main ----

def construct():
    t0 = time.time()
    best_pts = _hex_lattice_init(N)
    best_score = _true_score(best_pts)

    def _time_left():
        return _TIME_LIMIT - _MARGIN - (time.time() - t0)

    # --- Phase 1: L-BFGS-B on smooth surrogate, annealed temperature ---
    temps = [0.05, 0.03, 0.015, 0.008, 0.004]
    cur = best_pts.copy()
    for t in temps:
        if _time_left() < 20.0:
            break
        def obj(x):
            return _surrogate_neg(x, t)
        def obj_grad(x):
            # finite difference gradient (central, small h)
            h = 1e-5
            g = np.zeros_like(x)
            for k in range(x.size):
                xp = x.copy(); xp[k] += h
                xm = x.copy(); xm[k] -= h
                g[k] = (obj(xp) - obj(xm)) / (2.0 * h)
            return g
        try:
            res = minimize(
                obj, cur, jac=obj_grad,
                method="L-BFGS-B",
                options=dict(maxiter=400, ftol=1e-10, gtol=1e-8, maxfun=20000),
            )
            if res.success or res.fun < obj(cur):
                cand = res.x.reshape(-1, 2)
                s = _true_score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
                cur = cand
        except Exception:
            pass

    # --- Phase 2: high-precision polish on the EXACT objective ---
    # Exact objective = -min(nn/2, margin). Use subgradient via KDTree + finite diff.
    cur = best_pts.copy()
    def exact_neg(x):
        p = x.reshape(-1, 2)
        return -_true_score(p)
    def exact_grad(x):
        p = x.reshape(-1, 2)
        tree = cKDTree(p)
        d, i = tree.query(p, k=2)
        nn = d[:, 1]
        half = nn / 2.0
        r = np.hypot(p[:, 0], p[:, 1])
        margin = 1.0 - r
        # the min is attained at either a pair (nearest) or a container point.
        # Approximate gradient: for each point, gradient of its own nn/2 and margin.
        # Use finite difference for robustness.
        h = 1e-6
        g = np.zeros_like(x)
        base = exact_neg(x)
        for k in range(x.size):
            xp = x.copy(); xp[k] += h
            xm = x.copy(); xm[k] -= h
            g[k] = (exact_neg(xp) - exact_neg(xm)) / (2.0 * h)
        return g
    # A few short L-BFGS-B polish runs
    for it in range(3):
        if _time_left() < 12.0:
            break
        try:
            res = minimize(
                exact_neg, cur, jac=exact_grad,
                method="L-BFGS-B",
                options=dict(maxiter=300, ftol=1e-12, gtol=1e-10, maxfun=15000),
            )
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _true_score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
                cur = cand
        except Exception:
            pass

    # final validity check
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(~np.isfinite(best_pts)) or np.any(r >= 1.0) or best_pts.shape != (N, 2):
        # fall back to a guaranteed-valid lattice
        best_pts = _hex_lattice_init(N)
    return best_pts

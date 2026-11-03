"""Symmetry-constrained smooth epigraph optimizer for 300-circle packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
_BASE = 100          # circles per 120-degree sector (3-fold symmetry)
_EPS = 1e-12

_DEADLINE = 110.0    # seconds; leave a safety margin under the 120 s limit
_start = None


def _build_initial():
    """3-fold symmetric initial guess: 100 base circles in a hex grid,
    restricted to the 120-degree sector, then replicated by rotation."""
    base = None
    for spacing in np.linspace(0.10, 0.04, 60):
        s3 = spacing * np.sqrt(3.0) / 2.0
        k = int(1.0 / spacing) + 3
        base_pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * s3
                base_pts.append((x, y))
        base_pts = np.array(base_pts)
        r = np.hypot(base_pts[:, 0], base_pts[:, 1])
        base_pts = base_pts[r <= 1.0 - spacing / 2.0]
        # keep only points in the 120-degree sector: angle in [0, 2pi/3)
        ang = np.arctan2(base_pts[:, 1], base_pts[:, 0])
        ang = np.mod(ang, 2.0 * np.pi)
        base_pts = base_pts[ang < 2.0 * np.pi / 3.0]
        if len(base_pts) >= _BASE:
            # take the _BASE points closest to the origin for balance
            order = np.argsort(r, kind="stable")
            base = base_pts[order[:_BASE]]
            break
    
    if base is None or len(base) < _BASE:
        # Fallback: random points in sector
        # This is a rough fallback, but ensures we have points
        base = np.random.rand(_BASE, 2)
        base[:, 0] *= 0.8
        base[:, 1] *= 0.8
        # Rotate to sector? Just use random for now, optimizer will fix
        # Better fallback: hex grid without sector filter, take first 100
        spacing = 0.1
        s3 = spacing * np.sqrt(3.0) / 2.0
        k = 15
        base_pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * s3
                base_pts.append((x, y))
        base_pts = np.array(base_pts)
        r = np.hypot(base_pts[:, 0], base_pts[:, 1])
        base_pts = base_pts[r <= 0.9]
        if len(base_pts) >= _BASE:
            base = base_pts[:_BASE]
        else:
            # If still not enough, pad with random
            needed = _BASE - len(base_pts)
            rand_pts = np.random.rand(needed, 2) * 0.5
            base = np.vstack([base_pts, rand_pts])

    # replicate by 120 and 240 degree rotations
    th = 2.0 * np.pi / 3.0
    R1 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    R2 = np.array([[np.cos(2 * th), -np.sin(2 * th)], [np.sin(2 * th), np.cos(2 * th)]])
    pts = np.vstack([base, base @ R1.T, base @ R2.T])
    # enforce strict interior
    rr = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts * np.minimum(1.0, (1.0 - 1e-4) / np.maximum(rr, _EPS))[:, None]
    return pts


def _score(pts):
    """Largest feasible common radius for a set of centres."""
    n = len(pts)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d.min() / 2.0
    bound = 1.0 - np.hypot(pts[:, 0], pts[:, 1]).max()
    return min(pair, bound)


def construct():
    global _start
    _start = time.time()
    pts = _build_initial()
    n = len(pts)
    x0 = np.concatenate([pts.ravel(), [_score(pts) + 1e-3]])

    def objective(x):
        p = x[:2 * n].reshape(n, 2)
        t = x[2 * n]
        # pairwise distances
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        # boundary distances
        bound = 1.0 - np.hypot(p[:, 0], p[:, 1])
        
        # Flatten d (off-diagonal) and bound
        # We need to be careful with the soft-min calculation
        # Let's use the log-sum-exp trick on the negative values (for min)
        
        # d_flat is size n^2 - n
        # We can't easily use matrix ops for the objective value itself without constructing the full vector
        # But for gradient, we can.
        
        # For objective, just compute the soft-min value
        scale = 300.0
        # Get off-diagonal elements
        idx = ~np.eye(n, dtype=bool)
        d_off = d[idx]
        
        a = (d_off - t) / scale
        b = (bound - t) / scale
        m = np.concatenate([a, b])
        
        mmax = m.max()
        # Handle case where mmax is inf or nan?
        if not np.isfinite(mmax):
            return 0.0
            
        lse = mmax - np.log(np.sum(np.exp(-(m - mmax))))
        soft_min = lse / scale
        return soft_min + 1e-6 * t

    def objective_grad(x):
        p = x[:2 * n].reshape(n, 2)
        t = x[2 * n]
        
        # Pairwise distances
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1.0) # Avoid div by 0 in gradient, weight will be 0
        
        # Boundary distances
        bound = 1.0 - np.hypot(p[:, 0], p[:, 1])
        
        scale = 300.0
        
        # We need weights w_k for each constraint k.
        # Constraints: d_ij - t (for i<j) and bound_i - t (for i)
        
        # Let's construct the full matrix of "values" V
        # V_ij = d_ij - t for i != j
        # V_ii = -inf (mask out)
        # V_bound_i = bound_i - t
        
        # To compute weights efficiently:
        # w_ij proportional to exp(-(V_ij - S))
        # S = max(V)
        
        # Get off-diagonal d
        idx = ~np.eye(n, dtype=bool)
        d_off = d[idx]
        
        a = (d_off - t) / scale
        b = (bound - t) / scale
        m = np.concatenate([a, b])
        
        mmax = m.max()
        if not np.isfinite(mmax):
            return np.zeros_like(x)
            
        exp_m = np.exp(-(m - mmax))
        w_sum = exp_m.sum()
        w = exp_m / w_sum
        
        w_pair = w[:len(d_off)]
        w_bound = w[len(d_off):]
        
        # Gradient of soft-min wrt p
        # g_p = -1/scale * ( sum_{i<j} w_ij * grad(d_ij) + sum_i w_bound_i * grad(bound_i) )
        
        # grad(d_ij) wrt p_i = (p_i - p_j) / d_ij
        # grad(d_ij) wrt p_j = (p_j - p_i) / d_ij
        
        # We need to sum w_ij * (p_i - p_j) / d_ij over j for each i.
        # Let's create a matrix W_pair_mat of size n x n
        # W_pair_mat[i, j] = w_ij if i != j, else 0
        # Note: w_pair is flattened from d_off.
        # d_off is d[idx]. idx is row-major.
        # So w_pair[k] corresponds to d[i, j] where i, j are the k-th off-diagonal element.
        
        W_mat = np.zeros((n, n))
        W_mat[idx] = w_pair
        
        # K_mat[i, j] = W_mat[i, j] / d[i, j]
        # d[i, i] = 1, W_mat[i, i] = 0, so K_mat[i, i] = 0
        K_mat = W_mat / d
        
        # grad_i = sum_j K_mat[i, j] * (p_i - p_j)
        # = p_i * sum_j K_mat[i, j] - sum_j K_mat[i, j] * p_j
        # = p_i * row_sum(K_mat) - K_mat @ p
        
        row_sums = K_mat.sum(axis=1)
        grad_pair = p * row_sums[:, None] - K_mat @ p
        
        # Gradient of bound_i wrt p_i
        # bound_i = 1 - sqrt(p_ix^2 + p_iy^2)
        # grad = -p_i / |p_i|
        r = np.hypot(p[:, 0], p[:, 1])
        r_safe = np.maximum(r, _EPS)
        grad_bound_vec = -p / r_safe[:, None]
        
        # Weighted sum of bound gradients
        grad_bound = np.sum(w_bound[:, None] * grad_bound_vec, axis=0)
        
        # Total gradient wrt p
        g_p = -(grad_pair + grad_bound) / scale
        
        # Gradient wrt t
        # soft_min = (S - LSE)/scale
        # d(soft_min)/dt = -1/scale * (dS/dt - dLSE/dt)
        # S = max(m). dS/dt = 1 if the max element is active?
        # In soft-min, the derivative of the value wrt t is -1/scale * sum(w_k * d m_k / dt)
        # m_k = (val_k - t)/scale. d m_k / dt = -1/scale.
        # So d soft_min / dt = -1/scale * sum(w_k * (-1/scale))?
        # Wait.
        # F = (S - LSE)/scale.
        # S = max(m).
        # LSE = log(sum(exp(-(m-S)))).
        # dF/dt = (1/scale) * (dS/dt - dLSE/dt).
        # This is tricky because S and LSE both depend on t.
        # Actually, simpler:
        # F is a weighted average of (val_k - t) with weights w_k, scaled by 1/scale?
        # No, soft-min is not a linear combination.
        # However, for the epigraph formulation:
        # We are minimizing F(x, t).
        # The term 1e-6 * t in objective encourages t to increase?
        # No, we want to MAXIMIZE t.
        # The objective is `soft_min + 1e-6 * t`.
        # `soft_min` approximates `min(val_k - t)`.
        # If we minimize `min(val_k - t)`, we are maximizing `t - min(val_k)`.
        # Let's check the sign.
        # If `min(val_k - t) > 0`, then `t < min(val_k)`.
        # We want `t` to be as large as possible, so we want `min(val_k - t)` to be close to 0 from below?
        # No, if `t` is the radius, constraints are `val_k >= 2t` (for pairs, `d_ij >= 2t` so `d_ij/2 - t >= 0`) and `1-|c_i| >= t`.
        # Let `g_k = val_k - t`. We want `g_k >= 0`.
        # We want to maximize `t`.
        # Equivalent to minimize `-t` subject to `g_k >= 0`.
        # Or minimize `max(0, -g_k)`?
        # The soft-min of `g_k` is positive if all `g_k` are positive.
        # If we minimize `soft_min(g_k)`, we push `g_k` down towards 0.
        # This means `t` increases until `g_k` hits 0.
        # So minimizing `soft_min` is correct for finding the max `t`.
        
        # What is `d(soft_min)/dt`?
        # Let `m_k = g_k / scale`.
        # `soft_min = (S - LSE)/scale`.
        # `S = max(m)`. `LSE = log(sum(exp(-(m-S))))`.
        # `dS/dt = (1/scale) * d(max(g_k))/dt`.
        # `dLSE/dt = ...`
        
        # Actually, there is a known result:
        # The gradient of the soft-min of `f_i(x)` wrt `x` is `sum w_i grad f_i`.
        # Here `f_k(x, t) = (val_k(x) - t) / scale`.
        # `grad_x soft_min = sum w_k grad_x f_k = sum w_k (grad_x val_k) / scale`.
        # `grad_t soft_min = sum w_k grad_t f_k = sum w_k (-1/scale) = -1/scale`.
        
        # So `g_t = -1/scale`.
        # Plus the gradient of `1e-6 * t`, which is `1e-6`.
        # So `g_t = -1/scale + 1e-6`.
        
        g_t = -1.0 / scale + 1e-6
        
        return np.concatenate([g_p.ravel(), [g_t]])

    best = pts.copy()
    best_r = _score(pts)

    def _deadline_hit():
        return (time.time() - _start) > _DEADLINE

    # main optimization
    for it in range(1, 8):
        if _deadline_hit():
            break
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           jac=objective_grad,
                           options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8,
                                    "maxls": 40})
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = res.x[:2 * n].reshape(n, 2)
                rr = _score(cand)
                if np.isfinite(rr) and rr > best_r:
                    best_r = rr
                    best = cand.copy()
                    x0 = res.x.copy()
        except Exception:
            break

    # polish pass: tighter, more iterations, from best
    if not _deadline_hit():
        x0 = np.concatenate([best.ravel(), [best_r + 1e-3]])
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           jac=objective_grad,
                           options={"maxiter": 1500, "ftol": 1e-15, "gtol": 1e-10,
                                    "maxls": 60})
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = res.x[:2 * n].reshape(n, 2)
                rr = _score(cand)
                if np.isfinite(rr) and rr > best_r:
                    best_r = rr
                    best = cand.copy()
        except Exception:
            pass

    # final safety: ensure strictly inside and finite
    rr = np.hypot(best[:, 0], best[:, 1])
    best = best * np.minimum(1.0, (1.0 - 1e-6) / np.maximum(rr, _EPS))[:, None]
    return best

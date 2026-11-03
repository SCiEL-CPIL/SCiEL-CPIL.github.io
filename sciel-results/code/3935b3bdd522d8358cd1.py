"""Optimized packing of 300 equal circles in a unit circle using scipy L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 118.0  # evaluator kills at 150s; keep a safety margin


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    # Best known r ~ 0.05315 -> spacing ~ 0.1063. Use a slightly finer grid.
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # fallback: enlarge grid
        return _hex_init_fallback(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _hex_init_fallback(n=N):
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    if n < 2:
        return 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])) if n > 0 else 1.0
    
    # pairwise distances
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _optimize(pts, start_time, time_limit):
    """Optimize using scipy L-BFGS-B on a smooth epigraph formulation."""
    n = len(pts)
    pts = pts.copy()
    
    # Ensure valid start
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
        
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Initial guess for t
    t0 = best_score * 0.99
    
    def objective_and_grad(z):
        nonlocal pts
        # z is [t, x1, y1, x2, y2, ...]
        t = z[0]
        p = z[1:].reshape(n, 2)
        
        # Hard projection to keep circles inside container during optimization
        # This ensures the boundary constraint is strictly satisfied
        r = np.hypot(p[:, 0], p[:, 1])
        mask = r > 1.0
        if np.any(mask):
            p[mask] = p[mask] / r[mask, None]
        
        # Compute distances
        diff = p[:, None, :] - p[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1e10) # Avoid division by zero for diagonal
        d = np.sqrt(d2)
        
        # Boundary distances
        b = 1.0 - r
        
        # We want to maximize t subject to d_ij >= 2t and b_i >= t.
        # This is equivalent to minimizing:
        #  -t + sum( max(0, 2t - d_ij)^2 ) + sum( max(0, t - b_i)^2 )
        
        # Penalty terms
        # Pairwise
        viol_pair = np.maximum(0.0, 2.0 * t - d)
        # Boundary
        viol_bound = np.maximum(0.0, t - b)
        
        # Objective: -t + 0.5 * (sum(viol_pair^2) + sum(viol_bound^2))
        # Using 0.5 to simplify gradient
        
        obj = -t + 0.5 * (np.sum(viol_pair * viol_pair) + np.sum(viol_bound * viol_bound))
        
        # Gradient
        grad = np.zeros_like(z)
        
        # d(obj)/dt = -1 + sum( 2 * viol_pair * 2 ) + sum( 2 * viol_bound * 1 )
        # Wait, d/dt (0.5 * v^2) = v * dv/dt
        # v_pair = max(0, 2t - d). dv_pair/dt = 2 if 2t > d else 0.
        # So term is viol_pair * 2.
        # v_bound = max(0, t - b). dv_bound/dt = 1 if t > b else 0.
        # So term is viol_bound * 1.
        
        # Let's recompute carefully.
        # L = -t + 0.5 * (sum_{i<j} (max(0, 2t - d_ij))^2 + sum_i (max(0, t - b_i))^2)
        
        # dL/dt = -1 + sum_{i<j} 2 * viol_pair_ij * (d/dt (2t - d_ij)) + sum_i viol_bound_i * (d/dt (t - b_i))
        # d/dt (2t - d_ij) = 2 (since d_ij doesn't depend on t)
        # d/dt (t - b_i) = 1
        
        # So dL/dt = -1 + 4 * sum(viol_pair) + sum(viol_bound)
        
        # dL/dp_i:
        # From pairwise term:
        # 0.5 * sum_{j != i} (max(0, 2t - d_ij))^2
        # Derivative wrt p_i:
        # sum_{j != i} 2 * viol_pair_ij * (d/dp_i (2t - d_ij))
        # = sum_{j != i} 2 * viol_pair_ij * (- d/dp_i d_ij)
        # d/dp_i d_ij = (p_i - p_j) / d_ij
        # So term = - sum_{j != i} 2 * viol_pair_ij * (p_i - p_j) / d_ij
        
        # From boundary term:
        # 0.5 * (max(0, t - b_i))^2
        # Derivative wrt p_i:
        # viol_bound_i * (d/dp_i (t - b_i))
        # b_i = 1 - r_i. d/dp_i b_i = - (p_i / r_i)
        # d/dp_i (t - b_i) = p_i / r_i
        # So term = viol_bound_i * (p_i / r_i)
        
        # Combine:
        # Grad_p_i = - sum_{j != i} 2 * viol_pair_ij * (p_i - p_j) / d_ij + viol_bound_i * (p_i / r_i)
        
        # Vectorized computation
        # d is (n,n)
        # viol_pair is (n,n)
        
        # We need sum_j 2 * viol_pair[i,j] * (p_i - p_j) / d_ij
        # Note: viol_pair[i,i] is 0 (since d=1e10, 2t - 1e10 < 0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vec = diff / d[:, :, None] # (n, n, 2), unit[i,j] = (p_i - p_j) / d_ij
        unit_vec = np.where(np.isfinite(unit_vec), unit_vec, 0.0)
        
        # Weighted sum
        # w_pair[i,j] = 2 * viol_pair[i,j]
        w_pair = 2.0 * viol_pair
        grad_p_pair = np.sum(w_pair[:, :, None] * unit_vec, axis=1) # (n, 2)
        
        # Boundary gradient
        with np.errstate(divide='ignore', invalid='ignore'):
            inward_vec = p / r[:, None] # (n, 2)
        inward_vec = np.where(np.isfinite(inward_vec), inward_vec, 0.0)
        
        grad_p_bound = viol_bound[:, None] * inward_vec
        
        grad_p = -grad_p_pair + grad_p_bound
        
        grad[0] = -1.0 + 4.0 * np.sum(viol_pair) + np.sum(viol_bound)
        grad[1:] = grad_p.reshape(-1)
        
        return obj, grad

    # Define objective wrapper for minimize
    def objective_only(z):
        return objective_and_grad(z)[0]
        
    def grad_only(z):
        return objective_and_grad(z)[1]

    # Initial z
    z0 = np.concatenate(([t0], pts.flatten()))
    
    # Bounds: t can be anything (though physically limited), p coordinates in [-1, 1]
    # Actually, we project p inside, so bounds on p are less critical but good for L-BFGS
    bounds = [(t0 * 0.5, 1.0)] + [(-1.0, 1.0)] * (2 * n)
    
    # We will run minimize in a loop, checking time
    current_z = z0.copy()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Run L-BFGS-B for a limited number of iterations or until convergence
        # maxiter=50 per call to allow frequent time checks
        res = minimize(
            objective_only,
            current_z,
            jac=grad_only,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success:
            # Check if we converged
            current_z = res.x
            # Extract pts and score
            new_pts = current_z[1:].reshape(n, 2)
            # Re-apply projection just in case
            r = np.hypot(new_pts[:, 0], new_pts[:, 1])
            out = r > 1.0
            if np.any(out):
                new_pts[out] = new_pts[out] / r[out, None]
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            else:
                # If no improvement, we might be stuck or converged
                # Try one more time with tighter tolerance? Or break?
                # L-BFGS-B usually stops when it can't improve.
                break
        else:
            # If it didn't succeed (e.g. maxiter reached), continue to next chunk
            current_z = res.x
            
            # Check time again
            if time.time() - start_time > time_limit:
                break

    return best_pts, best_score


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    # ensure valid start: clip inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    base_score = _score(pts)
    best_pts = pts.copy()
    best_score = base_score

    try:
        opt_pts, opt_score = _optimize(pts, start_time, TIME_LIMIT)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    except Exception:
        pass

    # final safety: ensure finite, inside, shape (N,2)
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        # rebuild
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]

    return best_pts

"""Improved program for packing 300 circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _hex_lattice_init(n):
    """Generate hexagonal lattice points and select n closest to center."""
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    s *= 0.95
    
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
    
    points = np.array(points)
    idx = np.argsort(points[:, 2])
    selected = points[idx[:n]][:, :2]
    
    if len(selected) < n:
        remaining = n - len(selected)
        extra = np.random.randn(remaining, 2) * 0.3
        selected = np.vstack([selected, extra])
    
    noise = np.random.randn(n, 2) * 0.001
    selected = selected + noise
    
    return selected


def _ring_init(n):
    """Initialize with concentric rings."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    count = 1
    
    ring = 1
    while count < n:
        # Number of points on this ring
        # Approximate circumference / spacing
        # spacing ~ s from hex lattice
        s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.95
        radius = ring * s * np.sqrt(3.0) / 2.0 * 2 / np.sqrt(3)  # approx
        # Better: radius such that points fit
        # Let's just use a standard ring packing approach
        # For ring k, radius r_k, number of points m_k
        # We want to fill the disk.
        # Approx radius of ring k is proportional to k.
        # Let's use r_k = k * r_1 where r_1 is chosen to fit.
        # Total area ~ pi * R_max^2.
        # Let's just iterate.
        
        # Simpler: use hex lattice as primary, ring as secondary if needed.
        # Actually, let's stick to hex lattice as it's proven good.
        break
        
    return _hex_lattice_init(n)


def _compute_score(pts):
    """Compute the packing score (minimum radius)."""
    n = len(pts)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    radii = np.sqrt(np.sum(pts**2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary)


def _soft_min_obj_and_grad(pts, beta):
    """
    Compute negative soft-min objective and its gradient.
    Returns (value, grad).
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    radii_sq = np.sum(pts**2, axis=1)
    radii = np.sqrt(radii_sq)
    boundary_dists = 1.0 - radii
    
    pair_vals = dists / 2.0
    bound_vals = boundary_dists
    
    ii, jj = np.triu_indices(n, k=1)
    pair_vals_upper = pair_vals[ii, jj]
    
    all_vals = np.concatenate([pair_vals_upper, bound_vals])
    all_vals = np.maximum(all_vals, 1e-12)
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_val + np.log(sum_exp)
    
    obj_val = log_sum_exp / beta
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    dists_upper = dists[ii, jj]
    safe_dists = np.maximum(dists_upper, 1e-10)
    w_pairs = weights[:len(ii)]
    
    A = np.zeros((n, n))
    A[ii, jj] = w_pairs / (2.0 * safe_dists)
    A[jj, ii] = A[ii, jj]
    
    row_sums = np.sum(A, axis=1)
    A_P = A @ pts
    grad_pairs = - pts * row_sums[:, np.newaxis] + A_P
    
    # Boundary contributions
    w_bounds = weights[len(ii):]
    safe_radii = np.maximum(radii, 1e-10)
    grad_bounds = w_bounds[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    
    return obj_val, grad


def _epigraph_polish(pts, max_iters=50):
    """
    Epigraph formulation polish:
    Maximize t
    Subject to:
      |p_i - p_j| >= 2t  for all i < j
      |p_i| <= 1 - t      for all i
    
    We use SLSQP or similar. But SLSQP is slow for many constraints.
    Instead, we use a penalty method or active set approach.
    Here we use a simple iterative refinement:
    1. Compute current min constraint value t.
    2. Identify active constraints.
    3. Solve a QP-like system to move points to increase t.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        # Compute constraints
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - radii
        
        ii, jj = np.triu_indices(n, k=1)
        pair_vals = dists[ii, jj] / 2.0
        bound_vals = boundary_dists
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        t = np.min(all_vals)
        
        if t < 1e-15:
            break
        
        # Active constraints (within 1e-8 of t)
        active_mask = all_vals < t + 1e-8
        active_pairs = np.where(active_mask[:len(ii)])[0]
        active_bounds = np.where(active_mask[len(ii):])[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # We want to increase t.
        # For each active pair (i,j): grad t wrt p_i = (p_i - p_j) / (2 * dist_ij)
        # For each active bound i: grad t wrt p_i = -p_i / |p_i|
        
        # We solve for delta_p such that:
        # grad_v_k . delta_p >= 1  for all active k
        # This is a feasibility problem. We can use a least-squares approach.
        # Minimize ||delta_p||^2 subject to G @ delta_p >= 1
        # Where G is the matrix of gradients.
        
        # Build G matrix
        m = len(active_pairs) + len(active_bounds)
        if m == 0:
            break
            
        G = np.zeros((m, 2 * n))
        
        for idx in active_pairs:
            i, j = ii[idx], jj[idx]
            d = dists[i, j]
            if d < 1e-10:
                continue
            direction = (pts[i] - pts[j]) / d
            G[idx, 2*i:2*i+2] = direction / 2.0
            G[idx, 2*j:2*j+2] = -direction / 2.0
        
        for idx in active_bounds:
            i = active_bounds[idx]
            r = radii[i]
            if r < 1e-10:
                continue
            G[len(active_pairs) + idx, 2*i:2*i+2] = -pts[i] / r
        
        # Solve min ||x||^2 s.t. G x >= 1
        # This is equivalent to projecting 0 onto the polyhedron G x >= 1.
        # We can use scipy.optimize.linprog or a simple iterative method.
        # For simplicity, we use a scaled gradient step.
        
        # Approximate solution: x = G^T (G G^T)^{-1} 1
        # This minimizes ||x||^2 subject to G x = 1 (relaxation of >=)
        
        try:
            GGt = G @ G.T
            # Add regularization for stability
            GGt += 1e-10 * np.eye(m)
            one_vec = np.ones(m)
            coeffs = np.linalg.solve(GGt, one_vec)
            delta = G.T @ coeffs
            delta = delta.reshape(n, 2)
        except np.linalg.LinAlgError:
            break
        
        # Normalize delta to have small magnitude
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-10:
            break
        delta = delta / delta_norm
        
        # Line search
        best_step = 0.0
        best_t = t
        for step in [0.01, 0.05, 0.1, 0.2, 0.5]:
            new_pts = pts + step * delta
            new_t = _compute_score(new_pts)
            if new_t > best_t:
                best_t = new_t
                best_step = step
        
        if best_step == 0.0:
            break
            
        pts = pts + best_step * delta
    
    return pts


def _local_polish(pts, max_iters=100, tol=1e-10):
    """
    Local optimization using exact min with steepest descent.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 2:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - radii
        
        ii, jj = np.triu_indices(n, k=1)
        pair_vals = dists[ii, jj] / 2.0
        bound_vals = boundary_dists
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        min_val = np.min(all_vals)
        
        if min_val < tol:
            break
        
        active_mask = np.abs(all_vals - min_val) < 1e-6
        active_pairs = np.where(active_mask[:len(ii)])[0]
        active_bounds = np.where(active_mask[len(ii):])[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        grad = np.zeros_like(pts)
        
        if len(active_pairs) > 0:
            for idx in active_pairs:
                i, j = ii[idx], jj[idx]
                d = dists[i, j]
                if d < 1e-10:
                    continue
                direction = (pts[i] - pts[j]) / d
                grad[i] += direction / (2.0 * len(active_pairs))
                grad[j] -= direction / (2.0 * len(active_pairs))
        
        if len(active_bounds) > 0:
            for idx in active_bounds:
                i = active_bounds[idx]
                r = radii[i]
                if r < 1e-10:
                    continue
                direction = -pts[i] / r
                grad[i] += direction / (2.0 * len(active_bounds))
        
        grad_norm = np.sqrt(np.sum(grad**2))
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        best_step = 0.0
        best_obj = min_val
        step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        for step in step_sizes:
            new_pts = pts + step * grad
            new_score = _compute_score(new_pts)
            if new_score > best_obj:
                best_obj = new_score
                best_step = step
        
        if best_step == 0.0:
            for step in [0.0001, 0.0005]:
                new_pts = pts + step * grad
                new_score = _compute_score(new_pts)
                if new_score > best_obj:
                    best_obj = new_score
                    best_step = step
        
        if best_step == 0.0:
            break
        
        pts = pts + best_step * grad
    
    return pts


def construct():
    """Construct 300 circle centers in unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Initialize with hexagonal lattice
    pts = _hex_lattice_init(N)
    
    radii = np.sqrt(np.sum(pts**2, axis=1))
    mask = radii < 1.0
    if np.sum(mask) < N:
        pts = pts * (0.95 / np.max(radii))
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Adaptive beta schedule
    beta_schedule = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    for beta in beta_schedule:
        if time.time() > _DEADLINE - 10:
            break
        
        def obj(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val
        
        def grad(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            _, g = _soft_min_obj_and_grad(x, beta)
            return g.ravel()
        
        try:
            result = minimize(
                obj,
                pts.ravel(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 500
                }
            )
            
            new_pts = result.x.reshape(N, 2)
            
            radii = np.sqrt(np.sum(new_pts**2, axis=1))
            if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
        except Exception:
            pass
    
    # Final polish with high beta
    if time.time() < _DEADLINE - 15:
        for beta in [20000, 50000]:
            if time.time() > _DEADLINE - 5:
                break
            
            def obj(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                val, _ = _soft_min_obj_and_grad(x, beta)
                return val
            
            def grad(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                _, g = _soft_min_obj_and_grad(x, beta)
                return g.ravel()
            
            try:
                result = minimize(
                    obj,
                    best_pts.ravel(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxfun': 300
                    }
                )
                
                new_pts = result.x.reshape(N, 2)
                radii = np.sqrt(np.sum(new_pts**2, axis=1))
                if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                    new_score = _compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
            except Exception:
                pass
    
    # Epigraph polish
    if time.time() < _DEADLINE - 10:
        try:
            polished = _epigraph_polish(best_pts, max_iters=30)
            radii = np.sqrt(np.sum(polished**2, axis=1))
            if np.all(radii < 1.0) and np.all(np.isfinite(polished)):
                polished_score = _compute_score(polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished.copy()
        except Exception:
            pass
    
    # Local polish with exact min
    if time.time() < _DEADLINE - 3:
        try:
            polished = _local_polish(best_pts, max_iters=100, tol=1e-12)
            radii = np.sqrt(np.sum(polished**2, axis=1))
            if np.all(radii < 1.0) and np.all(np.isfinite(polished)):
                polished_score = _compute_score(polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END

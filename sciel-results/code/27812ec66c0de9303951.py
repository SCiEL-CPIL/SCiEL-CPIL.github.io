"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(pts):
    """Compute the packing radius: min of half-min-distance and min(1-|c_i|)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist_sq = dx * dx + dy * dy
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    # Distance to boundary
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    
    return min(min_dist / 2.0, min_boundary)

def _soft_min_obj(pts, beta=50.0):
    """
    Objective function for optimization: minimize -soft_min(constraints).
    Constraints:
    1. d_ij >= 2r  ->  (d_ij^2 - (2r)^2) / (2r)^2 >= 0 is not directly usable for soft min of r.
    We want to maximize r such that all constraints hold.
    Equivalent to minimizing -r.
    
    Let's define the objective as minimizing the violation.
    Actually, a common approach is to maximize the minimum of:
    - min_ij (d_ij / 2)
    - min_i (1 - |c_i|)
    
    We can use a soft-min of the "gaps".
    Gap_ij = d_ij / 2
    Gap_i = 1 - |c_i|
    
    We want to maximize min(Gaps).
    Let f(pts) = -soft_min(Gaps, beta).
    Minimizing f is equivalent to maximizing soft_min(Gaps).
    As beta -> inf, soft_min -> min.
    """
    pts = pts.reshape(N, 2)
    
    # Pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    
    gaps_dist = dist / 2.0
    
    # Boundary gaps
    r = np.hypot(pts[:, 0], pts[:, 1])
    gaps_boundary = 1.0 - r
    
    # Combine all gaps into one vector
    # To avoid memory issues with 300*299/2 pairs, we process in chunks or just use the full vector if memory allows.
    # 300*300 = 90,000 elements. This is small.
    all_gaps = np.concatenate([gaps_dist.flatten(), gaps_boundary.flatten()])
    
    # Soft min: min(x) approx -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow, shift by min
    min_gap = np.min(all_gaps)
    exp_terms = np.exp(-beta * (all_gaps - min_gap))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_gap - log_sum / beta
    
    # We want to maximize soft_min_val, so we minimize -soft_min_val
    return -soft_min_val

def _soft_min_grad(pts, beta=50.0):
    """
    Gradient of the soft-min objective.
    """
    pts = pts.reshape(N, 2)
    
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    
    gaps_dist = dist / 2.0
    
    r = np.hypot(pts[:, 0], pts[:, 1])
    gaps_boundary = 1.0 - r
    
    # Weights for soft min
    # w_k = exp(-beta * x_k) / sum(exp(-beta * x_j))
    # Shift by min for stability
    all_gaps = np.concatenate([gaps_dist.flatten(), gaps_boundary.flatten()])
    min_gap = np.min(all_gaps)
    exp_terms = np.exp(-beta * (all_gaps - min_gap))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Gradient of soft_min wrt x_k is -w_k
    # We are minimizing -soft_min, so gradient is w_k
    
    # Gradient wrt points for distance gaps
    # d(gap_ij)/d(pts_i) = (pts_i - pts_j) / d_ij
    # d(gap_ij)/d(pts_j) = (pts_j - pts_i) / d_ij
    
    grad = np.zeros_like(pts)
    
    # Reshape weights for distances
    w_dist = weights[:N*N].reshape(N, N)
    w_bound = weights[N*N:]
    
    # Compute gradient contribution from distances
    # For each pair i,j, weight w_ij
    # grad_i += w_ij * (pts_i - pts_j) / d_ij
    # grad_j += w_ij * (pts_j - pts_i) / d_ij
    
    # Vectorized approach:
    # dx/dy are already computed.
    # inv_dist = 1 / dist (with inf -> 0)
    inv_dist = np.zeros_like(dist)
    mask = dist > 1e-10
    inv_dist[mask] = 1.0 / dist[mask]
    
    # grad_dist[i] = sum_j w_ij * (pts_i - pts_j) / d_ij
    # (pts_i - pts_j) is dx, dy.
    # We need sum_j w_ij * dx[i,j] * inv_dist[i,j]
    
    # w_dist * inv_dist
    w_inv = w_dist * inv_dist
    
    # grad_x[i] = sum_j w_inv[i,j] * dx[i,j]
    # grad_y[i] = sum_j w_inv[i,j] * dy[i,j]
    
    grad_x = np.sum(w_inv * dx, axis=1)
    grad_y = np.sum(w_inv * dy, axis=1)
    
    # Gradient from boundary
    # gap_i = 1 - |c_i|
    # d(gap_i)/d(c_i) = -c_i / |c_i|
    # grad_i += w_bound[i] * (-c_i / |c_i|)
    
    r_safe = np.where(r > 1e-10, r, 1.0)
    boundary_grad_x = -w_bound * (pts[:, 0] / r_safe)
    boundary_grad_y = -w_bound * (pts[:, 1] / r_safe)
    
    grad[:, 0] = grad_x + boundary_grad_x
    grad[:, 1] = grad_y + boundary_grad_y
    
    return grad.flatten()

def _generate_hex_init(n):
    """Generate hexagonal lattice points clipped to unit disk, select n closest to center."""
    spacing = 0.10
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < n:
        # Fallback: random points
        return np.random.uniform(-0.9, 0.9, size=(n, 2))

    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return points[idx]

def _polish(pts, time_limit):
    """
    Local refinement using L-BFGS-B on the exact min-distance objective.
    Since L-BFGS-B requires a smooth objective, we use the soft-min with high beta
    or a custom gradient for the exact min.
    However, exact min is non-differentiable.
    We can use a "tightening" strategy:
    1. Start with soft-min beta=50.
    2. Increase beta iteratively.
    """
    start_time = time.time()
    pts = pts.copy().flatten()
    
    best_score = _compute_score(pts.reshape(N, 2))
    best_pts = pts.reshape(N, 2).copy()
    
    betas = [20, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Define objective and gradient for current beta
        def obj(x):
            return _soft_min_obj(x, beta)
        
        def grad(x):
            return _soft_min_grad(x, beta)
        
        try:
            res = minimize(
                obj, 
                pts, 
                jac=grad, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_pts = res.x.reshape(N, 2)
            
            # Ensure points are inside
            r = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask_out = r >= 1.0
            if np.any(mask_out):
                new_pts[mask_out] *= (0.999 / r[mask_out])
                
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = res.x
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    """
    Construct N=300 equal circles in unit disk.
    """
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s kill
    
    # 1. Generate multiple initial configurations
    # We will run a few "annealing" runs from different starts.
    
    # Start 1: Hex grid
    pts_hex = _generate_hex_init(N)
    
    # Start 2: Random
    np.random.seed(42)
    pts_rand = np.random.uniform(-0.95, 0.95, size=(N, 2))
    
    # Ensure random points are not too close (basic repulsion)
    # Simple relaxation
    for _ in range(50):
        dx = pts_rand[:, 0:1] - pts_rand[:, 0:1].T
        dy = pts_rand[:, 1:2] - pts_rand[:, 1:2].T
        dist = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dist, np.inf)
        # Repulsive force
        force = np.zeros_like(pts_rand)
        mask = dist < 0.1
        # Normalize
        inv_d = np.zeros_like(dist)
        inv_d[mask] = 1.0 / dist[mask]
        # Sum forces
        fx = np.sum((dx * inv_d) * mask, axis=1)
        fy = np.sum((dy * inv_d) * mask, axis=1)
        pts_rand[:, 0] += 0.01 * fx
        pts_rand[:, 1] += 0.01 * fy
        # Project back
        r = np.hypot(pts_rand[:, 0], pts_rand[:, 1])
        mask_out = r > 0.95
        pts_rand[mask_out] *= 0.95 / r[mask_out]

    candidates = [pts_hex, pts_rand]
    
    best_pts = None
    best_score = -np.inf
    
    # Optimize each candidate
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        if remaining < 10:
            break
            
        # Phase 1: Global optimization with soft-min (low beta)
        # We can do a few steps of gradient descent with annealing here if time permits,
        # but L-BFGS-B with soft-min is usually quite good from a decent start.
        
        # Let's do a quick "anneal" by running L-BFGS-B with increasing beta
        polished_pts, score = _polish(init_pts, remaining - 5)
        
        if score > best_score:
            best_score = score
            best_pts = polished_pts
            
    # If we have time left, do a final polish on the best result with even higher beta
    if best_pts is not None:
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        if remaining > 5:
            # Final high-precision polish
            final_pts, final_score = _polish(best_pts, remaining - 2)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score

    # Final validation
    if best_pts is None:
        best_pts = pts_hex # Fallback
        
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    for i in range(len(best_pts)):
        if r[i] >= 1.0:
            best_pts[i] *= 0.999 / r[i]
            
    return best_pts
# EVOLVE-BLOCK-END

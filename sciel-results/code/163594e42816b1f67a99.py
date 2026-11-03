"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _remaining_time():
    return max(0.0, TIME_LIMIT - (time.time() - _START_TIME))


def _hex_init(n):
    """Generate initial hexagonal lattice positions inside unit disk."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98


def _compute_score(pts):
    """Compute the packing radius: min of pairwise distances/2 and boundary distances."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def _soft_min(x, beta):
    """Soft minimum of array x using log-sum-exp trick."""
    # x is 1D array
    m = np.min(x)
    return m - np.log(np.sum(np.exp(beta * (x - m)))) / beta


def _optimize(pts, n_steps, beta_start, beta_end, lr_start, lr_end, time_budget):
    """Gradient ascent on soft-min objective with annealed beta and learning rate."""
    pts = pts.copy()
    n = len(pts)
    
    for step in range(n_steps):
        if _remaining_time() < time_budget:
            break
        
        # Interpolate beta and learning rate
        frac = step / max(1, n_steps - 1)
        beta = beta_start + (beta_end - beta_start) * frac
        lr = lr_start + (lr_end - lr_start) * frac
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Soft minimum of pairwise distances
        pair_min = _soft_min(dists, beta)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        bound_dists = 1.0 - radii
        bound_min = _soft_min(bound_dists, beta)
        
        # Objective: min of the two soft-mins
        # Use soft-min of the two values
        obj = _soft_min(np.array([pair_min, bound_min]), beta * 2)
        
        # Gradient with respect to pts
        # Gradient of soft-min of distances
        # d/dx_i soft_min_j d_ij = -sum_j (d_ij * exp(-beta*d_ij) / sum_k exp(-beta*d_ik)) * (x_i - x_j) / d_ij
        
        # Pairwise gradient
        # Weight matrix: w_ij = exp(-beta * d_ij) / sum_k exp(-beta * d_ik)
        with np.errstate(over='ignore', under='ignore'):
            exp_vals = np.exp(-beta * dists)
        np.fill_diagonal(exp_vals, 0)
        row_sums = np.sum(exp_vals, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)
        weights = exp_vals / row_sums
        
        # Gradient: grad_i = sum_j w_ij * (x_i - x_j) / d_ij
        # This is the gradient of -soft_min, so we want to maximize soft_min, 
        # so we move in direction of -gradient of -soft_min = gradient of soft_min
        # Actually, we want to maximize the minimum distance, so we want to increase soft_min
        # The gradient of soft_min w.r.t. x_i is:
        # sum_j w_ij * (x_j - x_i) / d_ij  (this points in direction to increase min)
        
        # Compute sum_j w_ij * (x_j - x_i) / d_ij
        # (x_j - x_i) / d_ij is the unit vector from i to j
        unit_vecs = diff / np.maximum(dists, 1e-12)[:, :, np.newaxis]
        np.fill_diagonal(unit_vecs, 0)
        
        # grad_i = sum_j weights[i,j] * unit_vecs[i,j]
        grad_pair = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary gradient: for each point, gradient of soft_min(1 - |x_i|)
        # d/dx_i (1 - |x_i|) = -x_i / |x_i|
        # Soft-min gradient similar
        with np.errstate(over='ignore', under='ignore'):
            exp_bound = np.exp(-beta * bound_dists)
        bound_sums = np.sum(exp_bound, axis=0, keepdims=True)
        bound_sums = np.maximum(bound_sums, 1e-30)
        bound_weights = exp_bound / bound_sums
        
        # For boundary, each point only contributes to its own boundary distance
        # So grad_bound_i = -x_i / |x_i| * (weight for point i)
        # But in soft-min over all points, the weight for point i is exp(-beta*(1-|x_i|))/sum
        # Actually, soft_min_i (1 - |x_i|) has gradient:
        # sum_i w_i * (-x_i/|x_i|) where w_i = exp(-beta*(1-|x_i|))/sum_j exp(-beta*(1-|x_j|))
        # But this is a scalar soft-min, not per-point. Let me reconsider.
        
        # Actually, the boundary constraint is min_i (1 - |c_i|), which is a scalar.
        # Its soft approximation is soft_min_i (1 - |c_i|)
        # Gradient w.r.t. c_k: sum_i w_i * d/dc_k (1 - |c_i|) = w_k * (-c_k/|c_k|)
        # where w_i = exp(-beta*(1-|c_i|)) / sum_j exp(-beta*(1-|c_j|))
        
        bound_grad = np.zeros_like(pts)
        for i in range(n):
            if radii[i] > 1e-10:
                bound_grad[i] = bound_weights[i] * (-pts[i] / radii[i])
        
        # Combine gradients: we want to maximize min(pair_min, bound_min)
        # Use soft-min of the two, gradient is weighted combination
        # For simplicity, combine with equal weights or use the gradient of the combined soft-min
        # Let's just add them with appropriate scaling
        grad = grad_pair + bound_grad
        
        # Normalize gradient to avoid instability
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Update
        pts = pts + lr * grad
        
        # Keep points inside the disk (soft projection)
        new_radii = np.hypot(pts[:, 0], pts[:, 1])
        too_far = new_radii > 0.999
        if np.any(too_far):
            pts[too_far] *= 0.999 / new_radii[too_far]
    
    return pts


def construct():
    """Main entry point: construct 300 circle centers in unit disk."""
    pts = _hex_init(N)
    
    # Phase 1: Coarse optimization with moderate beta
    if _remaining_time() > 30:
        pts = _optimize(pts, n_steps=2000, beta_start=50, beta_end=200, 
                        lr_start=0.001, lr_end=0.0001, time_budget=20)
    
    # Phase 2: Fine polish with high beta
    if _remaining_time() > 20:
        pts = _optimize(pts, n_steps=3000, beta_start=200, beta_end=500,
                        lr_start=0.0001, lr_end=0.00001, time_budget=15)
    
    # Final validation: ensure all points are strictly inside
    radii = np.hypot(pts[:, 0], pts[:, 1])
    mask = radii < 0.9999
    if np.sum(mask) < N:
        # Fallback: return hex lattice
        pts = _hex_init(N)
    
    return pts[:N]
# EVOLVE-BLOCK-END

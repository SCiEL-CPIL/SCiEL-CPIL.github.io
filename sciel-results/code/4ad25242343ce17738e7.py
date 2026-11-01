"""
Packing 300 equal circles in a unit circle.
Strategy:
1. Initialize with a hexagonal lattice cut to the unit disk, keeping the N=300 closest-to-center points.
2. Use a repulsion-based optimization (soft-potential gradient descent / simulated annealing hybrid)
   to maximize the minimum distance.
   - The objective is to maximize the minimum of:
       a) pairwise distances / 2
       b) (1 - |c_i|)
   - We maximize the sum of soft-minimum of these terms, or simply use a force-based approach.
   - A common effective method is to maximize the sum of inverse distances (repulsion) while
     applying a strong attraction to the center and a hard constraint penalty for boundary violations.
   - Actually, a more direct approach for "maximize minimum distance" is to use a
     "soft minimum" objective: f(r) = -sum(1/(d_ij + eps)) - sum(1/(1 - |c_i| + eps))
     and minimize this (i.e., maximize the distances).
   - Alternatively, use a gradient ascent on the minimum distance with smoothing.
   
   Let's use a robust repulsion + centering force with simulated annealing-like temperature decay.
   Forces:
   - Repulsion between circles: F_rep = k_rep * (1/d^2) * direction (if d < threshold)
   - Boundary force: F_bnd = k_bnd * (1 - |c|) if near boundary? No, we want to push away from boundary if |c| > 1 - r.
     Actually, the constraint is |c_i| + r <= 1. So we want to keep |c_i| <= 1 - r.
     Since r is determined by the minimum distance, it's coupled.
     
   Better approach:
   Maximize the minimum of all pairwise distances and (1 - |c_i|).
   Let d_min = min( min_{i<j} |c_i - c_j|, min_i (2*(1 - |c_i|)) )  -- wait, radius is min(dist/2, 1-|c|).
   So we want to maximize r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) ).
   
   Let's define a score function S = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) ).
   To optimize this, we can use a "soft" version:
   S_soft = ( sum_{i<j} (|c_i - c_j|/2)^p + sum_i (1 - |c_i|)^p )^(1/p) for small p, or use log-sum-exp.
   
   A simpler and very effective method for circle packing is the "repulsion" method:
   - Initialize positions.
   - Iterate:
     - Compute all pairwise distances.
     - For each pair (i,j), if distance d_ij < 2*r_current (or some threshold), push them apart.
     - For each circle i, if 1 - |c_i| < r_current, push it towards the center.
     - Use a decaying step size.
   
   However, determining r_current dynamically is tricky.
   
   Let's use a standard gradient-based optimization on a smooth objective:
   Objective: Maximize J = sum_{i<j} log(d_ij) + sum_i log(1 - |c_i|)
   This is the "max-entropy" or "determinant of Gram matrix" approach, which tends to maximize the minimum distance.
   Gradient of log(d_ij) with respect to c_i is (c_i - c_j) / d_ij^2.
   Gradient of log(1 - |c_i|) with respect to c_i is -c_i / (|c_i| * (1 - |c_i|)).
   
   We can use gradient ascent with a decaying learning rate.
   We must ensure that 1 - |c_i| > 0, so we need to handle the boundary carefully.
   If a point goes outside, we project it back or apply a strong penalty.
   
   Let's implement:
   1. Hex lattice init.
   2. Gradient ascent on J = sum log(d_ij) + sum log(1 - |c_i|) with a small epsilon to avoid log(0).
   3. Periodically compute the actual score r = min(min_dist/2, min(1-|c|)).
   4. Keep the best configuration.
   5. Use a deadline guard.
"""

import numpy as np
import time

N = 300
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin from 150s

def _hex_lattice_init(n):
    """Initialize with hexagonal lattice points closest to origin."""
    # Estimate spacing
    # Area of unit circle = pi. Area per hex point ~ (sqrt(3)/2) * s^2.
    # n * (sqrt(3)/2) * s^2 ~ pi  => s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    # Generate a grid of hex points
    # Range of indices
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
    
    points = np.array(points)
    # Sort by distance to origin
    idx = np.argsort(points[:, 2])
    points = points[idx[:n]]
    return points[:, :2]

def _compute_score(points):
    """Compute the maximum radius r for the given points."""
    if len(points) < 2:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - radii)
    
    r = min(min_dist / 2.0, min_boundary_dist)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return r

def _objective_grad(points):
    """
    Compute gradient of J = sum_{i<j} log(d_ij + eps) + sum_i log(1 - |c_i| + eps)
    We want to maximize J, so we move in the direction of the gradient.
    """
    n = len(points)
    grad = np.zeros_like(points)
    eps = 1e-6
    
    # Pairwise repulsion term: sum_{i<j} log(d_ij)
    # grad_i = sum_{j != i} (c_i - c_j) / d_ij^2
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = points[i] - points[j]
            d_sq = np.dot(d_vec, d_vec)
            if d_sq < 1e-12:
                # Add small random jitter to avoid singularity
                d_vec += np.random.randn(2) * 1e-6
                d_sq = np.dot(d_vec, d_vec)
            # Gradient of log(sqrt(d_sq)) = 1/2 log(d_sq)
            # d/dx_i (1/2 log((x_i - x_j)^2)) = (x_i - x_j) / (x_i - x_j)^2
            grad[i] += d_vec / d_sq
            grad[j] -= d_vec / d_sq
            
    # Boundary term: sum_i log(1 - |c_i|)
    # grad_i = -c_i / (|c_i| * (1 - |c_i|))
    for i in range(n):
        r = np.linalg.norm(points[i])
        if r > 1.0 - eps:
            # If too close to boundary, apply a very strong inward force
            # The log term goes to -inf, so gradient magnitude goes to +inf (inward)
            # To avoid numerical issues, clamp the denominator
            denom = max(eps, 1.0 - r)
            if r < eps:
                grad[i] += np.random.randn(2) * 1e-3 # Random push if at origin
            else:
                grad[i] -= points[i] / (r * denom)
        else:
            if r < eps:
                grad[i] += np.random.randn(2) * 1e-3
            else:
                grad[i] -= points[i] / (r * (1.0 - r))
                
    return grad

def construct():
    global START_TIME, TIME_LIMIT
    
    # 1. Initialize
    points = _hex_lattice_init(N)
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return points
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Optimization loop
    # Use a decaying learning rate
    # Start with a moderate lr
    lr = 0.01
    decay = 0.999
    momentum = 0.5
    vel = np.zeros_like(points)
    
    # Pre-compute pairs for efficiency? N=300, N^2=90000. 
    # Vectorized gradient is faster. Let's try to vectorize the pairwise part.
    
    def _vectorized_grad(points):
        n = len(points)
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero, we'll mask it later
        
        # Gradient of log(dist) = 0.5 * log(dist_sq)
        # d/dx_i = (x_i - x_j) / dist_sq
        # grad_i = sum_j (x_i - x_j) / dist_sq_ij
        # We can compute this as:
        # grad[i] = sum_j diff[i, j] / dist_sq[i, j]
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        dist_sq_safe = np.where(mask, dist_sq, 1.0)
        
        # grad_pairwise[i] = sum_{j != i} diff[i, j] / dist_sq[i, j]
        # diff[i, j] is (points[i] - points[j])
        # So grad_pairwise = sum_j (diff[i, j] / dist_sq[i, j])
        
        # Note: diff[i, j] / dist_sq[i, j] is a vector of shape (n, n, 2)
        # We sum over j (axis 1)
        
        # To save memory, we can do it in chunks or just use the loop if N is small enough.
        # N=300, 300x300x2 floats is ~ 1.8MB. It's fine.
        
        inv_dist_sq = 1.0 / dist_sq_safe
        inv_dist_sq[~mask] = 0.0 # Zero out diagonal
        
        # grad_pairwise[i] = sum_j (points[i] - points[j]) * inv_dist_sq[i, j]
        # = points[i] * sum_j inv_dist_sq[i, j] - sum_j points[j] * inv_dist_sq[i, j]
        
        sum_inv = np.sum(inv_dist_sq, axis=1) # shape (n,)
        # sum_j points[j] * inv_dist_sq[i, j]
        # This is a matrix product: inv_dist_sq @ points
        sum_pts_weighted = inv_dist_sq @ points # shape (n, 2)
        
        grad_pairwise = points * sum_inv[:, np.newaxis] - sum_pts_weighted
        
        # Boundary gradient
        radii = np.linalg.norm(points, axis=1)
        # grad_bnd[i] = -points[i] / (radii[i] * (1 - radii[i]))
        # Handle radii ~ 0
        safe_radii = np.where(radii > 1e-6, radii, 1.0)
        denom = safe_radii * (1.0 - radii)
        # If 1 - radii <= 0, we are outside. We want to push in.
        # The log term is undefined. Let's add a penalty term for boundary violation in the objective.
        # Actually, let's just clamp the denominator to a small positive number to create a strong force.
        denom_safe = np.where(denom > 1e-6, denom, 1e-6)
        # If radii > 1, 1-radii is negative. We want the force to be inward.
        # -points[i] / (radii * (1-radii))
        # If 1-radii < 0, the term is positive, so -points / positive = -points (inward). Correct.
        # But if radii is close to 1, denom is small, force is large.
        
        grad_bnd = -points / denom_safe[:, np.newaxis]
        
        return grad_pairwise + grad_bnd

    # Optimization loop
    iter_count = 0
    while time.time() - START_TIME < TIME_LIMIT - 5:
        # Compute gradient
        grad = _vectorized_grad(points)
        
        # Update velocity (momentum)
        vel = momentum * vel + lr * grad
        points = points + vel
        
        # Keep points inside the unit circle (hard constraint projection)
        # If a point is outside, move it back to the boundary
        radii = np.linalg.norm(points, axis=1)
        outside = radii > 1.0
        if np.any(outside):
            # Project back to the circle
            # We want |c| <= 1. But we also need space for the circle radius.
            # For now, just project to |c| = 1 - epsilon? No, just |c| = 1 is enough for validity,
            # but it will reduce the score. Better to project to a safe radius.
            # Let's project to radius 0.999 to be safe.
            points[outside] *= (0.999 / radii[outside, np.newaxis])
            # Zero out velocity for these points to prevent immediate re-violation?
            vel[outside] = 0.0
        
        # Decay learning rate
        lr *= decay
        
        # Compute score periodically
        if iter_count % 50 == 0:
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        iter_count += 1
        
    # Final check
    final_score = _compute_score(points)
    if final_score > best_score:
        best_points = points.copy()
        
    # Ensure valid output
    # Check for non-finite values
    if not np.all(np.isfinite(best_points)):
        return _hex_lattice_init(N)
        
    return best_points

# Keep the original function name and signature
# The task requires: construct() -> array of shape (N, 2)

# We need to make sure the global variables are handled correctly if construct is called multiple times
# but the task implies a single call per evaluation.

# Let's refine the initialization to be a bit tighter if possible.
# The hex lattice init is good.

# One issue: the vectorized gradient might be slow if not optimized.
# N=300, 300^2 = 90k operations. Vectorized numpy should handle this in ~1ms.
# 150s / 1ms = 150k iterations. That's plenty.

# Let's add a simple check to ensure we don't return points that are too close to the boundary
# such that the radius is negative or zero.

# The _compute_score function already handles this.

# Final verification of the code structure.

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Score: {_compute_score(pts)}")

"""
Rewritten program: Hexagonal lattice initialization + Gradient Ascent with Soft Min Objective.

Strategy:
1. Initialize centers using a hexagonal lattice to approximate the densest packing.
2. Define an objective function that approximates the minimum distance (to boundary and between circles)
   using a differentiable "soft minimum" (e.g., sum of inverse distances or power sums).
3. Perform gradient ascent to maximize this objective, effectively pushing circles apart and towards the center.
4. Use adaptive step sizes and a strict time limit to ensure completion within 120s.
"""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START

def _hex_lattice_init():
    """
    Generate a hexagonal lattice of points inside the unit disk.
    Returns the N points closest to the origin.
    """
    # Theoretical spacing for 300 points in area pi:
    # Area per point ~ pi/300. Hex cell area ~ sqrt(3)/2 * s^2.
    # s ~ sqrt(2*pi / (300*sqrt(3))) ~ 0.1099
    # We try a few spacings to ensure we get at least N points, 
    # but keep the lattice tight.
    
    best_points = None
    # Start with a slightly tighter lattice to ensure enough candidates
    for s in [0.105, 0.108, 0.110, 0.112]:
        k = int(1.0 / s) + 5
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        
        if len(points) >= N:
            points = np.array(points)
            # Keep the N closest to center
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            
            # We want the tightest packing, so smaller s is generally better 
            # if it still provides N points. 
            # However, we just want *a* good start. The first valid one is usually fine.
            if best_points is None:
                best_points = pts
    
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: Uniform random inside disk (very bad, but prevents crash)
    # Or a fine grid
    s = 0.05
    k = int(1.0 / s) + 5
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _compute_min_dist(points):
    """
    Compute the actual minimum distance between any two circles and to the boundary.
    This is the true score metric.
    """
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    # Use a chunked approach or vectorized if memory allows. 
    # 300x300 matrix is 90,000 floats ~ 720KB, which is fine.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _optimize(points, time_limit):
    """
    Optimize the packing using gradient ascent on a soft-min objective.
    """
    start_time = time.time()
    pts = points.copy()
    current_score = _compute_min_dist(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    # Objective function: We want to maximize the minimum distance.
    # A common smooth approximation is -sum(1/d_i^p) or -sum(d_i^-p).
    # Let's use a soft-min based on the sum of inverse distances.
    # Minimize F(pts) = sum(1/d_ij^2) + sum(1/(1-|p_i|)^2)
    # This pushes points away from each other and away from the boundary.
    
    def objective_and_grad(pts):
        # Distances to boundary
        r = np.hypot(pts[:, 0], pts[:, 1])
        d_bound = 1.0 - r
        # Avoid division by zero or negative
        d_bound_safe = np.maximum(d_bound, 1e-6)
        
        # Pairwise distances
        # To save memory/time, we can compute gradients analytically.
        # Let's compute the gradient of the sum of inverse squared distances.
        
        grad = np.zeros_like(pts)
        
        # Boundary term: d( (1-|p|)^-2 ) / dp = -2(1-|p|)^-3 * d(1-|p|)/dp
        # d(1-|p|)/dp = -p/|p|
        # So term is -2 * (1-|p|)^-3 * (-p/|p|) = 2 * p / (|p| * (1-|p|)^3)
        # We want to MINIMIZE this, so we subtract this gradient.
        # Wait, we are doing GRADIENT ASCENT on the "goodness".
        # Let's define Goodness = - ( Sum(1/d_ij^2) + Sum(1/d_bound^2) )
        # We want to maximize Goodness.
        # Gradient of Goodness = - Gradient(Penalty)
        
        # Penalty P = Sum_ij 1/d_ij^2 + Sum_i 1/d_bound_i^2
        # Grad P wrt p_i:
        # Boundary part: d/dp_i [ (1-|p_i|)^-2 ] = -2(1-|p_i|)^-3 * (-p_i/|p_i|) = 2 p_i / (|p_i| (1-|p_i|)^3)
        # Pairwise part: Sum_j!=i d/dp_i [ 1/|p_i-p_j|^2 ] = Sum_j!=i -2 (p_i-p_j) / |p_i-p_j|^4
        
        # Let's compute P and Grad P, then step in direction -Grad P.
        
        # 1. Boundary Gradient
        r_safe = np.maximum(r, 1e-6)
        # d_bound is 1-r.
        # Term in P: (1-r)^-2. Deriv wrt r: -2(1-r)^-3 * (-1) = 2(1-r)^-3.
        # Deriv wrt p: 2(1-r)^-3 * (p/r).
        grad_bound = 2.0 * (d_bound_safe ** -3) * (pts / r_safe[:, np.newaxis])
        
        # 2. Pairwise Gradient
        # diff[i,j] = p_i - p_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists_sq, 1.0) # Placeholder, we'll mask it
        
        # Term: 1/d^2. Deriv wrt p_i: -2 (p_i - p_j) / d^4
        # We need Sum_j -2 (p_i - p_j) / |p_i - p_j|^4
        # Vectorized:
        # inv_d4 = 1 / dists_sq^2
        # contrib = -2 * diff * inv_d4[:, :, np.newaxis]
        # grad_pair = sum(contrib, axis=1)
        
        # To save memory, let's do it in chunks if N is large, but 300 is small.
        inv_d4 = 1.0 / (dists_sq ** 2)
        np.fill_diagonal(inv_d4, 0.0) # Zero out self-interaction
        
        # contrib[i,j] = -2 * (p_i - p_j) * inv_d4[i,j]
        # grad_pair[i] = sum_j contrib[i,j]
        # Using einsum or matmul?
        # diff is (N, N, 2). inv_d4 is (N, N).
        # We can multiply diff by inv_d4[:, :, np.newaxis] and sum along axis 1.
        # This creates a (N, N, 2) array, which is 300*300*2 = 180k floats. Fine.
        
        weighted_diff = diff * (inv_d4 * -2.0)[:, :, np.newaxis]
        grad_pair = np.sum(weighted_diff, axis=1)
        
        total_grad = grad_bound + grad_pair
        
        # Objective value (Penalty)
        # Sum of 1/d^2
        pair_sum = np.sum(inv_d4) # Note: diagonal is 0, so this is sum over i!=j
        # Wait, the penalty is sum 1/d^2. 
        # My inv_d4 was 1/d^4. 
        # Let's re-calculate penalty value for monitoring, though we only need grad for step.
        # Actually, for step size adaptation, we might want the magnitude of the penalty.
        
        # We are minimizing P. Gradient Ascent on -P means step = -grad_P.
        # So we want to move in direction -total_grad.
        
        return total_grad

    # Adaptive Gradient Ascent
    step_size = 0.01
    patience = 0
    max_patience = 50
    lr_decay = 0.95
    
    while time.time() - start_time < time_limit:
        # Compute gradient
        grad = objective_and_grad(pts)
        
        # Normalize gradient to control step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        # Step direction is -grad (since we minimize penalty)
        direction = -grad / grad_norm
        
        # Perform step
        new_pts = pts + step_size * direction
        
        # Enforce boundary constraint strictly (projection)
        r_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask_out = r_new > 1.0
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / r_new[mask_out, np.newaxis] * (1.0 - 1e-6)
            
        # Check if valid (no overlaps, inside)
        # The objective handles overlaps via penalty, but we should ensure we don't explode.
        # Let's compute the true score to track best.
        new_score = _compute_min_dist(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            patience = 0
            # If improved, maybe increase step size slightly?
            # step_size = min(step_size * 1.05, 0.1)
        else:
            pts = new_pts.copy() # Accept the move even if score drops? 
            # In gradient ascent, we usually accept if it reduces the penalty (increases soft score).
            # Here we are using true score for "best", but the gradient is based on soft score.
            # If true score drops, it's likely because of a non-smooth transition or local trap.
            # Let's just keep the points but increase patience.
            patience += 1
            
            if patience > max_patience:
                # Reduce step size
                step_size *= lr_decay
                patience = 0
                
                # If step size is too small, break
                if step_size < 1e-6:
                    break
                    
                # Optional: Random jitter to escape local minima
                if np.random.rand() < 0.1:
                    pts += np.random.normal(0, 0.001, pts.shape)
                    # Re-apply boundary
                    r_jit = np.hypot(pts[:, 0], pts[:, 1])
                    mask_jit = r_jit > 1.0
                    if np.any(mask_jit):
                        pts[mask_jit] = pts[mask_jit] / r_jit[mask_jit, np.newaxis] * (1.0 - 1e-6)

    return best_pts

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s
    
    # 1. Initialize
    points = _hex_lattice_init()
    
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * math.cos(angle)
            points[i, 1] = r * math.sin(angle)
            
    init_score = _compute_min_dist(points)
    
    # 2. Optimize
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.5:
        remaining = time_limit - elapsed - 5.0
        if remaining > 10:
            points = _optimize(points, remaining)
    
    # 3. Final Validity Check and Cleanup
    # Ensure points are inside
    r = np.hypot(points[:, 0], points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        points[mask] = points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
        
    # Ensure no coincident points (should be handled by optimization, but just in case)
    # We don't strictly need to fix this if the score is low, but it makes the candidate invalid if distance is 0.
    # The optimizer should prevent distance 0.
    
    return points

# EVOLVE-BLOCK-END

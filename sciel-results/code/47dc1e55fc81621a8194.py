import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _hexagonal_init(n):
    """Generate n points in a hexagonal lattice centered at origin."""
    # Generate a grid of hexagonal points
    # Basis vectors for hexagonal packing
    a = 1.0
    b = np.sqrt(3) / 2.0
    
    points = []
    # Generate enough points to have at least n
    # Radius in grid units
    R = 10
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            x = i * a + j * 0.5 * a
            y = j * b
            points.append((x, y))
    
    points = np.array(points)
    # Sort by distance to origin
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n]]
    
    # Center the points
    points -= np.mean(points, axis=0)
    
    # Scale to fit in [-0.5, 0.5] with some margin
    # Max coordinate is roughly 0.5 - r
    # Let's scale so that max abs coord is 0.45
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        points *= 0.45 / max_coord
        
    return points

def _objective_and_grad(flat_positions, r):
    """
    Objective: Sum of squared overlaps between circles.
    flat_positions: array of shape (2*N,)
    r: current radius
    Returns: (objective_value, gradient)
    """
    positions = flat_positions.reshape((N, 2))
    
    # Calculate pairwise distances
    # diff[i, j] = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Overlap constraint: dist >= 2*r
    # Violation: max(0, 2*r - dist)^2
    violations = np.zeros((N, N))
    diffs = 2 * r - dist
    # Only consider where diffs > 0
    active = diffs > 0
    violations[active] = (diffs[active]) ** 2
    
    obj = np.sum(violations)
    
    # Gradient calculation
    # d(obj)/d(positions[i])
    # If active[i,j]:
    # d/dx_i (2r - dist_ij)^2 = 2 * (2r - dist_ij) * d(2r - dist_ij)/dx_i
    # = 2 * (2r - dist_ij) * (-(x_i - x_j) / dist_ij)
    # = -2 * (2r - dist_ij) / dist_ij * (x_i - x_j)
    
    grad = np.zeros((N, 2))
    # We can vectorize this somewhat, but for N=50, a loop is fast enough
    # Or use broadcasting
    # Let's try to compute gradient via broadcasting
    # factor[i,j] = -2 * (2r - dist_ij) / dist_ij for active pairs, 0 otherwise
    factor = np.zeros((N, N))
    # Avoid division by zero
    safe_dist = np.where(dist > 1e-10, dist, 1.0)
    factor[active] = -2 * (2 * r - dist[active]) / safe_dist[active]
    
    # grad[i] = sum_j factor[i,j] * (positions[i] - positions[j])
    # diff[i,j] is positions[i] - positions[j]
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Flatten gradient
    return obj, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, 20s margin
    
    # Initial guess
    r = 0.05
    positions = _hexagonal_init(N)
    
    # Flatten positions for optimizer
    flat_pos = positions.flatten()
    
    # Bounds for positions given radius r
    # x, y in [-0.5 + r, 0.5 - r]
    
    best_r = 0.0
    best_pos = flat_pos.copy()
    
    # First, find a feasible r for the initial hexagonal grid
    # We can binary search r or just try to optimize for a specific r
    
    # Let's try to maximize r by iterative increase
    # Start with a small r and increase
    
    current_r = 0.05
    # Check if current positions are valid for current_r
    # We will use the optimizer to adjust positions to fit current_r
    
    # Optimization loop
    while time.time() < deadline:
        # Try to increase r
        # Step size
        delta_r = 0.001
        new_r = current_r + delta_r
        
        # Set bounds
        bounds = [(-0.5 + new_r, 0.5 - new_r)] * (2 * N)
        
        # Optimize positions
        try:
            res = minimize(
                _objective_and_grad, 
                flat_pos, 
                args=(new_r,), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e-6:
                # Feasible
                current_r = new_r
                flat_pos = res.x
                if current_r > best_r:
                    best_r = current_r
                    best_pos = flat_pos.copy()
                # If we made good progress, keep going
            else:
                # Not feasible, try smaller step or break
                if delta_r < 1e-4:
                    break
                delta_r /= 2
                # Optionally, try a random restart if stuck
                if np.random.random() < 0.1:
                    # Perturb positions
                    flat_pos = flat_pos + np.random.normal(0, 0.01, size=flat_pos.shape)
                    # Clip to bounds
                    for i in range(2 * N):
                        flat_pos[i] = np.clip(flat_pos[i], -0.5 + current_r, 0.5 - current_r)
        except Exception:
            break
            
    # Convert best_pos back to (N, 2) array
    best_positions = best_pos.reshape((N, 2))
    
    # Final validation check
    # Ensure all points are inside
    for i in range(N):
        for k in range(2):
            if abs(best_positions[i, k]) > 0.5 - best_r:
                best_positions[i, k] = np.clip(best_positions[i, k], -0.5 + best_r, 0.5 - best_r)
                
    # Re-calculate actual radius to be safe
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < min_dist:
                min_dist = d
    for i in range(N):
        for k in range(2):
            d_wall = 0.5 - abs(best_positions[i, k])
            if d_wall < min_dist:
                min_dist = d_wall
                
    actual_r = min_dist / 2.0
    
    # If actual_r is significantly smaller than best_r, use actual_r
    # The returned array is the centers. The score is calculated by the evaluator.
    # We just need to return the centers.
    
    return best_positions

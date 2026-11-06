import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    def fibonacci_disk(n):
        """Generate n points in a disk using the Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            r = np.sqrt((i + 0.5) / n) * 0.98  # Scale to fit inside unit circle
            theta = i * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        return np.array(points)

    # 1. Initialization: Fibonacci Disk
    points = fibonacci_disk(N)
    best_points = points.copy()
    best_r = compute_radius(best_points)

    # 2. Optimization: Force-based relaxation
    # We use a force-based approach to relax the configuration.
    # The force is derived from a potential that penalizes overlaps.
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 50
    relax_steps_per_iter = 100

    while time.time() < deadline:
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Use a soft repulsion force: F = 1 / d^2 for d < 2*r_target
        # r_target is the current best radius, slightly increased to push for improvement
        r_target = best_r * 1.001
        d_limit = 2.0 * r_target
        
        if r_target > 1e-6:
            mask = dists < d_limit
            # Force magnitude: 1/d^2, but scaled to avoid explosion
            # We only apply force where mask is True
            factor = np.zeros_like(dists)
            factor[mask] = 1.0 / (dists[mask] ** 2)
            
            # Ensure diagonal is 0
            np.fill_diagonal(factor, 0.0)
            
            # forces[i] = sum_j factor[i,j] * diff[i,j]
            forces += np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_overlap = np.maximum(0.0, radii + r_target - 1.0)
        b_mask = boundary_overlap > 0
        if np.any(b_mask):
            # Direction: -pts / radii
            # Factor: boundary_overlap / radii
            b_factor = np.zeros_like(radii)
            b_factor[b_mask] = boundary_overlap[b_mask] / radii[b_mask]
            forces -= points * b_factor[:, np.newaxis]
        
        # Update positions
        points += step_size * forces
        
        # Safety: Ensure points don't fly away
        max_r = np.max(np.hypot(points[:, 0], points[:, 1]))
        if max_r > 2.0:
            points = points / max_r * 1.0
        
        # Check if improved
        current_r = compute_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb to escape local minimum
                perturb_scale = 0.005
                points += np.random.normal(0, perturb_scale, (N, 2))
                
                # Ensure validity
                radii = np.hypot(points[:, 0], points[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / radii[outside, np.newaxis] * 0.999
                
                no_improve_count = 0
                step_size *= 0.8

    # 3. Final Polish: Local Optimization with L-BFGS-B
    # Use the best configuration found so far as the starting point.
    # We will maximize the minimum distance, which is equivalent to maximizing r.
    # We can use a smooth approximation of the minimum distance.
    
    def objective(x):
        """Objective function to minimize. We want to maximize the minimum distance.
        So we minimize -min_distance.
        """
        pts = x.reshape(N, 2)
        # Compute pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Compute boundary distance
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # The radius is the minimum of these two
        r = min(min_pair, min_boundary)
        
        # We want to maximize r, so we minimize -r
        return -r

    # Start from best_points
    x0 = best_points.flatten()
    
    # Define bounds: points must be inside the unit circle
    # This is a non-convex constraint, so we use L-BFGS-B with penalty or just rely on the objective.
    # A better approach is to use a penalty for boundary violations.
    
    def penalized_objective(x):
        pts = x.reshape(N, 2)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_violation = np.sum(np.maximum(0, radii - 1.0)**2)
        
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        min_boundary = np.min(1.0 - radii)
        r = min(min_pair, min_boundary)
        
        return -r + 10.0 * boundary_violation

    # Run L-BFGS-B
    try:
        result = minimize(penalized_objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
        
        if result.success or result.fun < -best_r:
            new_points = result.x.reshape(N, 2)
            new_r = compute_radius(new_points)
            if new_r > best_r and np.all(np.hypot(new_points[:, 0], new_points[:, 1]) < 1.0):
                best_r = new_r
                best_points = new_points
    except Exception:
        pass

    # Ensure validity
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_points)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_points = best_points * scale
        
    return best_points

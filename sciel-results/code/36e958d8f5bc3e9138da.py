"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def _fibonacci_lattice(n):
    """Generate n points in [0,1)^2 using Fibonacci lattice."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / golden_ratio
    i = np.arange(n)
    # Fractional part of i / phi
    frac = (i * inv_phi) % 1.0
    x = frac
    y = (i * 0.5) % 1.0
    points = np.column_stack([x, y])
    return points

def _score(points):
    """Compute the packing score (largest common radius)."""
    # Distance between all pairs
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundaries (square is [-0.5, 0.5]^2)
    boundary_dists = np.min(np.abs(0.5 - np.abs(points)), axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_dist / 2.0, min_boundary)

def _optimize_from_initial(points, time_limit=100):
    """Optimize the packing using gradient-based optimization."""
    start_time = time.time()
    best_score = _score(points)
    best_points = points.copy()
    
    # Use L-BFGS-B with a smooth objective function
    # We maximize the minimum distance, which is equivalent to maximizing radius
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 0.0
        pts = x.reshape(-1, 2)
        # Check boundaries
        if np.any(np.abs(pts) > 0.5):
            return -1.0
        # Compute minimum distance
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        # Distance to boundary
        boundary_dists = np.min(0.5 - np.abs(pts), axis=1)
        min_boundary = np.min(boundary_dists)
        # Return negative of the minimum (to minimize)
        return -min(min_dist, min_boundary)
    
    def gradient(x):
        pts = x.reshape(-1, 2)
        grad = np.zeros_like(pts)
        
        # Find the closest pair
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_dist = dists[i, j]
        
        # Find the closest boundary point
        boundary_dists = np.min(0.5 - np.abs(pts), axis=1)
        min_boundary_idx = np.argmin(boundary_dists)
        min_boundary = boundary_dists[min_boundary_idx]
        
        if min_dist < min_boundary:
            # Gradient from pair interaction
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                unit = diff / norm
                grad[i] += unit
                grad[j] -= unit
        else:
            # Gradient from boundary
            idx = min_boundary_idx
            for k in range(2):
                if pts[idx, k] > 0:
                    grad[idx, k] += 1.0
                else:
                    grad[idx, k] -= 1.0
        
        return grad.flatten()
    
    # Try multiple optimization runs with different initializations
    for restart in range(3):
        if time.time() - start_time > time_limit - 10:
            break
        
        current_points = best_points.copy() if restart > 0 else points.copy()
        
        # Add small random perturbation for restarts
        if restart > 0:
            current_points += np.random.normal(0, 0.005, size=current_points.shape)
            # Clip to valid range
            current_points = np.clip(current_points, -0.49, 0.49)
        
        x0 = current_points.flatten()
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.x is not None:
            new_points = result.x.reshape(-1, 2)
            # Validate
            if np.all(np.isfinite(new_points)) and np.all(np.abs(new_points) <= 0.5):
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        
        if time.time() - start_time > time_limit - 5:
            break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Generate Fibonacci lattice initial points
    points = _fibonacci_lattice(N)
    
    # Scale to fit in [-0.5, 0.5]^2
    points = points - 0.5
    
    # Initial score
    initial_score = _score(points)
    
    # Optimize
    if time.time() - start_time < deadline - 20:
        optimized_points, opt_score = _optimize_from_initial(points, time_limit=deadline - start_time - 10)
        if opt_score > initial_score:
            points = optimized_points
    
    # Final validation
    points = np.clip(points, -0.5, 0.5)
    
    return points
# EVOLVE-BLOCK-END

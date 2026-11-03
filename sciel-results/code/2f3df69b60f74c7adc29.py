"""Packing 50 equal circles in a unit circle via simulated annealing."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 50

def _score(points):
    """Compute the objective: minimum of half-min-pair-distance and min-margin."""
    if len(points) != N:
        return -1.0
    # Check all points are finite
    if not np.all(np.isfinite(points)):
        return -1.0
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    # Compute margins to container
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = np.min(1.0 - norms)
    return min(min_pair_dist / 2.0, min_margin)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Try multiple random restarts with simulated annealing
    np.random.seed(42)
    
    # Initial best guess from hexagonal packing
    def hexagonal_init():
        """Generate initial hexagonal packing configuration."""
        # Use a hexagonal grid and select the best N points
        spacing = 0.25
        points = []
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.sqrt(x*x + y*y) <= 1.0 - spacing/2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Sort by distance from center and take innermost N
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:N]
            return points[idx]
        else:
            # Fallback: random points
            return np.random.uniform(-0.9, 0.9, size=(N, 2))
    
    def anneal(initial_points, max_iters=50000, t_start=0.1, t_end=1e-6):
        """Simulated annealing to optimize circle packing."""
        points = initial_points.copy()
        current_score = _score(points)
        best_local_points = points.copy()
        best_local_score = current_score
        
        # Cooling schedule
        cooling = (t_end / t_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate a random perturbation
            idx = np.random.randint(N)
            step_size = 0.01 * (1.0 + 0.5 * np.random.rand())
            angle = np.random.uniform(0, 2 * np.pi)
            delta = np.array([np.cos(angle), np.sin(angle)]) * step_size
            
            new_points = points.copy()
            new_points[idx] += delta
            
            # Keep inside container
            norm = np.linalg.norm(new_points[idx])
            if norm > 1.0:
                new_points[idx] *= (1.0 - 1e-9) / norm
            
            new_score = _score(new_points)
            
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_points = points.copy()
            else:
                # Accept with probability based on temperature
                delta_score = new_score - current_score
                if np.random.rand() < np.exp(delta_score / max(t_start * cooling**it, 1e-10)):
                    points = new_points
                    current_score = new_score
            
            # Periodically check deadline
            if it % 1000 == 0 and time.time() > deadline:
                break
        
        return best_local_points, best_local_score
    
    # Multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            initial = hexagonal_init()
        else:
            # Random initial configuration
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.random.uniform(0, 0.9, N)
            initial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Run annealing
        result_points, result_score = anneal(initial)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
    
    # Final local optimization using scipy
    if best_points is not None and time.time() < deadline:
        def neg_score(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        result = minimize(neg_score, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if result.fun < -best_score and time.time() < deadline:
            final_pts = result.x.reshape(N, 2)
            final_score = -result.fun
            if final_score > best_score and np.all(np.isfinite(final_pts)):
                best_score = final_score
                best_points = final_pts
    
    # Ensure valid output
    if best_points is None:
        # Fallback to simple hexagonal packing
        best_points = hexagonal_init()
    
    # Final validation and clipping
    best_points = np.array(best_points, dtype=np.float64)
    # Clip points to be strictly inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points

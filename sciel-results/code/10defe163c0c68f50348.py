"""
Packing 50 equal circles in a unit circle.

Strategy:
1. Use a structured initial layout based on concentric rings with hexagonal offsets.
2. Apply a physics-based relaxation (repulsive forces) to maximize the minimum distance.
3. Use simulated annealing to escape local minima.
4. Enforce boundary constraints by projecting points back into the feasible region.
"""

import numpy as np
import time

N = 50

def _compute_score(points):
    """
    Compute the radius of the largest equal circles that can be placed at `points`.
    Returns -inf if any point is outside the unit circle (with a small tolerance).
    """
    # Check boundary: all points must be strictly inside or on the unit circle
    # The radius of the small circle is limited by the distance to the boundary
    dist_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    
    # Check pairwise distances
    # To be efficient for N=50, we can compute all pairwise distances
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    # The minimum pairwise distance
    # We ignore the diagonal (distance to self)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pairwise_dist = np.min(dist_matrix)
    
    # The radius is half the minimum pairwise distance, but also limited by boundary
    r = min(min_pairwise_dist / 2.0, np.min(dist_to_boundary))
    
    # If any point is outside (dist_to_boundary < 0), the score is invalid
    if np.min(dist_to_boundary) < -1e-9:
        return -np.inf
        
    return r

def _construct_initial_layout():
    """
    Create an initial layout using concentric rings.
    Ring 0: center
    Ring k: 6*k points (hexagonal arrangement)
    """
    points = np.zeros((N, 2))
    
    # Center point
    points[0] = [0.0, 0.0]
    count = 1
    
    ring = 1
    while count < N:
        n_in_ring = 6 * ring
        # Place n_in_ring points on a circle of radius r
        # Optimal radius for this ring to pack 6*ring circles of radius ~0.12
        # The spacing between points on the ring should be roughly 2*r_circle
        # Circumference = 2 * pi * r_ring
        # Number of points = 6 * ring
        # Distance between adjacent points = 2 * pi * r_ring / (6 * ring)
        # We want this to be roughly 2 * r_circle
        # So r_ring ~= (6 * ring * 2 * r_circle) / (2 * pi) = (6 * ring * r_circle) / pi
        
        # Estimate r_circle from the task description: ~0.125
        r_circle_est = 0.125
        r_ring = (ring * 2 * r_circle_est) / np.sqrt(3) * 1.2  # Approximation
        
        # Actually, for hexagonal packing, the radial distance for ring k is k * 2 * r_circle / sqrt(3) ?
        # Let's just use a heuristic: r_ring = ring * 0.25
        # We will refine this during optimization.
        r_ring = ring * 0.22
        
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        # Add a slight offset for odd rings to stagger
        if ring % 2 == 1:
            angles += np.pi / n_in_ring
            
        x = r_ring * np.cos(angles)
        y = r_ring * np.sin(angles)
        
        # Only take as many points as needed
        take = min(n_in_ring, N - count)
        points[count:count+take] = np.column_stack((x[:take], y[:take]))
        count += take
        ring += 1
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Initial layout
    points = _construct_initial_layout()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Simulated Annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.995  # Cooling rate
    step_size = 0.05
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate a new point
        # Move one random point
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        # Project back into the unit circle if necessary
        norm = np.linalg.norm(new_points, axis=1)
        out_mask = norm > 1.0
        if np.any(out_mask):
            new_points[out_mask] /= norm[out_mask, np.newaxis]
            # Actually, we should project to the boundary or pull back
            # Let's just scale to be inside
            new_points[out_mask] = new_points[out_mask] / (norm[out_mask, np.newaxis] + 1e-9) * 0.999
            
        new_score = _compute_score(new_points)
        
        # Acceptance criterion
        if new_score > best_score or np.random.rand() < np.exp((new_score - best_score) / T):
            points = new_points
            if new_score > best_score:
                best_points = points.copy()
                best_score = new_score
                
        T *= alpha
        step_size *= 0.999
        
    # Final refinement: gradient ascent on the minimum distance
    # We want to maximize min(distances)
    # This is non-differentiable, so we use a sub-gradient approach or just more SA with smaller steps
    
    # Let's do a second phase of SA with smaller steps and lower temperature
    T = 0.001
    step_size = 0.01
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        norm = np.linalg.norm(new_points, axis=1)
        out_mask = norm > 1.0
        if np.any(out_mask):
            new_points[out_mask] = new_points[out_mask] / (norm[out_mask, np.newaxis] + 1e-9) * 0.999
            
        new_score = _compute_score(new_points)
        
        if new_score > best_score or np.random.rand() < np.exp((new_score - best_score) / T):
            points = new_points
            if new_score > best_score:
                best_points = points.copy()
                best_score = new_score
                
        T *= 0.999
        step_size *= 0.9995
        
    return best_points

# EVOLVE-BLOCK-END

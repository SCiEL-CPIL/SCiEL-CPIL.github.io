import numpy as np
import time
import sys

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization
    # Generate hexagonal lattice points
    # We want to maximize the radius, so we want the points to be as far apart as possible.
    # A hexagonal lattice with spacing s has min distance s.
    # We need to fit N points in a circle of radius 1.
    # The area of the hexagonal cluster is roughly proportional to N * s^2.
    # Area of disk is pi.
    # N * s^2 * (sqrt(3)/2) <= pi  => s <= sqrt(2*pi / (N*sqrt(3)))
    # For N=300, s <= sqrt(6.28 / (300*1.732)) = sqrt(6.28/519.6) = sqrt(0.012) = 0.1095
    # Let's try spacings around this value.
    
    best_init = None
    best_init_score = -1.0
    
    # Try a range of spacings
    s_max = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    for s in np.linspace(s_max * 0.95, s_max * 1.05, 20):
        # Generate lattice points
        # Range of indices to cover the unit circle
        # Max index k such that k*s <= 1
        k = int(1.0 / s) + 2
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append((x, y))
        
        points = np.array(points)
        
        # Filter points inside the unit circle (with some margin for radius)
        # We don't know the exact radius yet, but it will be ~ s/2.
        # So filter points with |p| < 1 - s/2
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii < 1.0 - s / 2.0
        points = points[mask]
        
        if len(points) < N:
            continue
            
        # Select N points. 
        # Heuristic: Pick points that are "most central" or maximize min-dist.
        # Simple heuristic: Sort by distance to center, pick first N.
        # This tends to form a compact cluster.
        idx = np.argsort(radii)
        candidates = points[idx[:N]]
        
        # Calculate score of this candidate
        # Score is min(min_pair_dist/2, min_boundary_dist)
        # Pairwise distances
        # This is O(N^2)
        if N > 1:
            # Compute pairwise distances
            # Using broadcasting
            # diff = candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
            # dists = np.linalg.norm(diff, axis=2)
            # min_dist = np.min(dists[np.triu_indices(N, k=1)])
            
            # More memory efficient for N=300
            min_pair_dist = np.inf
            for i in range(N):
                dists = np.linalg.norm(candidates - candidates[i], axis=1)
                # Ignore self
                dists[i] = np.inf
                min_pair_dist = min(min_pair_dist, np.min(dists))
            
            min_boundary_dist = np.min(1.0 - np.linalg.norm(candidates, axis=1))
            score = min(min_pair_dist / 2.0, min_boundary_dist)
        else:
            score = 1.0
            
        if score > best_init_score:
            best_init_score = score
            best_init = candidates.copy()
            
    if best_init is None:
        # Fallback: random points
        best_init = np.random.rand(N, 2) * 2 - 1
        # Ensure inside
        norms = np.linalg.norm(best_init, axis=1)
        mask = norms > 1.0
        best_init[mask] = best_init[mask] / norms[mask, np.newaxis] * 0.99
        
    # 2. Local Optimization
    # We want to maximize the minimum distance.
    # Current radius r
    def compute_score(points):
        # Min pair distance
        min_pair_dist = np.inf
        for i in range(N):
            dists = np.linalg.norm(points - points[i], axis=1)
            dists[i] = np.inf
            min_pair_dist = min(min_pair_dist, np.min(dists))
        
        min_boundary_dist = np.min(1.0 - np.linalg.norm(points, axis=1))
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    current_points = best_init
    current_score = best_init_score
    best_points = current_points.copy()
    best_score = current_score
    
    # Optimization loop
    # We use a gradient ascent on the minimum distance.
    # To speed up, we only consider "active" constraints.
    
    step_size = 0.01
    decay = 0.99
    min_step = 1e-6
    
    while time.time() < deadline:
        # Compute all pairwise distances
        # For N=300, 45k pairs.
        # Let's vectorize this.
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        norms = np.linalg.norm(current_points, axis=1)
        boundary_dists = 1.0 - norms
        min_boundary_dist = np.min(boundary_dists)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Determine active constraints
        # Pairs with dist < min_pair_dist + delta
        # Points with boundary_dist < min_boundary_dist + delta
        delta = 1e-3
        active_pairs = dists < (min_pair_dist + delta)
        active_boundary = boundary_dists < (min_boundary_dist + delta)
        
        # Compute gradients
        # grad[i] = sum of contributions from active pairs and boundary
        grad = np.zeros_like(current_points)
        
        # Pair contributions
        # For each active pair (i, j), move i away from j and j away from i.
        # Direction: (c_i - c_j) / dist and (c_j - c_i) / dist
        # We can sum these up.
        
        # Vectorized pair gradient calculation is tricky with variable active set.
        # Loop over active pairs?
        # Number of active pairs is usually small.
        
        # Get indices of active pairs
        # active_pairs is a boolean matrix.
        # We can iterate over non-zero elements.
        # But doing it in Python loop might be slow if many active pairs.
        # However, active pairs are those close to min, so should be few.
        
        # Alternative: Approximate gradient using all pairs weighted by distance?
        # No, that's not gradient of min.
        
        # Let's stick to iterating active pairs.
        # To speed up, we can use np.argwhere
        active_indices = np.argwhere(active_pairs)
        
        for i, j in active_indices:
            if i == j: continue
            v = current_points[i] - current_points[j]
            d = np.linalg.norm(v)
            if d < 1e-9:
                # Random direction
                v = np.random.rand(2)
                d = np.linalg.norm(v)
            # Move i in direction v, j in direction -v
            # Weight by how "active" it is?
            # Just unit vector.
            grad[i] += v / d
            grad[j] -= v / d
            
        # Boundary contributions
        active_boundary_indices = np.where(active_boundary)[0]
        for i in active_boundary_indices:
            norm = norms[i]
            if norm < 1e-9:
                continue
            # Move away from center: direction -c_i / norm
            grad[i] -= current_points[i] / norm
            
        # Normalize gradient?
        # The magnitude of grad varies.
        # Let's scale by step_size.
        
        # Update
        current_points += step_size * grad
        
        # Project back to feasible region?
        # If a point goes outside, pull it back?
        # The gradient should handle it, but numerical issues might occur.
        # Let's just clip to a safe radius, e.g., 0.99
        # But this might break the optimization if it's too aggressive.
        # Better: just let the boundary gradient handle it.
        
        # Decay step size
        step_size *= decay
        if step_size < min_step:
            break
            
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialization
    best_init = None
    best_init_score = -1.0
    
    s_max = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Try a range of spacings
    for s in np.linspace(s_max * 0.95, s_max * 1.05, 20):
        k = int(1.0 / s) + 2
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append((x, y))
        
        points = np.array(points)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii < 1.0 - s / 2.0
        points = points[mask]
        
        if len(points) < N:
            continue
            
        idx = np.argsort(radii)
        candidates = points[idx[:N]]
        
        # Calculate score
        min_pair_dist = np.inf
        for i in range(N):
            dists = np.linalg.norm(candidates - candidates[i], axis=1)
            dists[i] = np.inf
            min_pair_dist = min(min_pair_dist, np.min(dists))
            
        min_boundary_dist = np.min(1.0 - np.linalg.norm(candidates, axis=1))
        score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if score > best_init_score:
            best_init_score = score
            best_init = candidates.copy()
            
    if best_init is None:
        best_init = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(best_init, axis=1)
        mask = norms > 1.0
        best_init[mask] = best_init[mask] / norms[mask, np.newaxis] * 0.99
        
    # 2. Local Optimization
    current_points = best_init
    best_points = current_points.copy()
    best_score = best_init_score
    
    step_size = 0.01
    decay = 0.99
    min_step = 1e-6
    
    while time.time() < deadline:
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        norms = np.linalg.norm(current_points, axis=1)
        boundary_dists = 1.0 - norms
        min_boundary_dist = np.min(boundary_dists)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        delta = 1e-3
        active_pairs = dists < (min_pair_dist + delta)
        active_boundary = boundary_dists < (min_boundary_dist + delta)
        
        grad = np.zeros_like(current_points)
        
        active_indices = np.argwhere(active_pairs)
        for i, j in active_indices:
            if i == j: continue
            v = current_points[i] - current_points[j]
            d = np.linalg.norm(v)
            if d < 1e-9:
                v = np.random.rand(2)
                d = np.linalg.norm(v)
            grad[i] += v / d
            grad[j] -= v / d
            
        active_boundary_indices = np.where(active_boundary)[0]
        for i in active_boundary_indices:
            norm = norms[i]
            if norm < 1e-9:
                continue
            grad[i] -= current_points[i] / norm
            
        current_points += step_size * grad
        
        step_size *= decay
        if step_size < min_step:
            break
            
    return best_points
# EVOLVE-BLOCK-END

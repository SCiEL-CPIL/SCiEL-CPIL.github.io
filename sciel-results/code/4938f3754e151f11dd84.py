"""Circle packing in unit square using Delaunay-based radius computation and force optimization."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(centers):
    """
    Compute radii based on nearest neighbor distance and boundary distance.
    r_i = 0.5 * min(min_j ||c_i - c_j||, dist_to_boundary_i)
    """
    n = len(centers)
    # Compute pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Distance to boundaries
    # min(x, 1-x, y, 1-y)
    boundary_dists = np.min(np.stack([centers[:, 0], 1.0 - centers[:, 0], centers[:, 1], 1.0 - centers[:, 1]], axis=1), axis=1)
    
    # Min neighbor distance
    min_neighbor = np.min(dists, axis=1)
    
    # Radius is half the minimum
    radii = 0.5 * np.minimum(min_neighbor, boundary_dists)
    
    return radii

def _get_delaunay_edges(centers):
    """Get edges from Delaunay triangulation."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        return list(edges)
    except Exception:
        # Fallback: all pairs (should be rare if points are distinct)
        return [(i, j) for i in range(len(centers)) for j in range(i+1, len(centers))]

def _force_based_optimization(centers, time_limit=10.0):
    """
    Force-based optimization using Delaunay edges for stronger repulsion.
    """
    centers = centers.copy()
    n = len(centers)
    start_time = time.time()
    
    # Initial parameters
    step_size = 0.01
    decay = 0.99
    force_strength = 5.0
    boundary_strength = 10.0
    
    while time.time() - start_time < time_limit:
        # Compute radii for force calculation
        radii = _compute_radii(centers)
        
        # Get Delaunay edges
        edges = _get_delaunay_edges(centers)
        edge_set = set(edges)
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (Vectorized inner loop for efficiency)
        # We iterate over pairs, but use numpy for vector ops where possible
        # For N=26, O(N^2) is small (325 pairs), so python loop is fine
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    # Random jitter if overlapping exactly
                    jitter = np.random.randn(2) * 1e-4
                    forces[i] += jitter
                    forces[j] -= jitter
                    continue
                
                r_i = radii[i]
                r_j = radii[j]
                
                # Overlap
                overlap = r_i + r_j - dist
                
                # Determine force magnitude
                # Stronger force for Delaunay edges (tight packing)
                is_edge = (i, j) in edge_set or (j, i) in edge_set
                multiplier = 2.0 if is_edge else 1.0
                
                if overlap > 0:
                    # Repulsive force
                    force_mag = overlap * force_strength * multiplier
                    direction = diff / dist
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
                else:
                    # Weak attractive force to maintain packing density? 
                    # Or just let them settle. 
                    # Let's add a very weak attraction to prevent them from flying apart
                    # Actually, for packing, we want them as close as possible.
                    # The "overlap" term handles the repulsion. 
                    # We can add a small constant repulsion to keep them separated?
                    # No, the objective is to maximize sum of radii.
                    # The radii are determined by min distance.
                    # So we want to increase the minimum distance.
                    # This is equivalent to maximizing the minimum pairwise distance (TSP-like).
                    # A common heuristic is to use a repulsion force that decays with distance.
                    # F ~ 1/d^2 or similar.
                    
                    # Let's use a soft repulsion that always pushes apart
                    # F = strength * (target_dist / dist)^2 ?
                    # Or simply: if not overlapping, push apart slightly to increase min dist.
                    force_mag = 0.1 * force_strength / (dist * dist)
                    direction = diff / dist
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction

        # Boundary repulsion
        # Push away from boundaries to allow larger radii
        # Actually, we want circles to touch boundaries if possible to maximize radius.
        # But if they are too close, radius is limited by boundary.
        # We want to maximize min(dist_to_boundary, min_neighbor).
        # So we should push towards boundaries if neighbor distance is large?
        # This is complex. Let's just keep them inside with a strong repulsion from walls
        # to ensure validity, and let the neighbor repulsion dominate.
        
        for i in range(n):
            # Left
            d_left = centers[i][0]
            if d_left < radii[i] + 1e-3:
                forces[i][0] += boundary_strength * (radii[i] + 1e-3 - d_left)
            
            # Right
            d_right = 1.0 - centers[i][0]
            if d_right < radii[i] + 1e-3:
                forces[i][0] -= boundary_strength * (radii[i] + 1e-3 - d_right)
                
            # Bottom
            d_bottom = centers[i][1]
            if d_bottom < radii[i] + 1e-3:
                forces[i][1] += boundary_strength * (radii[i] + 1e-3 - d_bottom)
                
            # Top
            d_top = 1.0 - centers[i][1]
            if d_top < radii[i] + 1e-3:
                forces[i][1] -= boundary_strength * (radii[i] + 1e-3 - d_top)
                
            # Also, if far from boundary, pull towards boundary?
            # If neighbor distance > boundary distance, increasing boundary distance helps.
            # This is hard to do with simple forces.
            # Let's stick to preventing violation.
            
        # Normalize forces to prevent instability
        norm_forces = np.linalg.norm(forces, axis=1)
        max_norm = np.max(norm_forces)
        if max_norm > 1.0:
            forces = forces * (1.0 / max_norm)
            
        # Apply step
        centers += step_size * forces
        
        # Clip to [0, 1]
        centers = np.clip(centers, 0.0, 1.0)
        
        # Decay step size
        step_size *= decay
        
        # If step size is too small, break
        if step_size < 1e-6:
            break
            
    return centers

def _initial_hexagonal(n_points):
    """Create initial hexagonal packing layout."""
    centers = []
    row = 0
    while len(centers) < n_points:
        offset = 0.0 if row % 2 == 0 else 0.5
        # Adjust spacing to fit in [0,1]
        # Approx 5-6 circles per row
        cols = 5
        for col in range(cols):
            if len(centers) >= n_points:
                break
            x = (col + offset + 0.5) / cols
            y = (row + 0.5) / (n_points / cols + 0.5) # Rough estimate
            # Better: scale y to fit
            centers.append([x, y])
        row += 1
        
    # Scale to fit [0,1] properly
    if len(centers) > 0:
        centers = np.array(centers)
        # Normalize
        x_min, x_max = centers[:, 0].min(), centers[:, 0].max()
        y_min, y_max = centers[:, 1].min(), centers[:, 1].max()
        
        # Add padding
        pad = 0.1
        centers[:, 0] = pad + (centers[:, 0] - x_min) * (1 - 2*pad) / (x_max - x_min + 1e-8)
        centers[:, 1] = pad + (centers[:, 1] - y_min) * (1 - 2*pad) / (y_max - y_min + 1e-8)
        
    return np.array(centers[:n_points])

def _initial_grid(n_points):
    """Create initial grid layout."""
    side = int(np.ceil(np.sqrt(n_points)))
    centers = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n_points:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            centers.append([x, y])
            count += 1
    return np.array(centers)

def _initial_random(n_points, seed=42):
    """Create random initial layout."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 0.9, size=(n_points, 2))

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    best_obj = -np.inf
    best_centers = None
    
    # Initial configurations
    initial_configs = [
        _initial_hexagonal(N),
        _initial_grid(N),
        _initial_random(N, seed=42),
        _initial_random(N, seed=123),
    ]
    
    for config in initial_configs:
        if time.time() - start_time > time_limit:
            break
            
        # Step 1: Force-based optimization
        centers = _force_based_optimization(config, time_limit=10.0)
        
        # Step 2: Evaluate
        radii = _compute_radii(centers)
        obj = np.sum(radii)
        
        if obj > best_obj:
            best_obj = obj
            best_centers = centers.copy()
            
        # Step 3: Local refinement with L-BFGS-B
        if time.time() - start_time < time_limit - 15:
            centers_flat = centers.flatten()
            
            def objective(x):
                c = x.reshape(-1, 2)
                r = _compute_radii(c)
                return -np.sum(r)
                
            try:
                result = minimize(
                    objective,
                    centers_flat,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun < -best_obj:
                    best_obj = -result.fun
                    best_centers = result.x.reshape(-1, 2).copy()
            except Exception:
                pass
                
    # Final validation and clamping
    if best_centers is None:
        best_centers = initial_configs[0]
        
    radii = _compute_radii(best_centers)
    result = np.column_stack([best_centers, radii])
    
    # Ensure validity
    for i in range(N):
        x, y, r = result[i]
        # Clamp radius to boundary
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 1e-6)
        result[i, 2] = r
        
    # Ensure no overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i, :2] - result[j, :2])
            if result[i, 2] + result[j, 2] > d:
                scale = d / (result[i, 2] + result[j, 2] + 1e-8)
                result[i, 2] *= scale
                result[j, 2] *= scale
                
    return result

# EVOLVE-BLOCK-END

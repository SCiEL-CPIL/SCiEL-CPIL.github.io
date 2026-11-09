"""Improved program using FCC lattice packing with Farthest Point Sampling selection."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree


def construct():
    """
    Generate a face-centered cubic (FCC) lattice of points, and select the 100 points
    using Farthest Point Sampling (FPS) to maximize the minimum distance to the boundary
    and to other selected points.
    """
    def generate_fcc_points(a, n_range):
        """Generate FCC lattice points with lattice constant a."""
        points = []
        for i in range(-n_range, n_range + 1):
            for j in range(-n_range, n_range + 1):
                for k in range(-n_range, n_range + 1):
                    if (i + j + k) % 2 == 0:
                        points.append((i * a / 2.0, j * a / 2.0, k * a / 2.0))
        return np.array(points)
    
    def select_fps(points, n_select):
        """Select n_select points using Farthest Point Sampling."""
        if len(points) < n_select:
            return points
        
        # Precompute distance to boundary
        dist_to_boundary = np.min(0.5 - np.abs(points), axis=1)
        
        selected_indices = []
        selected_points = []
        remaining_indices = list(range(len(points)))
        
        for _ in range(n_select):
            if not remaining_indices:
                break
            
            if not selected_indices:
                # First point: max distance to boundary
                rem_arr = np.array(remaining_indices)
                scores = dist_to_boundary[rem_arr]
                best_local_idx = np.argmax(scores)
                best_global_idx = rem_arr[best_local_idx]
            else:
                # Subsequent points: max min(dist_to_boundary, dist_to_selected)
                rem_arr = np.array(remaining_indices)
                candidates = points[rem_arr]
                
                # Compute min distance to any selected point
                if len(selected_points) == 1:
                    dist_to_selected = np.linalg.norm(candidates - selected_points[0], axis=1)
                else:
                    # Use cKDTree for efficiency if many selected points
                    tree = cKDTree(selected_points)
                    dist_to_selected, _ = tree.query(candidates, k=1)
                
                scores = np.minimum(dist_to_boundary[rem_arr], dist_to_selected)
                best_local_idx = np.argmax(scores)
                best_global_idx = rem_arr[best_local_idx]
            
            selected_indices.append(best_global_idx)
            selected_points.append(points[best_global_idx])
            remaining_indices.remove(best_global_idx)
        
        return np.array(selected_points)

    best_points = None
    best_score = 0.0

    # Search for optimal lattice constant 'a'
    # The FPS selection will handle the trade-off between density and boundary distance.
    # We search a range of 'a' values.
    
    # Coarse search
    for a in np.linspace(0.1, 0.3, 21):
        n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
        points = generate_fcc_points(a, n_range)
        
        # Filter points inside the cube (with a small tolerance for floating point)
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        if len(points) < N:
            continue
        
        selected = select_fps(points, N)
        
        if len(selected) < N:
            continue
            
        # Calculate radius
        tree = cKDTree(selected)
        dists, _ = tree.query(selected, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        dist_to_boundary = np.min(0.5 - np.abs(selected), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        
        radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if radius > best_score:
            best_score = radius
            best_points = selected

    # Fine-tune around the best 'a'
    # Find the 'a' that gave the best score in the coarse search by re-running or tracking.
    # For simplicity, we'll just do a fine grid over the likely optimal range [0.15, 0.25]
    # based on the fact that r ~ 0.111 implies a ~ 0.111 * 2 * sqrt(2) ~ 0.31 for FCC?
    # Wait, if r = a / (2*sqrt(2)), then a = r * 2 * sqrt(2) ~ 0.111 * 2.82 ~ 0.31.
    # So the optimal 'a' should be around 0.31.
    
    for a in np.linspace(0.28, 0.35, 31):
        n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
        points = generate_fcc_points(a, n_range)
        
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        if len(points) < N:
            continue
        
        selected = select_fps(points, N)
        
        if len(selected) < N:
            continue
            
        tree = cKDTree(selected)
        dists, _ = tree.query(selected, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        dist_to_boundary = np.min(0.5 - np.abs(selected), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        
        radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if radius > best_score:
            best_score = radius
            best_points = selected

    if best_points is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_points
# EVOLVE-BLOCK-END

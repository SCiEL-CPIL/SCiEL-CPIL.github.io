"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

def _compute_radii_from_centers(centers):
    """
    Compute the maximum valid radius for each circle given its center,
    ensuring circles are inside the unit square and disjoint.
    Uses Voronoi diagram for efficient neighbor lookup.
    """
    n = len(centers)
    if n < 3:
        # Fallback for small n
        radii = np.zeros(n)
        for i in range(n):
            r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    r = min(r, d / 2.0)
            radii[i] = r
        return radii

    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback if Voronoi fails (e.g., degenerate points)
        radii = np.zeros(n)
        for i in range(n):
            r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    r = min(r, d / 2.0)
            radii[i] = r
        return radii

    radii = np.zeros(n)
    
    # For each point, find the minimum distance to its Voronoi cell boundary
    # This is determined by either:
    # 1. Distance to other points (via Voronoi vertices/edges)
    # 2. Distance to the unit square boundaries
    
    # We can compute the exact radius by checking:
    # - Distance to each boundary of the square
    # - Distance to the nearest vertex of the Voronoi cell (if inside the square)
    # - Distance to the nearest edge of the Voronoi cell (if the closest point on the edge is within the segment)
    
    # A simpler and robust approach for packing:
    # The radius is limited by:
    # 1. Box boundaries: min(x, 1-x, y, 1-y)
    # 2. Other circles: distance to nearest neighbor / 2 (but we need to be careful with Voronoi)
    
    # Actually, the standard Voronoi packing approach is:
    # r_i = min( dist_to_boundary_i, min_{j in neighbors(i)} dist(i, j) / 2 )
    # But Voronoi gives us the cell. The distance from a point to its Voronoi cell boundary
    # is exactly half the distance to the nearest neighbor.
    
    # Let's compute distance to boundaries first
    dist_to_box = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                             np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Now compute distance to Voronoi cell boundary for each point
    # voronoi.regions contains indices of vertices for each region
    # voronoi.vertices contains the vertex coordinates
    # voronoi.point_region maps points to regions
    
    # We need to find the minimum distance from each point to its Voronoi cell boundary.
    # This is equivalent to half the distance to the nearest neighbor.
    # However, we can also just compute the distance to all other points and take the min,
    # but that's O(N^2). With N=26, O(N^2) is 676 ops, which is trivial.
    # Let's just use the O(N^2) method for simplicity and robustness, as it's fast enough.
    
    # Distance to all other points
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Minimum distance to any other point
    min_dist_to_neighbor = np.min(dist_matrix, axis=1)
    
    # Radius is limited by half the distance to the nearest neighbor
    # and the distance to the box boundary
    radii = np.minimum(dist_to_box, min_dist_to_neighbor / 2.0)
    
    return radii


def _hex_packing_init():
    """
    Initialize with a hexagonal packing pattern, which is known to be dense.
    """
    centers = np.zeros((N, 2))
    
    # Hexagonal packing in a grid
    # Side length approx sqrt(N)
    side = int(np.ceil(np.sqrt(N)))
    
    # Hex spacing
    # For hex packing, horizontal spacing is 2*r, vertical is sqrt(3)*r
    # Let's just place them in a grid with some offset for hex-like structure
    # Or use a simple hex layout
    
    # Let's use a structured grid with hex offsets
    row = 0
    col = 0
    count = 0
    
    # Approximate spacing
    # If we have ~5x5 grid
    # Let's try to fit 26 circles
    # 5 columns, 5 rows = 25, plus 1 extra?
    # Or 4x4 = 16, 5x5=25.
    # Let's use a 5x5 grid but adjust positions
    
    # Actually, let's just use a jittered grid for initialization
    # The optimization will fix it.
    
    # Better init:
    # Place circles in a grid, then optimize.
    side = int(np.ceil(np.sqrt(N)))
    
    for i in range(N):
        gx = i % side
        gy = i // side
        
        # Hex-like offset
        offset_x = 0.0
        if gy % 2 == 1:
            offset_x = 0.5 / side
            
        x = (gx + 0.5) / side + offset_x
        y = (gy + 0.5) / side
        
        # Clip to [0.05, 0.95] to stay away from boundaries initially
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        
        centers[i] = [x, y]
        
    return centers


def _random_init():
    """Random initialization."""
    return np.random.uniform(0.1, 0.9, (N, 2))


def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Multi-start
    np.random.seed(42)
    
    # Try hex init first, then random
    inits = [_hex_packing_init()]
    for _ in range(5):
        inits.append(_random_init())
    
    for trial, init_centers in enumerate(inits):
        if time.time() > deadline:
            break
            
        centers = init_centers.copy()
        
        # Coordinate descent / local search
        # Step size schedule
        step = 0.05
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            current_score = np.sum(_compute_radii_from_centers(centers))
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
            
            # Try to improve by moving one circle at a time
            improved = False
            
            for i in range(N):
                if time.time() > deadline:
                    break
                    
                # Try small perturbations
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        new_centers = centers.copy()
                        new_centers[i, 0] += dx
                        new_centers[i, 1] += dy
                        
                        # Keep inside [0.01, 0.99]
                        if new_centers[i, 0] < 0.01 or new_centers[i, 0] > 0.99:
                            continue
                        if new_centers[i, 1] < 0.01 or new_centers[i, 1] > 0.99:
                            continue
                        
                        new_score = np.sum(_compute_radii_from_centers(new_centers))
                        
                        if new_score > current_score + 1e-8:
                            centers = new_centers
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                no_improve_count = 0
                # Slightly reduce step size
                step *= 0.95
                if step < 0.001:
                    step = 0.001
            else:
                no_improve_count += 1
                # Reduce step size when stuck
                step *= 0.8
                if step < 0.001:
                    step = 0.001
        
    # If no valid result, fall back to grid
    if best_centers is None:
        best_centers = _hex_packing_init()
    
    # Final radii computation
    radii = _compute_radii_from_centers(best_centers)
    
    # Construct result
    result = np.column_stack([best_centers, radii])
    
    # Final validation and clipping
    for i in range(N):
        x, y, ri = result[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    return result
# EVOLVE-BLOCK-END

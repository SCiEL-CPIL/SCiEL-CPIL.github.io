"""Circle packing in unit square: Voronoi-based radii with gradient ascent."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def _voronoi_radii(pts):
    """
    Given N points in [0,1]^2, compute the maximum inscribed circle radius
    for each point's Voronoi cell clipped to the unit square.
    Returns array of radii (N,).
    """
    n = len(pts)
    radii = np.zeros(n)
    
    # Handle boundary: distance to each wall
    wall_dists = np.min(np.stack([
        pts[:, 0],        # left
        1.0 - pts[:, 0],  # right
        pts[:, 1],        # bottom
        1.0 - pts[:, 1],  # top
    ], axis=1), axis=1)
    
    # Build Delaunay triangulation
    try:
        tri = Delaunay(pts)
    except Exception:
        return wall_dists.copy()
    
    # For each point, find its nearest neighbor distance (half the edge to nearest Delaunay neighbor)
    # Actually, the Voronoi cell edge distance = half the distance to the nearest other point
    # But more precisely, the radius of the largest inscribed circle in a Voronoi cell
    # is the minimum distance from the site to any edge of its Voronoi cell.
    
    # Efficient approach: for each point, the Voronoi cell edges are perpendicular bisectors
    # of Delaunay edges. The distance from point i to the bisector of edge (i,j) is |d_ij|/2.
    # So the radius is min over all Delaunay neighbors j of i: |d_ij|/2, and also wall distances.
    
    # Get Delaunay neighbors
    # tri.simplices gives triangles; we need neighbor relationships
    # Use the convex hull and interior edges
    
    # Simpler: compute pairwise distances and for each point find min distance to any other / 2
    # But that's O(n^2) which is fine for n=26
    
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = np.linalg.norm(pts[i] - pts[j])
    
    # For each point, the Voronoi radius is min(wall_dist, min_j dist(i,j)/2)
    # But this is only exact if the nearest neighbor defines the closest Voronoi edge,
    # which is true for the closest point but other Voronoi edges from non-nearest neighbors
    # could be closer. Actually for a Voronoi cell, the edges are at distance d_ij/2 from site i
    # for each Delaunay neighbor j. The closest edge is from the closest Delaunay neighbor.
    
    # For accuracy, let's use the actual Voronoi cell computation via scipy
    # But scipy doesn't directly give Voronoi vertices easily in older versions.
    
    # Alternative: the radius is the minimum of:
    # 1. Distance to walls
    # 2. For each pair (i,j) that are Delaunay neighbors: dist(i,j)/2
    
    # Get Delaunay neighbor pairs
    neighbor_pairs = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a+1, 3):
                i, j = simplex[a], simplex[b]
                if i < j:
                    neighbor_pairs.add((i, j))
                else:
                    neighbor_pairs.add((j, i))
    
    # Also add convex hull edges
    if hasattr(tri, 'convex_hull'):
        for simplex in tri.convex_hull:
            for a in range(3):
                for b in range(a+1, 3):
                    i, j = simplex[a], simplex[b]
                    if i < j:
                        neighbor_pairs.add((i, j))
                    else:
                        neighbor_pairs.add((j, i))
    
    # Compute radii
    for i in range(n):
        r = wall_dists[i]
        for (a, b) in neighbor_pairs:
            if a == i or b == i:
                other = b if a == i else a
                d = np.linalg.norm(pts[i] - pts[other])
                r = min(r, d / 2.0)
        radii[i] = r
    
    return radii


def _objective(pts):
    """
    Negative sum of Voronoi radii (we minimize negative = maximize sum).
    """
    radii = _voronoi_radii(pts)
    return -np.sum(radii)


def _objective_with_penalty(pts, penalty=1000.0):
    """
    Objective with penalty for points outside [0,1]^2.
    """
    n = len(pts)
    # Penalty for going outside
    penalty_val = 0.0
    for i in range(n):
        for coord in range(2):
            if pts[i, coord] < 0:
                penalty_val += penalty * pts[i, coord]**2
            elif pts[i, coord] > 1:
                penalty_val += penalty * (pts[i, coord] - 1)**2
    
    base_obj = _objective(pts)
    return base_obj + penalty_val


def _make_initial_points(n_restarts=8, seed=42):
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(seed)
    inits = []
    
    # Grid-based initial
    side = int(np.ceil(np.sqrt(N)))
    grid_pts = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                         for i in range(side) for j in range(side)])[:N]
    inits.append(grid_pts.copy())
    
    # Random inits
    for _ in range(n_restarts - 1):
        pts = rng.uniform(0.1, 0.9, size=(N, 2))
        inits.append(pts)
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s hard kill
    
    best_pts = None
    best_score = -np.inf
    
    # Multi-stage optimization with increasing penalty strength
    penalty_schedule = [100.0, 500.0, 1000.0]
    
    inits = _make_initial_points(n_restarts=8, seed=42)
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        current_pts = init_pts.copy()
        
        for stage, penalty in enumerate(penalty_schedule):
            if time.time() > deadline:
                break
            
            # Use L-BFGS-B with bounds
            bounds = [(0.0, 1.0)] * (2 * N)
            
            def obj_fn(x):
                pts = x.reshape(N, 2)
                return _objective_with_penalty(pts, penalty)
            
            # Initial gradient estimate
            try:
                res = minimize(
                    obj_fn,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                current_pts = res.x.reshape(N, 2)
            except Exception:
                pass
            
            # Clip to valid range
            current_pts = np.clip(current_pts, 0.0, 1.0)
        
        # Evaluate final result
        radii = _voronoi_radii(current_pts)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_pts = current_pts.copy()
    
    # If we haven't found anything valid, fall back to grid
    if best_pts is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final high-precision polish pass
    if time.time() < deadline - 20:
        try:
            bounds = [(0.0, 1.0)] * (2 * N)
            
            def obj_fn(x):
                pts = x.reshape(N, 2)
                return _objective(pts)
            
            res = minimize(
                obj_fn,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            polished_pts = res.x.reshape(N, 2)
            polished_pts = np.clip(polished_pts, 0.0, 1.0)
            
            polished_radii = _voronoi_radii(polished_pts)
            polished_score = np.sum(polished_radii)
            
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    # Build output: (x, y, radius) for each circle
    final_radii = _voronoi_radii(best_pts)
    
    # Ensure radii are non-negative and circles fit in square
    for i in range(N):
        # Shrink slightly to ensure strict interior
        max_r = min(best_pts[i, 0], 1.0 - best_pts[i, 0], 
                    best_pts[i, 1], 1.0 - best_pts[i, 1])
        final_radii[i] = min(final_radii[i], max_r) * (1 - 1e-6)
        final_radii[i] = max(final_radii[i], 1e-6)
    
    result = np.column_stack([best_pts, final_radii])
    
    # Final validity check: ensure no overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < final_radii[i] + final_radii[j] - 1e-9:
                # Shrink the larger radius
                shrink = (final_radii[i] + final_radii[j] - d) / 2.0 + 1e-7
                if final_radii[i] >= final_radii[j]:
                    final_radii[i] -= shrink
                else:
                    final_radii[j] -= shrink
    
    result[:, 2] = final_radii
    return result
# EVOLVE-BLOCK-END

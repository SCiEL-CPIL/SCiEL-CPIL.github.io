"""Circle packing in unit square - Voronoi-based dual representation."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _compute_voronoi_inradii(xy):
    """
    Given centers, compute the maximum radius for each circle such that it stays
    inside its Voronoi cell and the unit square.
    This is the "inradius" of the Voronoi cell intersected with the square.
    """
    n = len(xy)
    if n < 3:
        r = np.zeros(n)
        for i in range(n):
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            r[i] = wall_dist * 0.5
        return r
    
    try:
        vor = Voronoi(xy)
    except Exception:
        # Fallback: just use wall distances and nearest neighbors
        r = np.zeros(n)
        for i in range(n):
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            min_dist = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                min_dist = min(min_dist, d)
            r[i] = min(wall_dist, min_dist) * 0.5
        return r
    
    r = np.zeros(n)
    for i in range(n):
        # Distance to walls
        wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        min_r = wall_dist
        
        # Find the Voronoi region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region (shouldn't happen for points strictly inside square if we added boundaries)
            # But Voronoi of points inside square can have unbounded cells extending out.
            # We handle walls separately via wall_dist.
            # We need to check if the cell is cut by the square boundary effectively.
            # Actually, for points inside [0,1]^2, the Voronoi cell might extend outside.
            # The intersection of the Voronoi cell and [0,1]^2 is what matters.
            # If the cell is unbounded, it definitely intersects the boundary.
            # The max radius is limited by wall_dist and distance to other Voronoi vertices/edges.
            pass
        else:
            # Get the vertices of this Voronoi region
            vertices = vor.vertices[region]
            
            # The max radius is the min distance to any edge of this polygon.
            # Edges are segments between consecutive vertices.
            # Also need to consider the square boundaries.
            
            # Distance to square boundaries is already in wall_dist.
            # Distance to Voronoi edges:
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % len(vertices)]
                
                # Distance from point xy[i] to segment v1-v2
                # Project point onto line v1-v2
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                if dx == 0 and dy == 0:
                    continue
                t = ((xy[i, 0] - v1[0]) * dx + (xy[i, 1] - v1[1]) * dy) / (dx * dx + dy * dy)
                t = max(0, min(1, t))
                proj_x = v1[0] + t * dx
                proj_y = v1[1] + t * dy
                dist = np.sqrt((xy[i, 0] - proj_x)**2 + (xy[i, 1] - proj_y)**2)
                min_r = min(min_r, dist)
        
        r[i] = min_r
    
    return r


def _objective_voronoi(params, n):
    """
    Objective function for Voronoi-based optimization.
    We optimize the positions (xy) to maximize the sum of the Voronoi inradii.
    params: flat array of shape (2*n,) containing [x_0, y_0, x_1, y_1, ...]
    """
    xy = params.reshape(-1, 2)
    
    # Check if any points are too close or outside (soft penalty)
    penalty = 0.0
    for i in range(n):
        x, y = xy[i, 0], xy[i, 1]
        if x < 0 or x > 1 or y < 0 or y > 1:
            penalty += 100.0 * (max(0, x-1) + max(0, 1-x) + max(0, y-1) + max(0, 1-y))**2
    
    # Compute inradii
    r = _compute_voronoi_inradii(xy)
    
    # We want to maximize sum(r)
    # Also penalize if any radius is very small (encourages uniformity/spreading)
    # But primarily maximize sum.
    
    # Add a small penalty for points being very close to each other to avoid numerical issues in Voronoi
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)
            if d < 1e-4:
                penalty += (1e-4 - d)**2 * 1000
    
    return -np.sum(r) + penalty


def _gradient_descend_voronoi(xy_init, deadline, max_iter=2000):
    """
    Gradient descent on positions using the Voronoi inradius objective.
    """
    n = len(xy_init)
    params = xy_init.ravel().copy()
    best_params = params.copy()
    best_val = _objective_voronoi(params, n)
    
    # Use L-BFGS-B which handles bounds well
    bounds = [(0.0, 1.0), (0.0, 1.0)] * n
    
    try:
        result = minimize(
            _objective_voronoi,
            params,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.x is not None:
            new_val = _objective_voronoi(result.x, n)
            if new_val < best_val:
                best_val = new_val
                best_params = result.x.copy()
    except Exception:
        pass
    
    return best_params.reshape(-1, 2)


def _construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Multi-start with different initial configurations
    # 1. Grid
    # 2. Random
    # 3. Hexagonal-ish
    
    inits = []
    
    # Grid 5x5 (25 points) + 1 center
    side = 5
    grid_pos = []
    for i in range(side):
        for j in range(side):
            if len(grid_pos) < N:
                grid_pos.append([(i + 0.5) / side, (j + 0.5) / side])
    # Add one more point in the center of the largest gap? Or just random
    grid_pos.append([0.5, 0.5])
    inits.append(np.array(grid_pos[:N]))
    
    # Grid 6x6 (36 points) -> take first 26
    side = 6
    grid_pos = []
    for i in range(side):
        for j in range(side):
            if len(grid_pos) < N:
                grid_pos.append([(i + 0.5) / side, (j + 0.5) / side])
    inits.append(np.array(grid_pos[:N]))
    
    # Random
    random.seed(42)
    rand_pos = np.array([[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)] for _ in range(N)])
    inits.append(rand_pos)
    
    # Hexagonal lattice approximation
    hex_pos = []
    spacing = 0.2
    for i in range(6):
        for j in range(6):
            if len(hex_pos) < N:
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing * np.sqrt(3)
                if i % 2 == 1:
                    y += spacing * np.sqrt(3) / 2
                if x < 1.0 and y < 1.0:
                    hex_pos.append([x, y])
    if len(hex_pos) >= N:
        inits.append(np.array(hex_pos[:N]))
    
    # Optimize each start
    for init_idx, xy_init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        # Run gradient descent
        opt_xy = _gradient_descend_voronoi(xy_init, deadline - 10.0)
        
        # Compute final radii
        r = _compute_voronoi_inradii(opt_xy)
        
        # Validate
        total, valid = _eval_packing(opt_xy, r)
        
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = opt_xy.copy()
            best_r = r.copy()
        
        # Local polish: Best-fit local search on the result
        if time.time() < deadline - 5.0:
            # Simple local search: try moving each point slightly
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                original_xy = opt_xy.copy()
                original_r = r.copy()
                original_total = total
                
                # Try small random perturbations
                for _ in range(20):
                    new_xy = opt_xy.copy()
                    new_xy[i] += np.array([random.gauss(0, 0.01), random.gauss(0, 0.01)])
                    new_xy[i] = np.clip(new_xy[i], 0.0, 1.0)
                    
                    new_r = _compute_voronoi_inradii(new_xy)
                    new_total, new_valid = _eval_packing(new_xy, new_r)
                    
                    if new_valid and new_total > original_total + 1e-6:
                        opt_xy = new_xy
                        r = new_r
                        total = new_total
            
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = opt_xy.copy()
                best_r = r.copy()
    
    # Fallback if no valid result
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy_init = np.array([[(i % side + 0.5) / side, (i // side + 0.5) / side] for i in range(N)])
        r = np.full(N, 0.35 / side)
        total, valid = _eval_packing(xy_init, r)
        if valid:
            best_xy = xy_init
            best_r = r
            best_total = total
            best_valid = True
        else:
            # Last resort
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.3 / side
            return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.3 / side
    
    return result

# Wrapper to match interface
def construct():
    return _construct()
# EVOLVE-BLOCK-END

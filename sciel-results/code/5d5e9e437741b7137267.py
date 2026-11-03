import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices_batch(xys_rots):
    """Batch compute vertices for all hexagons. xys_rots is (N, 3) array."""
    n_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    return np.stack((x, y), axis=-1)

def sat_overlap_batch(vertices_list):
    """
    Vectorized SAT overlap check for all pairs.
    Returns (has_overlap, penalty) where penalty is sum of squared penetration depths.
    """
    n_pts = len(vertices_list)
    axes_all = []
    for v in vertices_list:
        hex_axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                hex_axes.append(n)
        axes_all.append(hex_axes)
    
    has_overlap = False
    penalty = 0.0
    
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            all_axes = axes_all[i] + axes_all[j]
            
            min_overlap = float('inf')
            separated = False
            for n in all_axes:
                proj1 = v1 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                proj2 = v2 @ n
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    separated = True
                    break
                
                overlap = min(max1, max2) - max(min1, min2)
                min_overlap = min(min_overlap, overlap)
            
            if not separated:
                if min_overlap > 1e-9:
                    has_overlap = True
                    penalty += min_overlap ** 2
    
    return has_overlap, penalty

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """
    Compute outer hex side from a list of vertex arrays.
    """
    all_points = np.vstack(vertices_list)
    return outer_hex_side(all_points, outer_rot_deg)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    sqrt3 = np.sqrt(3.0)
    
    # Triangular lattice basis vectors for unit hexagon tiling
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    def lattice_point(i, j):
        return i * a1 + j * a2
    
    # Generate candidate lattice point sets (12 points)
    # The "ring" configuration: center + 6 neighbors + 5 more
    # Let's enumerate all possible 12-point subsets from a reasonable range
    
    # First, generate all lattice points in a bounding region
    max_idx = 4
    lattice_points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            pt = lattice_point(i, j)
            lattice_points.append(pt)
    lattice_points = np.array(lattice_points)
    
    # For efficiency, precompute distances and use a smarter enumeration
    # The key configurations are those where points are "connected" (no gaps)
    
    # Let's use a different approach: generate configurations by starting from
    # a seed and adding neighbors greedily, or use known good patterns
    
    # Pattern 1: Ring (1 + 6 + 5)
    def make_ring_lattice():
        pts = []
        # Center
        pts.append(lattice_point(0, 0))
        # 6 neighbors
        neighbors = [
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
        ]
        for i, j in neighbors:
            pts.append(lattice_point(i, j))
        # 5 more from the next ring
        next_ring = [
            (2, 0), (1, 1), (-1, 2), (-2, 1), (-2, 0),
            (-1, -1), (0, -2), (1, -2), (2, -1), (2, 1), (1, 2)
        ]
        # Pick 5 that give good symmetry
        for i, j in next_ring[:5]:
            pts.append(lattice_point(i, j))
        return np.array(pts)
    
    # Pattern 2: 4x3 grid
    def make_grid_lattice(rows, cols):
        pts = []
        for r in range(rows):
            for c in range(cols):
                i = c
                j = r
                if r % 2 == 1:
                    i = c  # offset handled by lattice
                pts.append(lattice_point(i, j))
        return np.array(pts)
    
    # Pattern 3: 3x4 grid
    # Pattern 4: 2x6 grid
    # Pattern 5: 6x2 grid
    
    # Generate diverse initial configurations
    configs = []
    
    # Ring configuration
    ring_pts = make_ring_lattice()
    ring_pts -= np.mean(ring_pts, axis=0)
    configs.append(ring_pts)
    
    # Various grid configurations
    for rows, cols in [(4, 3), (3, 4), (2, 6), (6, 2), (5, 3), (3, 5)]:
        if rows * cols >= N:
            grid_pts = make_grid_lattice(rows, cols)[:N]
            grid_pts -= np.mean(grid_pts, axis=0)
            configs.append(grid_pts)
    
    # Also try some "staircase" patterns
    def make_staircase(n1, n2):
        pts = []
        for i in range(n1):
            for j in range(n2):
                pts.append(lattice_point(i, j))
        return np.array(pts)
    
    for n1, n2 in [(3, 4), (4, 3), (2, 6), (6, 2)]:
        if n1 * n2 >= N:
            st_pts = make_staircase(n1, n2)[:N]
            st_pts -= np.mean(st_pts, axis=0)
            configs.append(st_pts)
    
    # Remove duplicates (approximately)
    unique_configs = []
    for cfg in configs:
        is_dup = False
        for uc in unique_configs:
            if np.allclose(cfg, uc, atol=0.1):
                is_dup = True
                break
        if not is_dup:
            unique_configs.append(cfg)
    configs = unique_configs
    
    best_L = float('inf')
    best_params = None
    
    # For each configuration, search over rotation angles
    # The outer hexagon rotation is coupled with the inner configuration rotation
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Search over rotation angles
        # Try a range of angles, then refine
        for base_rot in range(0, 60, 5):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Try this rotation and several outer rotations
            for outer_rot in [0.0, 30.0, 15.0, 45.0, 75.0]:
                if time.time() - start_time > DEADLINE - 15:
                    break
                
                # Apply rotation to the configuration
                theta = np.radians(base_rot)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
                
                rotated_pts = (R @ cfg.T).T
                
                # Center the configuration
                rotated_pts -= np.mean(rotated_pts, axis=0)
                
                # Create the xys_rots array
                xys_rots = np.zeros((N, 3))
                xys_rots[:, 0] = rotated_pts[:, 0]
                xys_rots[:, 1] = rotated_pts[:, 1]
                xys_rots[:, 2] = base_rot  # All hexagons have same rotation
                
                # Check for overlaps (should be none for lattice points)
                vertices_list = get_hex_vertices_batch(xys_rots)
                has_overlap, penalty = sat_overlap_batch(vertices_list)
                
                if has_overlap and penalty > 1e-6:
                    continue
                
                # Compute outer hex side
                L_val = outer_hex_side_batch(vertices_list, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = np.concatenate((xys_rots.flatten(), [outer_rot]))
    
    # If we found a good solution, try to polish it with local optimization
    if best_params is not None and time.time() - start_time < DEADLINE - 30:
        def objective(params):
            if time.time() - start_time > DEADLINE - 5:
                return 1e6
            
            xys_rots = params[:3*N].reshape(N, 3)
            outer_rot = params[3*N]
            
            vertices_list = get_hex_vertices_batch(xys_rots)
            
            has_overlap, penalty = sat_overlap_batch(vertices_list)
            
            L = outer_hex_side_batch(vertices_list, outer_rot)
            
            return L + 100.0 * penalty
        
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            
            vertices_list = get_hex_vertices_batch(xys_rots)
            
            has_overlap, penalty = sat_overlap_batch(vertices_list)
            
            if not has_overlap or penalty < 1e-6:
                L_val = outer_hex_side_batch(vertices_list, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    # Final fallback
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle

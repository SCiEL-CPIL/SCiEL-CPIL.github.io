"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices_vec(centers, angles_rad, side=1.0):
    """
    Vectorized computation of hexagon vertices.
    centers: (N, 2) array
    angles_rad: (N,) array
    Returns: (N, 6, 2) array of vertices
    """
    N = len(centers)
    # Angles for the 6 vertices relative to the hexagon center
    # Standard hexagon vertices are at 0, 60, 120, 180, 240, 300 degrees
    # If we rotate by angle_rad, the vertices are at angle_rad + 0, angle_rad + 60, ...
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    # (N, 6) angles
    all_angles = angles_rad[:, np.newaxis] + base_angles[np.newaxis, :]
    
    cos_a = np.cos(all_angles) # (N, 6)
    sin_a = np.sin(all_angles) # (N, 6)
    
    # (N, 6, 2)
    offsets = np.stack([side * cos_a, side * sin_a], axis=-1)
    
    # Broadcast centers to (N, 1, 2)
    centers_expanded = centers[:, np.newaxis, :]
    
    return centers_expanded + offsets

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Fast computation of the minimum outer side length to contain all inner hexagons.
    inner: list of [x, y, angle_deg]
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    inner_arr = np.array(inner)
    centers = inner_arr[:, :2]
    angles_deg = inner_arr[:, 2]
    angles_rad = np.deg2rad(angles_deg)
    
    # Get all vertices of all inner hexagons
    # Shape: (N, 6, 2)
    verts = get_hex_vertices_vec(centers, angles_rad, 1.0)
    
    # Flatten to (N*6, 2)
    all_verts = verts.reshape(-1, 2)
    
    ox, oy = outer_center
    # Translate to outer center
    all_verts = all_verts - np.array([ox, oy])
    
    # Rotate to align with outer hexagon axes
    # The outer hexagon has vertices at outer_angle_deg + 0, 60, ...
    # Its normals are at outer_angle_deg + 30, 90, ...
    # We want to project onto these normals.
    # Alternatively, we can rotate the points by -outer_angle_deg so the outer hexagon
    # is in standard orientation (vertices at 0, 60...), and then project onto standard normals (30, 90...).
    
    theta = np.deg2rad(outer_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Rotation matrix for -theta:
    # [ cos(-t)  -sin(-t) ]   [ cos_t   sin_t ]
    # [ sin(-t)   cos(-t) ] = [ -sin_t  cos_t ]
    
    # all_verts is (M, 2)
    x = all_verts[:, 0]
    y = all_verts[:, 1]
    
    lx = x * cos_t + y * sin_t
    ly = -x * sin_t + y * cos_t
    
    # Standard normals for a hexagon with vertices at 0, 60, ...
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1) # (6, 2)
    
    # Project all points onto all 6 normals
    # all_verts_rot is (M, 2), normals is (6, 2)
    # We want max projection for each normal, then max over all normals.
    # Actually, for a convex container, the distance from center to the face is determined by the max projection of any point onto the outward normal.
    # The side length S of the outer hexagon is related to the apothem A (distance from center to face) by A = S * sqrt(3) / 2.
    # So S = 2 * A / sqrt(3).
    # A = max over all points and all normals of (dot(point, normal)).
    # Wait, the normals point OUTWARD. The projection of a point P onto normal n is P . n.
    # If P is inside, P . n <= A.
    # So A = max(P . n) for all P, n.
    
    # Compute projections: (M, 6)
    projections = all_verts.dot(normals.T)
    
    # Max projection for each normal
    max_projections = np.max(projections, axis=0) # (6,)
    
    # The required apothem is the max of these
    max_apothem = np.max(max_projections)
    
    # Ensure non-negative (should be if centered well)
    if max_apothem < 0:
        max_apothem = 0.0
        
    side = 2.0 * max_apothem / SQRT3
    return float(side)

def compute_overlap_penalty_fast(inner, min_dist=SQRT3 - 0.001):
    """
    Fast computation of overlap penalty based on center distances.
    This is an approximation but sufficient for guiding optimization.
    """
    inner_arr = np.array(inner)
    centers = inner_arr[:, :2]
    N = len(centers)
    
    if N < 2:
        return 0.0
    
    # Compute pairwise distances
    # diff: (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # We only care about i < j
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    # Penalty for pairs closer than min_dist
    # diff = min_dist - dist
    # penalty = sum(diff^2) for diff > 0
    diffs = min_dist - pair_dists
    positive_diffs = np.maximum(diffs, 0)
    penalty = np.sum(positive_diffs**2)
    
    return float(penalty)

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Define initial layouts
    # Layout 1: Standard hexagonal lattice
    axial_coords_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    # Layout 2: Slightly different arrangement to break symmetry
    axial_coords_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    layouts = [axial_coords_1, axial_coords_2]
    
    best_side_global = float('inf')
    best_inner_global = None
    best_outer_center_global = (0.0, 0.0)
    best_outer_angle_global = 0.0
    
    for layout_coords in layouts:
        if time.time() > deadline - 30:
            break
            
        # Convert axial to Cartesian
        epsilon = 1e-6
        scale = 1.0 + epsilon
        centers = []
        for q, r in layout_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
            
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # 2. Optimize Outer Hexagon only first
        def objective_outer_fast(params):
            cx, cy, angle_deg = params
            return compute_required_side_fast(initial_inner, (cx, cy), angle_deg)
            
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        best_outer_val = float('inf')
        best_outer_params = None
        
        # Try a few starting angles
        for init_angle in [0, 15, 30, 45, 60]:
            if time.time() > deadline - 20:
                break
            init_params = np.array([cx_avg, cy_avg, float(init_angle)])
            # Simple local search
            # Use scipy minimize if available, else manual gradient descent or Nelder-Mead
            from scipy.optimize import minimize
            try:
                res = minimize(objective_outer_fast, init_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_outer_val:
                    best_outer_val = res.fun
                    best_outer_params = res.x
            except Exception:
                pass
        
        if best_outer_params is None:
            best_outer_params = np.array([cx_avg, cy_avg, 0.0])
            best_outer_val = objective_outer_fast(best_outer_params)
            
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        
        # 3. Joint Optimization
        # Parameters: x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle
        # Total 3*N + 3 parameters
        
        def objective_joint(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]
            
            side = compute_required_side_fast(inner_list, (ocx, ocy), oangle)
            penalty = compute_overlap_penalty_fast(inner_list)
            
            return side + 1000.0 * penalty
        
        # Initial parameters
        init_params = []
        for x, y, a in initial_inner:
            init_params.extend([x, y, a])
        init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
        init_params = np.array(init_params)
        
        best_joint_val = objective_joint(init_params)
        best_joint_params = init_params.copy()
        
        # Multi-start local search with "Shake and Polish"
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Perturb initial parameters
            current_params = init_params + np.random.normal(0, 0.5, init_params.shape)
            # Keep angles in range
            current_params[2::3][:N] = np.mod(current_params[2::3][:N], 360)
            current_params[N*3+2] = np.mod(current_params[N*3+2], 360)
            
            current_val = objective_joint(current_params)
            
            # Polish
            from scipy.optimize import minimize
            try:
                res = minimize(objective_joint, current_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_joint_val:
                    best_joint_val = res.fun
                    best_joint_params = res.x
            except Exception:
                pass
            
            # Additional shake and polish
            if time.time() > deadline - 5:
                break
                
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                # Shake
                shake_params = best_joint_params + np.random.normal(0, 0.2, best_joint_params.shape)
                shake_params[2::3][:N] = np.mod(shake_params[2::3][:N], 360)
                shake_params[N*3+2] = np.mod(shake_params[N*3+2], 360)
                
                try:
                    res_shake = minimize(objective_joint, shake_params, method='Nelder-Mead',
                                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res_shake.fun < best_joint_val:
                        best_joint_val = res_shake.fun
                        best_joint_params = res_shake.x
                except Exception:
                    pass
        
        # Extract best inner and outer
        best_inner_local = []
        for i in range(N):
            x = best_joint_params[i*3]
            y = best_joint_params[i*3+1]
            a = best_joint_params[i*3+2]
            best_inner_local.append([x, y, a])
            
        best_ocx = best_joint_params[N*3]
        best_ocy = best_joint_params[N*3+1]
        best_oangle = best_joint_params[N*3+2]
        
        # Check validity
        if check_all_disjoint(best_inner_local, eps=1e-5):
            # Final polish of outer only
            def objective_outer_polish(params):
                cx, cy, angle_deg = params
                return compute_required_side_fast(best_inner_local, (cx, cy), angle_deg)
                
            from scipy.optimize import minimize
            try:
                res_final = minimize(objective_outer_polish, [best_ocx, best_ocy, best_oangle], 
                                     method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res_final.fun < best_side_global:
                    best_side_global = res_final.fun
                    best_inner_global = best_inner_local
                    best_outer_center_global = (res_final.x[0], res_final.x[1])
                    best_outer_angle_global = res_final.x[2]
            except Exception:
                pass
            
            if best_inner_global is None:
                best_side_global = compute_required_side_fast(best_inner_local, (best_ocx, best_ocy), best_oangle)
                best_inner_global = best_inner_local
                best_outer_center_global = (best_ocx, best_ocy)
                best_outer_angle_global = best_oangle

    # Fallback if no valid solution found
    if best_inner_global is None:
        best_inner_global = initial_inner
        best_outer_center_global = (0.0, 0.0)
        best_outer_angle_global = 0.0
        best_side_global = compute_required_side_fast(best_inner_global, best_outer_center_global, best_outer_angle_global)

    return best_inner_global, best_outer_center_global, best_side_global, best_outer_angle_global

# EVOLVE-BLOCK-END

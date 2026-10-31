"""
Combines the structural search of the current program with the vectorized, 
differentiable support-function optimization inspired by the polyhex approach.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices_vectorized(centers, angles_deg, side=1.0):
    """
    Vectorized computation of vertices for multiple hexagons.
    centers: (N, 2) array of (x, y)
    angles_deg: (N,) array of angles in degrees
    Returns: (N, 6, 2) array of vertices.
    """
    angles_rad = np.radians(angles_deg)
    # Base angles for flat-top hexagon: 0, 60, 120, 180, 240, 300
    base_angles = np.arange(6) * np.pi / 3.0
    # (N, 6)
    all_angles = angles_rad[:, np.newaxis] + base_angles[np.newaxis, :]
    
    cos_a = np.cos(all_angles) * side
    sin_a = np.sin(all_angles) * side
    
    # (N, 6, 2)
    verts = np.empty((len(centers), 6, 2))
    verts[:, :, 0] = centers[:, 0:1] + cos_a
    verts[:, :, 1] = centers[:, 1:2] + sin_a
    return verts

def compute_side_length(centers, angles_deg, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Uses vectorized projection.
    """
    if len(centers) == 0:
        return 0.0
        
    verts = get_hex_vertices_vectorized(centers, angles_deg, 1.0)
    # Flatten to (N*6, 2)
    verts_flat = verts.reshape(-1, 2)
    
    # Translate to outer center
    verts_rel = verts_flat - np.array(outer_center)
    
    # Rotate by -outer_angle to align with standard hex axes
    theta = math.radians(outer_angle_deg)
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    verts_rot = verts_rel @ R.T
    
    # Normals for standard hexagon (flat-top, vertices at 0, 60...)
    # Face normals are at 30, 90, 150, 210, 270, 330
    normal_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # (6, 2)
    
    # Projections: (N*6, 6)
    projections = verts_rot @ normals.T
    
    # Max projection for each normal
    max_projections = np.max(projections, axis=0)
    
    # The limiting projection determines the side length
    max_proj = np.max(max_projections)
    
    # S * sqrt(3) / 2 >= max_proj  => S = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / SQRT3

def compute_side_length_smooth(centers, angles_deg, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation using log-sum-exp for better optimization landscape.
    """
    if len(centers) == 0:
        return 0.0
        
    verts = get_hex_vertices_vectorized(centers, angles_deg, 1.0)
    verts_flat = verts.reshape(-1, 2)
    verts_rel = verts_flat - np.array(outer_center)
    
    theta = math.radians(outer_angle_deg)
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    verts_rot = verts_rel @ R.T
    
    normal_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    projections = verts_rot @ normals.T # (N*6, 6)
    
    # Smooth max over vertices for each direction
    # lse(x) = (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, shift by max
    max_per_dir = np.max(projections, axis=0) # (6,)
    shifted = projections - max_per_dir[np.newaxis, :]
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted, axis=0)
    smooth_max_per_dir = max_per_dir + (1.0 / beta) * np.log(sum_exp)
    
    # Smooth max over directions
    max_final = np.max(smooth_max_per_dir)
    shifted_final = smooth_max_per_dir - max_final
    sum_exp_final = np.sum(np.exp(beta * shifted_final))
    smooth_max_final = max_final + (1.0 / beta) * np.log(sum_exp_final)
    
    return 2.0 * smooth_max_final / SQRT3

def check_disjoint(centers, angles_deg, eps=1e-6):
    """
    Check if all inner hexagons are pairwise disjoint.
    Simple bounding box check first, then SAT for close pairs.
    """
    n = len(centers)
    verts_list = []
    for i in range(n):
        v = get_hex_vertices_vectorized(np.array([[centers[i][0], centers[i][1]]]), np.array([angles_deg[i]]), 1.0)[0]
        verts_list.append(v)
        
    for i in range(n):
        v1 = verts_list[i]
        # Bounding box of v1
        min_x1, max_x1 = np.min(v1[:,0]), np.max(v1[:,0])
        min_y1, max_y1 = np.min(v1[:,1]), np.max(v1[:,1])
        
        for j in range(i+1, n):
            v2 = verts_list[j]
            # Bounding box of v2
            min_x2, max_x2 = np.min(v2[:,0]), np.max(v2[:,0])
            min_y2, max_y2 = np.min(v2[:,1]), np.max(v2[:,1])
            
            # Quick reject
            if max_x1 < min_x2 - 2.0 or max_x2 < min_x1 - 2.0 or max_y1 < min_y2 - 2.0 or max_y2 < min_y1 - 2.0:
                continue
            
            # SAT Check
            overlap = False
            # Get axes from edges of v1 and v2
            axes = []
            for v in [v1, v2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
            
            for axis in axes:
                proj1 = v1 @ axis
                proj2 = v2 @ axis
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 + eps or max2 < min1 + eps:
                    overlap = False
                    break
                else:
                    overlap = True
            
            if overlap:
                return False
    return True

def generate_initial_configs():
    """
    Generate a set of diverse initial configurations for the 11 hexagons.
    We use axial coordinates for a tight packing.
    """
    configs = []
    
    # Config 1: Center + 6 ring + 4 extensions (standard flower)
    # Axial: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extensions: (2,-1), (1,1), (-2,1), (-1,-1)
    c1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (1,1), (-2,1), (-1,-1)]
    
    # Config 2: More compact block?
    # Try shifting some to fill gaps
    c2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    
    # Config 3: Asymmetric
    c3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (2,-2), (1,-2), (0,-2)]
    
    # Config 4: Line-like but bent
    c4 = [(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (1,1), (2,1), (-1,1), (-2,1), (-3,1)]
    
    # Config 5: Compact 3x3 minus corners?
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # (0,-1), (1,-1), (2,-1)
    # Plus (3,0), (3,1)? No, need 11.
    c5 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,-1), (1,-1), (2,-1), (3,0), (3,1)]
    
    for axial_coords in [c1, c2, c3, c4, c5]:
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        # Center the configuration
        cx_avg = np.mean([c[0] for c in centers])
        cy_avg = np.mean([c[1] for c in centers])
        centered_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        angles = [0.0] * N
        configs.append((np.array(centered_centers), np.array(angles)))
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    initial_configs = generate_initial_configs()
    
    for centers, angles in initial_configs:
        if time.time() > deadline - 10:
            break
            
        # 1. Optimize Outer Container for this fixed inner layout
        def obj_outer(params):
            cx, cy, ang = params
            return compute_side_length(centers, angles, (cx, cy), ang)
        
        def obj_outer_smooth(params):
            cx, cy, ang = params
            return compute_side_length_smooth(centers, angles, (cx, cy), ang, beta=30.0)
        
        # Start from origin
        x0 = [0.0, 0.0, 0.0]
        
        try:
            # Smooth pass
            res_smooth = minimize(obj_outer_smooth, x0, method='Nelder-Mead', 
                                  options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            # Exact pass
            res_exact = minimize(obj_outer, res_smooth.x, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            side = res_exact.fun
            if side < best_side:
                best_side = side
                best_outer_center = (res_exact.x[0], res_exact.x[1])
                best_outer_angle = res_exact.x[2]
                best_centers = centers.copy()
                best_angles = angles.copy()
                
        except Exception:
            continue
            
        # 2. Joint Optimization
        if time.time() > deadline - 20:
            continue
            
        # Flatten parameters: 3*N inner + 3 outer
        x0_joint = []
        for i in range(N):
            x0_joint.extend([best_centers[i][0], best_centers[i][1], best_angles[i]])
        x0_joint.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
        
        def obj_joint(params):
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            
            c = np.array(inner_params[:2*N]).reshape(N, 2)
            a = inner_params[2*N:]
            oc = (outer_params[0], outer_params[1])
            oa = outer_params[2]
            
            # Overlap penalty
            # Simplified check: if distance between centers < 2.0 * (1 - eps), it's likely overlapping
            # But rotation matters. For speed in objective, we use a soft penalty based on distance
            # and rely on final check.
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(c[i] - c[j])
                    # Hexagons of side 1 have distance between centers >= sqrt(3) if touching flat sides?
                    # Actually, min distance is sqrt(3) for flat-to-flat, 2.0 for point-to-point?
                    # Min distance is sqrt(3) * sin(60) * 2? No.
                    # Apothem is sqrt(3)/2. Distance between centers for touching = 2 * apothem = sqrt(3) ~ 1.732.
                    if dist < 1.732 - 1e-4:
                        penalty += (1.732 - dist) * 100.0
                        
            side = compute_side_length_smooth(c, a, oc, oa, beta=20.0)
            return side + penalty
        
        def obj_joint_exact(params):
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            
            c = np.array(inner_params[:2*N]).reshape(N, 2)
            a = inner_params[2*N:]
            oc = (outer_params[0], outer_params[1])
            oa = outer_params[2]
            
            # Hard overlap check is expensive. We'll do it after optimization.
            # For now, just return side length.
            return compute_side_length(c, a, oc, oa)
        
        try:
            # Run joint optimization
            res_joint = minimize(obj_joint, x0_joint, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            # Extract best
            inner_params = res_joint.x[:3*N]
            outer_params = res_joint.x[3*N:]
            
            c_new = np.array(inner_params[:2*N]).reshape(N, 2)
            a_new = inner_params[2*N:]
            oc_new = (outer_params[0], outer_params[1])
            oa_new = outer_params[2]
            
            # Validate disjointness
            if check_disjoint(c_new, a_new):
                side_new = compute_side_length(c_new, a_new, oc_new, oa_new)
                if side_new < best_side:
                    best_side = side_new
                    best_centers = c_new
                    best_angles = a_new
                    best_outer_center = oc_new
                    best_outer_angle = oa_new
                    
        except Exception:
            continue
        
    # 3. Final Polish
    if time.time() < deadline - 5 and best_centers is not None:
        def obj_polish(params):
            cx, cy, ang = params
            return compute_side_length(best_centers, best_angles, (cx, cy), ang)
        
        try:
            res_polish = minimize(obj_polish, [best_outer_center[0], best_outer_center[1], best_outer_angle],
                                  method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res_polish.fun < best_side:
                best_side = res_polish.fun
                best_outer_center = (res_polish.x[0], res_polish.x[1])
                best_outer_angle = res_polish.x[2]
        except Exception:
            pass

    # Ensure strict containment
    best_side *= (1.0 + 1e-6)
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
        
    return inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

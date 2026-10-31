import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # Safety margin before 150s kill

def get_hex_vertices_vectorized(centers, angles):
    """
    centers: (N, 2) array
    angles: (N,) array in radians
    Returns: (N, 6, 2) array of vertices
    """
    # Angles for vertices relative to center: 0, 60, 120, 180, 240, 300
    # Plus the rotation of the hexagon itself
    # Vertex k of hex i: 
    # x = cx + cos(angle_i + k*pi/3)
    # y = cy + sin(angle_i + k*pi/3)
    
    k = np.arange(6) * (math.pi / 3.0) # (6,)
    # Broadcast: angles is (N, 1), k is (1, 6) -> (N, 6)
    theta = angles[:, np.newaxis] + k[np.newaxis, :]
    
    cos_theta = np.cos(theta) # (N, 6)
    sin_theta = np.sin(theta) # (N, 6)
    
    # Centers are (N, 2) -> (N, 1, 2)
    cx = centers[:, 0:1, np.newaxis]
    cy = centers[:, 1:2, np.newaxis]
    
    # Expand to (N, 6, 2)
    # We want shape (N, 6, 2).
    # x_coords: (N, 6)
    # y_coords: (N, 6)
    x_verts = centers[:, 0:1] + cos_theta
    y_verts = centers[:, 1:2] + sin_theta
    
    # Stack to (N, 6, 2)
    verts = np.stack((x_verts, y_verts), axis=-1)
    return verts

def compute_outer_side_and_penalty(params):
    """
    params: array of length 3*N + 3
    First 3*N: x, y, angle for each of N inner hexagons
    Last 3: X, Y, Angle for outer hexagon
    
    Returns: (objective_value, outer_side, penalty)
    """
    # Unpack
    inner_params = params[:3*N]
    outer_params = params[3*N:]
    
    inner_centers = np.zeros((N, 2))
    inner_angles = np.zeros(N)
    for i in range(N):
        inner_centers[i, 0] = inner_params[i*3]
        inner_centers[i, 1] = inner_params[i*3 + 1]
        inner_angles[i] = inner_params[i*3 + 2]
        
    outer_cx = outer_params[0]
    outer_cy = outer_params[1]
    outer_angle = outer_params[2]
    
    # 1. Compute Outer Side Length
    # Get all inner vertices
    verts = get_hex_vertices_vectorized(inner_centers, inner_angles) # (N, 6, 2)
    all_verts = verts.reshape(-1, 2) # (6N, 2)
    
    # Translate relative to outer center
    all_verts[:, 0] -= outer_cx
    all_verts[:, 1] -= outer_cy
    
    # Rotate into outer frame
    cos_a = math.cos(outer_angle)
    sin_a = math.sin(outer_angle)
    # Rotation matrix: [[cos, sin], [-sin, cos]]? 
    # Standard rotation of coordinate system by -theta is [[cos, sin], [-sin, cos]]
    # If we rotate the frame by theta, we rotate points by -theta.
    # Let's stick to: Project onto normals.
    # Normals of outer hex (in its local frame) are at 30, 90, ... degrees.
    # If outer is rotated by outer_angle, the normals in global frame are at outer_angle + 30 + i*60.
    # Alternatively, rotate points by -outer_angle and project onto fixed normals.
    
    rot_mat = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_verts_rot = all_verts @ rot_mat.T
    
    # Fixed normals for hexagon with vertex at 0 deg (flat top/bottom? No, vertex at 0).
    # Vertices at 0, 60... Normals at 30, 90...
    theta_normals = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack((np.cos(theta_normals), np.sin(theta_normals))) # (6, 2)
    
    # Projections
    projs = all_verts_rot @ normals.T # (6N, 6)
    max_proj = np.max(projs)
    
    # Side length s relates to apothem a = s * sqrt(3) / 2
    # max_proj is the maximum distance from center to any vertex along a normal.
    # This is the apothem of the containing hexagon?
    # Yes, if the hexagon is defined by inequalities n_i . x <= a.
    # So a = max_proj.
    outer_side = 2.0 * max_proj / SQRT3
    
    # 2. Compute Overlap Penalty
    # For each pair (i, j), check if they overlap.
    # We use the SAT method.
    # To make it differentiable/smooth for the optimizer, we can use a soft penalty.
    # However, Nelder-Mead doesn't need gradients.
    # Let's calculate the actual minimum distance (gap) for each pair.
    # If gap < 0, they overlap. Penalty += gap^2.
    
    penalty = 0.0
    # Pre-compute vertices for each hex for reuse? 
    # We already have `verts` which is (N, 6, 2).
    
    # Axes for SAT:
    # For hex i, normals are at angle_i + 30 + k*60.
    # For hex j, normals are at angle_j + 30 + k*60.
    # We need to check projection on all these axes.
    
    for i in range(N):
        for j in range(i+1, N):
            # Get vertices
            vi = verts[i] # (6, 2)
            vj = verts[j] # (6, 2)
            
            # Determine axes
            # Normals for i
            axes_i = inner_angles[i] + np.radians(30 + np.arange(6)*60)
            # Normals for j
            axes_j = inner_angles[j] + np.radians(30 + np.arange(6)*60)
            
            all_axes = np.concatenate([axes_i, axes_j])
            
            # Project vertices onto these axes
            # (6, 2) dot (12, 2).T -> (6, 12)
            proj_i = vi @ np.column_stack((np.cos(all_axes), np.sin(all_axes))).T
            proj_j = vj @ np.column_stack((np.cos(all_axes), np.sin(all_axes))).T
            
            min_i = np.min(proj_i, axis=0)
            max_i = np.max(proj_i, axis=0)
            min_j = np.min(proj_j, axis=0)
            max_j = np.max(proj_j, axis=0)
            
            # Gap = min(max_i, max_j) - max(min_i, min_j)
            # If gap < 0, overlap.
            # Actually, separation is:
            # If max_i < min_j, gap = min_j - max_i
            # If max_j < min_i, gap = min_i - max_j
            # Else gap = 0 (overlap or touch)
            
            # Vectorized gap calculation
            # gap1 = min_j - max_i (positive if i is left of j)
            # gap2 = min_i - max_j (positive if j is left of i)
            # If both negative, they overlap.
            # The "minimum gap" is min(gap1, gap2).
            # If min_gap < 0, overlap magnitude is -min_gap.
            
            gap1 = min_j - max_i
            gap2 = min_i - max_j
            min_gap = np.minimum(gap1, gap2)
            
            # If min_gap is negative, it's an overlap.
            # We want to penalize overlap.
            # Smooth penalty: use max(0, -min_gap)^2
            overlaps = np.maximum(0, -min_gap)
            penalty += np.sum(overlaps**2)
            
    # Objective: Minimize outer side + large penalty for overlaps
    # We want to prioritize validity, then minimize size.
    # If penalty > 0, we should push it to 0 first.
    # A common trick: if penalty > 0, return penalty * 1000 + outer_side.
    # Otherwise return outer_side.
    
    if penalty > 1e-6:
        objective = 1000.0 * penalty + outer_side
    else:
        objective = outer_side
        
    return objective, outer_side, penalty

def construct():
    start_time = time.time()
    
    # Initial Configuration: Tight Hexagonal Cluster
    # 1 center, 6 around it, 4 more to make 11.
    # A common dense packing for 11 is a hexagon of radius 2 (19 hexes) but we only take 11.
    # Let's use a specific shape:
    # Row 0: 3 hexes
    # Row 1: 3 hexes
    # Row 2: 3 hexes
    # Row 3: 2 hexes
    # This is a parallelogram-ish shape.
    
    # Better: Start with the 7-hex cluster (center + 6 neighbors) and add 4 to the "corners" of the next ring.
    # Axial coords for 7-hex:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add 4 more from the next ring:
    # (2,0), (1,1), (-1,2), (-2,1)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    # Convert to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    init_centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        init_centers.append([x, y])
        
    init_centers = np.array(init_centers)
    
    # Initial angles: 0 for all
    init_angles = np.zeros(N)
    
    # Initial outer: Center at (0,0), angle 0
    init_outer = [0.0, 0.0, 0.0]
    
    # Construct initial params vector
    init_params = []
    for i in range(N):
        init_params.append(init_centers[i, 0])
        init_params.append(init_centers[i, 1])
        init_params.append(init_angles[i])
    init_params.extend(init_outer)
    init_params = np.array(init_params)
    
    best_objective, best_side, best_penalty = compute_outer_side_and_penalty(init_params)
    best_params = init_params.copy()
    
    # Optimization Strategy:
    # 1. Try to eliminate overlaps (Penalty -> 0) while minimizing side.
    # 2. Use multiple random restarts with small perturbations.
    
    num_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Perturb initial parameters
        # Jitter positions slightly
        jitter_pos = rng.normal(0, 0.1, size=(N, 2))
        # Jitter angles slightly
        jitter_ang = rng.normal(0, 0.1, size=N)
        # Jitter outer
        jitter_outer = rng.normal(0, 0.1, size=3)
        
        curr_params = init_params.copy()
        for i in range(N):
            curr_params[i*3] += jitter_pos[i, 0]
            curr_params[i*3+1] += jitter_pos[i, 1]
            curr_params[i*3+2] += jitter_ang[i]
        curr_params[3*N:] += jitter_outer
        
        # Optimize
        try:
            res = minimize(
                lambda p: compute_outer_side_and_penalty(p)[0],
                curr_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if res.success:
                obj, side, pen = compute_outer_side_and_penalty(res.x)
                
                # Accept if better
                # We prioritize low penalty, then low side.
                # If current best has pen > 0 and new has pen < 0.001, accept.
                # If both pen ~ 0, accept lower side.
                
                if best_penalty > 1e-4:
                    if pen < 0.001 and (pen < best_penalty or (pen < 0.001 and side < best_side)):
                        best_params = res.x
                        best_side = side
                        best_penalty = pen
                        best_objective = obj
                else:
                    if pen < 0.001 and side < best_side:
                        best_params = res.x
                        best_side = side
                        best_penalty = pen
                        best_objective = obj
                        
        except Exception:
            continue

    # Final Polish: If we have a valid solution (penalty ~ 0), do a tight optimization
    # to squeeze the side length further.
    if best_penalty < 0.001:
        try:
            # Use L-BFGS-B for better convergence if smooth enough? 
            # Nelder-Mead is safer for non-smooth boundaries.
            # Let's do another Nelder-Mead with tighter tolerance.
            res_final = minimize(
                lambda p: compute_outer_side_and_penalty(p)[0],
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            obj, side, pen = compute_outer_side_and_penalty(res_final.x)
            if pen < 0.001 and side < best_side:
                best_params = res_final.x
                best_side = side
                best_penalty = pen
        except Exception:
            pass

    # Extract final result
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        # Convert angle to degrees
        a_deg = math.degrees(a)
        inner.append([x, y, a_deg])
        
    outer_center = (best_params[3*N], best_params[3*N+1])
    outer_side = best_side
    outer_angle = math.degrees(best_params[3*N+2])
    
    # Fallback if something went wrong (e.g. huge penalty)
    if best_penalty > 1.0:
        # Return the initial lattice scaled up to be safe
        scale = 1.1
        fallback_inner = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            fallback_inner.append([x, y, 0.0])
        fallback_outer_center = (0.0, 0.0)
        fallback_outer_side = 4.0 # Safe estimate
        fallback_outer_angle = 0.0
        return fallback_inner, fallback_outer_center, fallback_outer_side, fallback_outer_angle

    return inner, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    construct()

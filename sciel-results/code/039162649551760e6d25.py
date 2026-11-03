import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_outer_side_and_penalty(params, outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns: (side_length, penalty)
        side_length is the min side of outer hex (centered at 0,0 for symmetry, or optimized? 
        Let's fix outer center to (0,0) and angle to 0. 
        Wait, the task allows outer center/angle to vary. 
        However, due to symmetry, the optimal packing of 12 hexes in a hex likely has the 
        outer hex centered at the centroid of the inner cluster. 
        To save computation, I will fix outer center to (0,0) and outer angle to 0. 
        If this is too restrictive, I can add 3 params for outer. 
        Let's try fixing outer to (0,0) angle 0 first. It reduces vars to 36.
        Actually, the "score" is the outer side length. 
        The constraint is that all inner hexes must be inside the outer hex.
        
        Let's define the outer hex by its side length S.
        The condition "inner hex i is inside outer hex" is equivalent to:
        For all 6 normals of the outer hex, the projection of all vertices of inner hex i 
        onto the normal must be <= S * sqrt(3) / 2 (apothem).
        
        So, for a given configuration of inner hexes, the minimum required S is:
        S_min = (max_projection_over_all_hexes_and_normals) * 2 / sqrt(3).
        
        So the objective function is simply S_min, provided there are no overlaps.
        If there are overlaps, we add a large penalty.
        """
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:] # radians
        
        # 1. Calculate minimum outer side length required to contain all
        # Outer hex is centered at (0,0) and has angle 0.
        # Normals for angle 0 hex (pointy top? No, standard regular hexagon 
        # usually has flat top or pointy top. 
        # The seed code used: normals at theta + 30 + k*60. 
        # If theta=0, normals are at 30, 90, 150, 210, 270, 330.
        # This corresponds to a "flat-top" hexagon? 
        # Let's stick to the seed's definition to be safe.
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        polys = []
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang[i]
            # Convert ang from rad to deg for get_hex_verts if it expects deg? 
            # My get_hex_verts expects angle_deg.
            verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
            polys.append(verts)
            verts_rel = verts - np.array([0.0, 0.0]) # Outer center is 0,0
            projs = verts_rel @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / math.sqrt(3)
        
        # 2. Calculate overlap penalty
        # Use SAT to check overlap depth.
        # For each pair, calculate min separation. If negative, it's overlap depth.
        # Penalty = sum of (overlap_depth)^2 * large_weight
        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                # Quick bounding box check
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue # No overlap possible
                
                # SAT
                min_sep = float('inf')
                axes = []
                for k in range(6):
                    edge = p1[(k+1)%6] - p1[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)
                for k in range(6):
                    edge = p2[(k+1)%6] - p2[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)
                
                for axis in axes:
                    proj1 = np.dot(p1, axis)
                    proj2 = np.dot(p2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Separation: positive if apart, negative if overlapping
                    sep = max(min1, min2) - min(max1, max2)
                    if sep < min_sep:
                        min_sep = sep
                
                if min_sep < 0:
                    # Overlap depth is -min_sep
                    # Quadratic penalty to encourage smooth movement
                    penalty += (-min_sep) ** 2 * 100.0
        
        return side, penalty

    def objective(params):
        side, penalty = calc_outer_side_and_penalty(params)
        return side + penalty

    # Initial configurations
    inits = []
    
    # 1. Honeycomb lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append(np.concatenate([init_pos.flatten(), init_ang]))
    
    # 2. Random starts
    np.random.seed(42)
    for _ in range(10):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append(np.concatenate([pos.flatten(), ang]))

    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break

        try:
            # Nelder-Mead is derivative-free and handles non-smooth penalties well
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            # Evaluate result
            final_side, final_penalty = calc_outer_side_and_penalty(res.x)
            
            # Check if valid (no significant overlap)
            if final_penalty < 1e-6:
                # We have a valid configuration.
                # The outer hexagon is currently assumed to be centered at (0,0) and angle 0.
                # Is this optimal? 
                # The task allows moving the outer hexagon. 
                # However, if we fix the inner hexes relative to each other, 
                # the minimal enclosing regular hexagon is unique up to rotation and translation.
                # By fixing the outer to (0,0) angle 0, we are forcing the inner cluster 
                # to align with that. 
                # To be safe, we should check if shifting the outer hexagon (or equivalently, 
                # the inner cluster) could reduce the side length.
                # Actually, the side length calculated in `calc_outer_side_and_penalty` 
                # is the minimal side length for an outer hex centered at (0,0) angle 0 
                # that contains the inner hexes. 
                # If we allow the outer hex to move, we are effectively asking: 
                # What is the minimal S such that there EXISTS a center C and angle A 
                # such that all inner hexes are inside Hex(C, A, S)?
                # This is equivalent to finding the minimal S such that the inner hexes 
                # can be contained in ANY regular hexagon of side S.
                # Due to the symmetry of the regular hexagon, the "shape" of the container 
                # is fixed. The position and rotation are free.
                # Minimizing the bounding box of a set of points in a fixed orientation 
                # (by translating the points) is a standard problem.
                # My current objective minimizes S for a FIXED outer orientation and center.
                # This is suboptimal because the optimal outer orientation might not be 0.
                
                # To handle this, I should include outer center and angle in the optimization?
                # That adds 3 variables. Total 39.
                # Or, I can do a second pass: for the best inner configuration found,
                # optimize the outer center and angle.
                
                if final_side < best_side:
                    best_side = final_side
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass

    # Polish the best solution: Optimize outer center and angle
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        ang = best_params[2*N:]
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2] # radians
            # Calculate min side for this outer config
            theta_rad = oa
            normals = []
            for k in range(6):
                ang_n = theta_rad + math.pi/6 + k * math.pi/3
                normals.append([math.cos(ang_n), math.sin(ang_n)])
            normals = np.array(normals)
            
            max_proj = 0.0
            for i in range(N):
                cx, cy = pos[i]
                a_rad = ang[i]
                verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
                verts_rel = verts - np.array(oc)
                projs = verts_rel @ normals.T
                max_proj_i = np.max(projs)
                if max_proj_i > max_proj:
                    max_proj = max_proj_i
            
            return max_proj * 2.0 / math.sqrt(3)

        # Initial guess: centroid, 0 angle
        centroid = np.mean(pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res_outer = minimize(
                outer_obj, 
                x0_outer, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_outer.fun < best_side:
                best_side = res_outer.fun
                best_outer_center = res_outer.x[:2]
                best_outer_angle = np.degrees(res_outer.x[2])
        except:
            pass

    # If no valid solution found, fallback to honeycomb
    if best_params is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        best_params = np.concatenate([pos.flatten(), ang])
        best_side, _ = calc_outer_side_and_penalty(best_params)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    # Construct return values
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle

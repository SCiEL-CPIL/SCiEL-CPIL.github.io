import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
DEADLINE = 140.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def time_left():
        return deadline - time.time()

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return unit_verts_0 @ rot.T + np.array([cx, cy])

    def calc_outer_side(params, outer_center=(0.0, 0.0), outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns min side length of outer hexagon containing all inner hexagons.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)
        
        max_proj = 0.0
        oc = np.array(outer_center)
        
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang_rad[i]
            verts = get_hex_verts(cx, cy, a_rad)
            verts_rel = verts - oc
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / SQRT3
        return side

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def get_constraints(params):
        """
        Returns a list of constraint dictionaries for scipy minimize.
        Each constraint ensures that the gap between hex i and hex j is >= 0.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        polys = []
        for i in range(N):
            cx, cy = pos[i]
            verts = get_hex_verts(cx, cy, ang_rad[i])
            polys.append(verts)
            
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                
                # Quick bounding box check to skip distant pairs
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue
                
                # Compute gap
                gap = sat_gap(p1, p2)
                
                # We want gap >= 0. 
                # To make it a standard inequality constraint (fun(x) >= 0), 
                # we can just use the gap function itself.
                # However, sat_gap is not differentiable at contact points.
                # SLSQP uses finite differences, which might struggle if the function is too noisy.
                # We will use a smooth approximation or just rely on the robustness of the initial guess.
                
                # For SLSQP, defining the constraint as a closure is efficient.
                def make_gap_closure(idx1, idx2):
                    def gap_func(p):
                        pos_p = p[:2*N].reshape(N, 2)
                        ang_p = p[2*N:]
                        v1 = get_hex_verts(pos_p[idx1][0], pos_p[idx1][1], ang_p[idx1])
                        v2 = get_hex_verts(pos_p[idx2][0], pos_p[idx2][1], ang_p[idx2])
                        
                        # Bounding box check
                        min_x1, min_y1 = np.min(v1, axis=0)
                        max_x1, max_y1 = np.max(v1, axis=0)
                        min_x2, min_y2 = np.min(v2, axis=0)
                        max_x2, max_y2 = np.max(v2, axis=0)
                        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                            return 1.0 # Large positive, satisfies constraint
                            
                        return sat_gap(v1, v2)
                    return gap_func

                # We only add constraints for pairs that are currently close or overlapping
                # to keep the number of constraints manageable.
                if gap < 0.5: 
                    constraints.append({'type': 'ineq', 'fun': make_gap_closure(i, j)})

        return constraints

    def objective_side(params):
        return calc_outer_side(params)

    # --- GENERATE INITIAL CONFIGURATIONS ---
    inits = []
    
    # 1. Honeycomb lattice (Densest known packing)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append(np.concatenate([init_pos.flatten(), init_ang]))
    
    # 2. Ring configuration
    ring_pos = []
    ring_ang = []
    r = 2.0
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_ang.append(a)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    inits.append(np.concatenate([ring_pos.flatten(), np.array(ring_ang)]))
    
    # 3. Random starts (with some structure)
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append(np.concatenate([pos.flatten(), ang]))
        
    # 4. Clustered randoms
    for _ in range(5):
        centers = np.random.uniform(-2, 2, (4, 2))
        pos = []
        ang = []
        for i in range(N):
            c = centers[i % 4]
            offset = np.random.normal(0, 0.5, 2)
            pos.append(c + offset)
            ang.append(np.random.uniform(0, 2*math.pi))
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        inits.append(np.concatenate([pos.flatten(), np.array(ang)]))

    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    # --- PHASE 1: HARD CONSTRAINT OPTIMIZATION ---
    # We iterate through initial guesses and try to minimize the outer side length
    # subject to non-overlap constraints.
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Check if initial guess is valid
        pos0 = x0[:2*N].reshape(N, 2)
        ang0 = x0[2*N:]
        polys0 = [get_hex_verts(pos0[i][0], pos0[i][1], ang0[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys0[i], polys0[j]) < -1e-5:
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            # Try to fix it with a quick penalty minimization first? 
            # Or just skip. Let's skip to save time, as lattice starts are usually valid.
            # However, random starts might be invalid.
            # Let's run a quick SLSQP to make it valid first if invalid.
            pass # Skipping for now, relying on good inits

        try:
            # Define constraints for this specific run
            # To avoid recomputing constraints every step in SLSQP (which is slow),
            # we define them once based on the initial state or update them dynamically?
            # SLSQP calls the constraint function multiple times. 
            # My `make_gap_closure` is self-contained.
            
            # We need to be careful: if the optimizer moves points far apart, 
            # the set of "active" constraints might change. 
            # Standard approach: include all pairs? 12*11/2 = 66 constraints.
            # This is manageable for SLSQP in 36 dimensions?
            # Let's try including all pairs to be safe.
            
            constraints_all = []
            for i in range(N):
                for j in range(i+1, N):
                    def make_gap_closure(idx1, idx2):
                        def gap_func(p):
                            pos_p = p[:2*N].reshape(N, 2)
                            ang_p = p[2*N:]
                            v1 = get_hex_verts(pos_p[idx1][0], pos_p[idx1][1], ang_p[idx1])
                            v2 = get_hex_verts(pos_p[idx2][0], pos_p[idx2][1], ang_p[idx2])
                            
                            # Bounding box check
                            min_x1, min_y1 = np.min(v1, axis=0)
                            max_x1, max_y1 = np.max(v1, axis=0)
                            min_x2, min_y2 = np.min(v2, axis=0)
                            max_x2, max_y2 = np.max(v2, axis=0)
                            if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                                return 1.0 # Large positive, satisfies constraint
                                
                            return sat_gap(v1, v2)
                        return gap_func
                    constraints_all.append({'type': 'ineq', 'fun': make_gap_closure(i, j)})

            # Use SLSQP
            res = minimize(
                objective_side, 
                x0, 
                method='SLSQP', 
                bounds=None,
                constraints=constraints_all,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success:
                final_params = res.x
                final_side = res.fun
                
                # Verify validity
                pos_f = final_params[:2*N].reshape(N, 2)
                ang_f = final_params[2*N:]
                polys_f = [get_hex_verts(pos_f[i][0], pos_f[i][1], ang_f[i]) for i in range(N)]
                is_valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if sat_gap(polys_f[i], polys_f[j]) < -1e-5:
                            is_valid = False
                            break
                    if not is_valid: break
                
                if is_valid and final_side < best_side:
                    best_side = final_side
                    best_params = final_params.copy()
                    
        except Exception as e:
            # print(f"Error in optimization for init {idx}: {e}")
            pass

    # --- PHASE 2: POLISH OUTER HEXAGON ORIENTATION ---
    # The previous step optimized inner positions assuming outer hex is at (0,0) with angle 0.
    # We can further reduce the side length by rotating/centering the outer hexagon optimally.
    
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        ang = best_params[2*N:]
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            side = calc_outer_side(best_params, outer_center=oc, outer_angle=np.degrees(oa))
            return side

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

    # Fallback
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
        best_side = calc_outer_side(best_params)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    # Construct return values
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle

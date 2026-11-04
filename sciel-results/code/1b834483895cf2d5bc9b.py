from scipy.optimize import differential_evolution
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
DEADLINE = 145.0  # 5s margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def time_left():
        return deadline - time.time()

    # Unit hexagon vertices (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    # Unit hexagon vertices (flat-top, vertex at 30 deg)
    unit_verts_30 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 390, 60)])

    def get_hex_verts(cx, cy, verts_template):
        return verts_template + np.array([cx, cy])

    def calc_outer_side(params, outer_center=(0.0, 0.0), outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, o0..o11] where o is 0 or 1 (orientation index)
        """
        pos = params[:2*N].reshape(N, 2)
        orient_idx = params[2*N:].astype(int)
        
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)
        
        max_proj = 0.0
        oc = np.array(outer_center)
        
        for i in range(N):
            cx, cy = pos[i]
            verts = get_hex_verts(cx, cy, unit_verts_0 if orient_idx[i] == 0 else unit_verts_30)
            verts_rel = verts - oc
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / SQRT3
        return side

    def calc_overlap_penalty(pos, orient_idx, penalty_weight=10.0):
        """
        pos: (N, 2)
        orient_idx: (N,)
        """
        polys = []
        for i in range(N):
            verts = get_hex_verts(pos[i][0], pos[i][1], unit_verts_0 if orient_idx[i] == 0 else unit_verts_30)
            polys.append(verts)
            
        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                # Quick bounding box
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue
                
                # SAT
                min_overlap = 0.0
                axes = []
                for k in range(6):
                    edge = p1[(k+1)%6] - p1[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9:
                        axes.append(n/norm)
                for k in range(6):
                    edge = p2[(k+1)%6] - p2[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9:
                        axes.append(n/norm)
                
                for axis in axes:
                    proj1 = np.dot(p1, axis)
                    proj2 = np.dot(p2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < 0:
                        min_overlap = 0.0
                        break
                    if overlap > min_overlap:
                        min_overlap = overlap
                        
                if min_overlap > 0:
                    penalty += min_overlap * min_overlap * penalty_weight
                    
        return penalty

    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        orient_idx = params[2*N:].astype(int)
        side = calc_outer_side(params)
        penalty = calc_overlap_penalty(pos, orient_idx)
        return side + penalty

    # --- GENERATE INITIAL CONFIGURATIONS ---
    inits = []
    
    # 1. Honeycomb lattice (all orientation 0)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_orient = np.zeros(N, dtype=int)
    inits.append(np.concatenate([init_pos.flatten(), init_orient]))

    # 2. 3x4 Grid (all orientation 0)
    grid_pos = []
    for i in range(3):
        for j in range(4):
            x = i * 1.5
            y = j * SQRT3
            if i % 2 == 1:
                y += SQRT3 / 2
            grid_pos.append([x, y])
    grid_pos = np.array(grid_pos)
    grid_pos -= np.mean(grid_pos, axis=0)
    inits.append(np.concatenate([grid_pos.flatten(), np.zeros(N, dtype=int)]))

    # 3. Ring (alternating orientations)
    ring_pos = []
    ring_orient = []
    r = 2.0
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_orient.append(i % 2)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    inits.append(np.concatenate([ring_pos.flatten(), np.array(ring_orient)]))

    # 4. Random with fixed orientations
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (N, 2))
        orient = np.random.randint(0, 2, N)
        pos -= np.mean(pos, axis=0)
        inits.append(np.concatenate([pos.flatten(), orient]))

    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    # --- LOCAL OPTIMIZATION ---
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        pos0 = x0[:2*N].reshape(N, 2)
        orient0 = x0[2*N:].astype(int)
        
        def obj_pos(p):
            pos = p.reshape(N, 2)
            side = calc_outer_side(np.concatenate([pos.flatten(), orient0]))
            penalty = calc_overlap_penalty(pos, orient0)
            return side + penalty
            
        try:
            res = minimize(obj_pos, pos0.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_side:
                pos_res = res.x.reshape(N, 2)
                pen = calc_overlap_penalty(pos_res, orient0)
                if pen < 1e-4:
                    side_res = calc_outer_side(np.concatenate([res.x, orient0]))
                    if side_res < best_side:
                        best_side = side_res
                        best_params = np.concatenate([res.x, orient0])
        except:
            pass

    # --- GLOBAL SEARCH (Differential Evolution) ---
    if time_left() > 60:
        def de_obj_0(p):
            pos = p.reshape(N, 2)
            orient = np.zeros(N, dtype=int)
            side = calc_outer_side(np.concatenate([pos.flatten(), orient]))
            penalty = calc_overlap_penalty(pos, orient)
            return side + penalty
            
        bounds = [(-5, 5)] * (2*N)
        try:
            res_de = differential_evolution(de_obj_0, bounds, maxiter=50, popsize=10, tol=1e-4, seed=42, polish=False)
            if res_de.fun < best_side:
                pos_de = res_de.x.reshape(N, 2)
                pen = calc_overlap_penalty(pos_de, np.zeros(N, dtype=int))
                if pen < 1e-4:
                    side_de = calc_outer_side(np.concatenate([res_de.x, np.zeros(N, dtype=int)]))
                    if side_de < best_side:
                        best_side = side_de
                        best_params = np.concatenate([res_de.x, np.zeros(N, dtype=int)])
        except:
            pass

    if time_left() > 30:
        def de_obj_30(p):
            pos = p.reshape(N, 2)
            orient = np.ones(N, dtype=int)
            side = calc_outer_side(np.concatenate([pos.flatten(), orient]))
            penalty = calc_overlap_penalty(pos, orient)
            return side + penalty
            
        bounds = [(-5, 5)] * (2*N)
        try:
            res_de = differential_evolution(de_obj_30, bounds, maxiter=50, popsize=10, tol=1e-4, seed=43, polish=False)
            if res_de.fun < best_side:
                pos_de = res_de.x.reshape(N, 2)
                pen = calc_overlap_penalty(pos_de, np.ones(N, dtype=int))
                if pen < 1e-4:
                    side_de = calc_outer_side(np.concatenate([res_de.x, np.ones(N, dtype=int)]))
                    if side_de < best_side:
                        best_side = side_de
                        best_params = np.concatenate([res_de.x, np.ones(N, dtype=int)])
        except:
            pass

    # --- FINAL POLISH: Optimize outer center and angle ---
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        orient_idx = best_params[2*N:].astype(int)
        
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
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
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
    final_orient = best_params[2*N:].astype(int)
    
    inner = []
    for i in range(N):
        angle_deg = 0.0 if final_orient[i] == 0 else 30.0
        inner.append([final_pos[i][0], final_pos[i][1], angle_deg])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle

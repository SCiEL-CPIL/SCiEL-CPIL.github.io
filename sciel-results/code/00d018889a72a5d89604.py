import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            vx = cx + side * math.cos(theta)
            vy = cy + side * math.sin(theta)
            verts.append((vx, vy))
        return np.array(verts)

    def get_hex_normals(cx, cy, side, angle_deg):
        # Normals are perpendicular to edges, pointing outward
        # Edge midpoints are at distance apothem from center
        # Angle of normal k is angle_deg + 30 + k*60
        normals = []
        angle_rad = math.radians(angle_deg)
        for k in range(6):
            theta = angle_rad + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            normals.append((nx, ny))
        return np.array(normals)

    def support(verts, normals):
        # Returns max dot product for each normal
        # verts: (6, 2), normals: (M, 2)
        # result: (M,)
        return np.max(verts @ normals.T, axis=1)

    def anti_support(verts, normals):
        # Returns min dot product for each normal
        return np.min(verts @ normals.T, axis=1)

    def calc_gap(v_i, v_j, n_i, n_j):
        # v_i, v_j: vertices of hex i, j
        # n_i, n_j: normals of hex i, j
        all_normals = np.vstack([n_i, n_j])
        
        sup_i = support(v_i, all_normals)
        min_i = anti_support(v_i, all_normals)
        sup_j = support(v_j, all_normals)
        min_j = anti_support(v_j, all_normals)
        
        # Gap in direction n: min( min_j - sup_i, min_i - sup_j )
        gap_1 = min_j - sup_i
        gap_2 = min_i - sup_j
        gaps = np.minimum(gap_1, gap_2)
        
        return np.max(gaps)

    def objective_and_constraints(params):
        S = params[0]
        inner_params = params[1:]
        
        # Unpack
        verts_list = []
        normals_list = []
        for i in range(N):
            cx = inner_params[i*3]
            cy = inner_params[i*3 + 1]
            ang = inner_params[i*3 + 2]
            v = get_hex_vertices(cx, cy, 1.0, ang)
            n = get_hex_normals(cx, cy, 1.0, ang)
            verts_list.append(v)
            normals_list.append(n)
            
        # Containment Constraints
        # Outer hexagon centered at (0,0), side S, angle 0
        outer_normals = get_hex_normals(0.0, 0.0, S, 0.0)
        outer_apothem = S * APOTHEM
        
        constraints = []
        for i in range(N):
            sup_i = support(verts_list[i], outer_normals)
            # sup_i <= outer_apothem
            # outer_apothem - sup_i >= 0
            for k in range(6):
                constraints.append(outer_apothem - sup_i[k])
                
        # Non-overlap Constraints
        # Using soft-max to smooth the max gap
        beta = 20.0
        for i in range(N):
            for j in range(i+1, N):
                gap_max = calc_gap(verts_list[i], verts_list[j], normals_list[i], normals_list[j])
                # We want gap_max >= 0
                # If gap_max is negative, penalty.
                # In SLSQP, we provide g(x) >= 0.
                # So we just return gap_max.
                # However, gap_max is non-smooth.
                # Let's stick with exact max for now, SLSQP might handle it.
                # If it fails, we switch to penalty.
                constraints.append(gap_max)
                
        return S, constraints

    def solve_packing(init_params, beta_smooth=0.0):
        # init_params: [S, x1, y1, t1, ..., x11, y11, t11]
        # t in degrees
        
        def obj(x):
            S, cons = objective_and_constraints(x)
            # If we use penalty for non-smoothness, we'd add it here.
            # For now, return S. Constraints handled by SLSQP.
            return S
            
        def cons_func(x):
            S, cons = objective_and_constraints(x)
            # SLSQP expects a list of dicts or arrays.
            # We have a single array of inequality constraints g(x) >= 0.
            return {'type': 'ineq', 'fun': lambda x: cons} # This lambda captures cons from closure? No, x changes.
            
        # Proper constraint definition
        def constraints(x):
            S, cons = objective_and_constraints(x)
            return cons

        # Bounds
        # S > 0
        # Positions can be anywhere, but roughly within S*2
        bounds = [(0.1, 10.0)]  # S
        for i in range(N):
            bounds.append((-10, 10)) # x
            bounds.append((-10, 10)) # y
            bounds.append((0, 360)) # angle (redundant but safe)

        try:
            res = minimize(obj, init_params, method='SLSQP', bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                return res.x
            else:
                return None
        except Exception:
            return None

    # Initial Guesses
    # 1. Hexagonal Cluster
    # Center at 0,0. 6 around it. 4 in gaps.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3) ~ 1.732
    d = math.sqrt(3)
    
    def make_hex_cluster():
        pts = []
        # Center
        pts.append((0.0, 0.0, 0.0))
        # Ring 1
        for k in range(6):
            ang = k * 60.0
            x = d * math.cos(math.radians(ang))
            y = d * math.sin(math.radians(ang))
            pts.append((x, y, 0.0))
        # Ring 2 (4 spots)
        # Gaps are at 30, 90, 150, 210, 270, 330?
        # The gaps in the hexagonal lattice are at the "corners" of the central hex.
        # Distance to gap center is d * sqrt(3) ? No.
        # In hex lattice, distance to next shell is 2*d?
        # Let's just place them roughly.
        # Actually, for 11, a 3-4-3 or 2-3-3-2 arrangement might be better?
        # Let's try a 3x3 square-ish arrangement?
        # For now, use the cluster and let optimizer fix it.
        # Add 4 more at larger radius
        d2 = d * 1.8
        for k in range(4):
            ang = 30.0 + k * 90.0
            x = d2 * math.cos(math.radians(ang))
            y = d2 * math.sin(math.radians(ang))
            pts.append((x, y, 0.0))
            
        params = [4.0] # Initial S
        for p in pts:
            params.extend([p[0], p[1], p[2]])
        return params

    def make_random_guess(S_init=4.0):
        np.random.seed(42)
        params = [S_init]
        for i in range(N):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            ang = np.random.uniform(0, 360)
            params.extend([x, y, ang])
        return params

    best_solution = None
    best_S = float('inf')
    
    # Run multiple starts
    starts = [make_hex_cluster(), make_random_guess(4.0), make_random_guess(3.5)]
    
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        res_params = solve_packing(start)
        if res_params is not None:
            S_val = res_params[0]
            # Verify constraints
            _, cons = objective_and_constraints(res_params)
            if np.min(cons) > -1e-5: # Feasible
                if S_val < best_S:
                    best_S = S_val
                    best_solution = res_params

    # If no feasible solution found with SLSQP, try a simple grid/penalty fallback or return the best infeasible one scaled?
    # The problem requires a valid packing.
    # If best_solution is None, we must return a valid one.
    # Fallback: Use the initial cluster and calculate the required S.
    if best_solution is None:
        # Use the hex cluster, calculate S needed to contain them
        params = make_hex_cluster()
        # Force feasibility by increasing S
        S_test = 4.0
        while True:
            params[0] = S_test
            _, cons = objective_and_constraints(params)
            if np.min(cons) >= -1e-5:
                break
            S_test += 0.1
            if S_test > 10:
                break
        best_solution = params
        best_S = S_test

    # Extract results
    S_final = best_solution[0]
    inner = []
    for i in range(N):
        x = best_solution[1 + i*3]
        y = best_solution[1 + i*3 + 1]
        ang = best_solution[1 + i*3 + 2]
        inner.append([float(x), float(y), float(ang)])
        
    center = [0.0, 0.0]
    side = float(S_final)
    rotation = 0.0
    
    return inner, center, side, rotation

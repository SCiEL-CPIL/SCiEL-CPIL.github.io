import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute constants
    SQRT3 = math.sqrt(3)
    
    # Vertices of a unit hexagon (side 1) relative to center, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Distance from center to vertex is 1 (circumradius = side for regular hex)
    base_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
    ])
    
    # Normals for outer hexagon containment check (point in hex centered at 0)
    # The outer hexagon is defined by side length S and angle 0.
    # Its boundaries are at distance S * sqrt(3)/2 from center in normal directions.
    # Normal directions: 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([
        [math.cos(math.radians(30 + 60*i)), math.sin(math.radians(30 + 60*i))]
        for i in range(6)
    ])
    
    def get_vertices(cx, cy, angle_deg):
        ang = math.radians(angle_deg)
        c = math.cos(ang)
        s = math.sin(ang)
        R = np.array([[c, -s], [s, c]])
        # base_verts are (6, 2). Rotate them.
        rotated = base_verts @ R.T
        return rotated + np.array([cx, cy])

    def point_in_unit_hex_local(p):
        # p is (2,) in local coords of a unit hexagon centered at 0, angle 0
        # Check 6 inequalities.
        # d_k = dot(p, normal_k) <= sqrt(3)/2
        # Normals in local frame (same as outer_normals for angle 0)
        # Actually, for a unit hexagon, the apothem is sqrt(3)/2.
        # The inequality is dot(p, n_i) <= apothem for all i.
        # Since it's centered, we check abs(dot(p, n_i)) <= apothem.
        dots = p @ outer_normals.T
        return np.all(np.abs(dots) <= SQRT3/2 + 1e-9)

    def objective(params):
        # params: 12 * 3 = 36 floats
        # [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            v = get_vertices(x, y, a)
            verts_list.append(v)
        
        # 1. Boundary violation (Outer Hexagon)
        # We want to minimize the required outer side length S.
        # S is determined by the max projection of any vertex onto any outer normal.
        max_proj = 0.0
        for v in verts_list:
            # v is (6, 2)
            projs = v @ outer_normals.T
            # projs is (6, 6). We need max abs value.
            max_proj = max(max_proj, np.max(np.abs(projs)))
        
        # S = 2 / sqrt(3) * max_proj
        S_req = (2.0 / SQRT3) * max_proj
        
        # 2. Overlap Penalty
        # If any vertex of hex i is inside hex j, penalty.
        overlap_pen = 0.0
        for i in range(N):
            vi = verts_list[i]
            for j in range(i + 1, N):
                vj = verts_list[j]
                # Check if any vertex of vi is inside vj
                # Transform vi vertices to vj local frame
                # vj center: params[j*3], params[j*3+1]
                # vj angle: params[j*3+2]
                cx_j = params[j*3]
                cy_j = params[j*3+1]
                a_j = params[j*3+2]
                ang_j = math.radians(a_j)
                c_j = math.cos(-ang_j)
                s_j = math.sin(-ang_j)
                R_j_inv = np.array([[c_j, -s_j], [s_j, c_j]])
                
                # Translate and rotate vi vertices
                vi_rel = vi - np.array([cx_j, cy_j])
                vi_local = vi_rel @ R_j_inv.T
                
                # Check if any is inside unit hex
                # Use vectorized check
                dots = vi_local @ outer_normals.T
                inside = np.all(np.abs(dots) <= SQRT3/2 + 1e-9, axis=1)
                if np.any(inside):
                    overlap_pen += np.sum(inside) ** 2
                    
                # Check if any vertex of vj is inside vi
                cx_i = params[i*3]
                cy_i = params[i*3+1]
                a_i = params[i*3+2]
                ang_i = math.radians(a_i)
                c_i = math.cos(-ang_i)
                s_i = math.sin(-ang_i)
                R_i_inv = np.array([[c_i, -s_i], [s_i, c_i]])
                
                vj_rel = vj - np.array([cx_i, cy_i])
                vj_local = vj_rel @ R_i_inv.T
                
                dots = vj_local @ outer_normals.T
                inside = np.all(np.abs(dots) <= SQRT3/2 + 1e-9, axis=1)
                if np.any(inside):
                    overlap_pen += np.sum(inside) ** 2
        
        return S_req + 100.0 * overlap_pen

    # Initial guesses
    def init_grid():
        # 4x3 grid of hexagons
        # Distance between centers in grid:
        # Horizontal: 2 * cos(30) = sqrt(3) ? No, side 1 hexes.
        # If aligned, horizontal distance between centers is 2 * cos(30) = sqrt(3) if touching sides?
        # Actually, if angle 0, vertices at 0, 60...
        # Side length 1. Distance between parallel sides is sqrt(3).
        # So if we place them in a grid with spacing sqrt(3) in y and 2*sqrt(3)/2 * 2?
        # Let's just use a rough grid.
        params = []
        for i in range(4):
            for j in range(3):
                x = 1.8 * (i - 1.5)
                y = 1.7 * (j - 1.0)
                a = 0.0
                params.extend([x, y, a])
        return np.array(params)

    def init_honeycomb():
        # 1 center, 6 ring, 5 more?
        params = []
        params.extend([0, 0, 0])
        for k in range(6):
            ang = math.radians(60 * k)
            x = 1.7 * math.cos(ang)
            y = 1.7 * math.sin(ang)
            params.extend([x, y, 0])
        # 5 more on the outside
        for k in range(5):
            ang = math.radians(60 * (k + 1) + 30)
            x = 3.0 * math.cos(ang)
            y = 3.0 * math.sin(ang)
            params.extend([x, y, 0])
        return np.array(params)

    best_score = float('inf')
    best_params = None
    
    inits = [init_grid(), init_honeycomb()]
    
    # Add a few random inits
    for _ in range(3):
        params = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 180)
            params.extend([x, y, a])
        inits.append(np.array(params))

    for init in inits:
        if time.time() > deadline - 5:
            break
        try:
            # Use Nelder-Mead with limited iterations
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        best_params = init_grid()
        
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
    
    # Calculate final S
    verts_list = []
    for i in range(N):
        v = get_vertices(inner[i][0], inner[i][1], inner[i][2])
        verts_list.append(v)
        
    max_proj = 0
    for v in verts_list:
        projs = v @ outer_normals.T
        max_proj = max(max_proj, np.max(np.abs(projs)))
        
    S = (2.0 / SQRT3) * max_proj
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END

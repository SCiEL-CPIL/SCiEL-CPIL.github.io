import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    sqrt3 = np.sqrt(3)
    
    # Base vertices for unit hexagon (side 1), pointy top
    # Angles: 90, 30, -30, -90, -150, 150
    base_vertices = np.array([
        [0.0, 1.0],
        [sqrt3/2, 0.5],
        [sqrt3/2, -0.5],
        [0.0, -1.0],
        [-sqrt3/2, -0.5],
        [-sqrt3/2, 0.5]
    ])
    
    def get_hex_vertices(cx, cy, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return base_vertices @ R.T + np.array([cx, cy])

    def objective(params):
        xs = params[0:36:3]
        ys = params[1:36:3]
        as_ = params[2:36:3]
        theta_out = params[36]
        
        # Calculate required S
        max_S = 0.0
        for i in range(N):
            verts = get_hex_vertices(xs[i], ys[i], as_[i])
            angles = theta_out + np.pi/6 + np.arange(6) * np.pi/3
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            projs = np.abs(verts @ normals.T)
            max_proj = np.max(projs)
            S_i = max_proj / (sqrt3/2)
            if S_i > max_S:
                max_S = S_i
        
        # Calculate overlap penalties
        penalty = 0.0
        # Precompute normals for unit hexagon
        angles_unit = np.pi/6 + np.arange(6) * np.pi/3
        normals_unit = np.column_stack([np.cos(angles_unit), np.sin(angles_unit)])
        apothem_unit = sqrt3 / 2
        
        for i in range(N):
            vi = get_hex_vertices(xs[i], ys[i], as_[i])
            for j in range(i+1, N):
                vj = get_hex_vertices(xs[j], ys[j], as_[j])
                
                overlap = False
                # Check i inside j
                c_j, s_j = np.cos(-as_[j]), np.sin(-as_[j])
                R_j = np.array([[c_j, -s_j], [s_j, c_j]])
                # Translate to j's center
                vi_local = (vi - np.array([xs[j], ys[j]])) @ R_j.T
                projs_j = np.abs(vi_local @ normals_unit.T)
                if np.all(projs_j <= apothem_unit + 1e-9):
                    # If all vertices of i are inside j, they overlap
                    # Actually, if ANY vertex is inside, they might overlap, but not necessarily.
                    # If one vertex is inside, and the polygons are convex, they overlap.
                    overlap = True
                else:
                    # Check if any single vertex is inside
                    # The check above `np.all` is wrong for "any vertex inside".
                    # We need `np.any` of the condition "vertex is inside".
                    # Vertex p is inside if ALL its projections are <= apothem.
                    # So for a vertex, `np.all(projs_vertex <= apothem)`.
                    # We need `np.any` over vertices.
                    pass
                
                # Corrected Overlap Check
                overlap = False
                # Check vertices of i inside j
                for p in vi_local:
                    if np.all(np.abs(p @ normals_unit.T) <= apothem_unit + 1e-9):
                        overlap = True
                        break
                        
                if not overlap:
                    # Check vertices of j inside i
                    c_i, s_i = np.cos(-as_[i]), np.sin(-as_[i])
                    R_i = np.array([[c_i, -s_i], [s_i, c_i]])
                    vj_local = (vj - np.array([xs[i], ys[i]])) @ R_i.T
                    for p in vj_local:
                        if np.all(np.abs(p @ normals_unit.T) <= apothem_unit + 1e-9):
                            overlap = True
                            break
                            
                if overlap:
                    penalty += 100.0
        
        return max_S + penalty

    # Initialization
    init_x = []
    init_y = []
    init_a = [0.0]*N
    
    for k in range(4):
        init_x.append(k * sqrt3)
        init_y.append(0.0)
    for k in range(4):
        init_x.append(sqrt3/2 + k * sqrt3)
        init_y.append(sqrt3)
    for k in range(4):
        init_x.append(k * sqrt3)
        init_y.append(2 * sqrt3)
        
    cx = np.mean(init_x)
    cy = np.mean(init_y)
    init_x = [x - cx for x in init_x]
    init_y = [y - cy for y in init_y]
    
    params = []
    for i in range(N):
        params.append(init_x[i])
        params.append(init_y[i])
        params.append(init_a[i])
    params.append(0.0)
    
    # Optimize
    try:
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        best_params = res.x
        best_S = res.fun
    except Exception:
        best_params = np.array(params)
        best_S = objective(params)
        
    # Check time
    if time.time() > deadline:
        # Return current best
        pass
        
    xs = best_params[0:36:3]
    ys = best_params[1:36:3]
    as_ = best_params[2:36:3]
    theta_out = best_params[36]
    
    inner = []
    for i in range(N):
        inner.append([float(xs[i]), float(ys[i]), float(np.degrees(as_[i]))])
        
    return inner, (0.0, 0.0), float(best_S), float(np.degrees(theta_out))

# EVOLVE-BLOCK-END

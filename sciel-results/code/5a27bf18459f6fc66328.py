# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _compute_all_vertices(centers, angles_deg, side=1.0):
    n = len(centers)
    verts = np.empty((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
    return verts

def _min_outer_side(vertices_all):
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    if global_max < 0:
        global_max = 0.0
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _min_outer_side_smooth(vertices_all, beta=50.0):
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    max_per_vertex = np.max(dots, axis=1)
    max_val = np.max(max_per_vertex)
    if max_val < 0:
        max_val = 0.0
    shifted = max_per_vertex - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted)) / len(max_per_vertex)) / beta
    s_min = (2.0 / np.sqrt(3.0)) * log_sum_exp
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    n = len(centers)
    vertices_list = []
    normals_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _initial_honeycomb():
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _objective_smooth(params, beta=50.0):
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side_smooth(all_verts, beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _objective_exact(params):
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _solve_lp_fixed_angles(centers_init):
    """
    Solve LP for fixed angle 0.
    Variables: x0..x11, y0..y11, s. Total 25 variables.
    Minimize s.
    Constraints:
    1. Containment: n_k . c_i + h <= s * h_outer.
       h = sqrt(3)/2. h_outer = s * sqrt(3)/2.
       n_k . c_i <= s * sqrt(3)/2 - sqrt(3)/2.
    2. Non-overlap: For each pair (i,j), assume separation direction k_ij.
       n_k . (c_i - c_j) >= sqrt(3) OR <= -sqrt(3).
       We determine k_ij from the initial configuration.
    """
    from scipy.optimize import linprog

    n = 12
    h = np.sqrt(3.0) / 2.0
    
    # Normals for angle 0
    # Vertices at 0, 60, ...
    # Edge 0: (1,0) to (0.5, 0.866). Normal (0.866, 0.5) -> 30 deg.
    # Edge 1: (0.5, 0.866) to (-0.5, 0.866). Normal (0, 1) -> 90 deg.
    # ...
    normals = np.column_stack([
        np.cos(np.pi/6 + k*np.pi/3),
        np.sin(np.pi/6 + k*np.pi/3)
    ])
    
    # Determine separation directions from init
    # c_init is (12, 2)
    sep_dirs = np.zeros((n, n, 6)) # Not used, just store the chosen normal index and sign
    
    # Build A_ub, b_ub
    # Minimize s: c_obj = [0...0, 1] (last variable is s)
    c_obj = np.zeros(2*n + 1)
    c_obj[-1] = 1.0
    
    A_ub = []
    b_ub = []
    
    # Containment constraints
    # n_k . c_i - s * h <= -h
    # n_k . c_i - s * h + h <= 0
    for i in range(n):
        for k in range(6):
            row = np.zeros(2*n + 1)
            row[i*2] = normals[k, 0]
            row[i*2+1] = normals[k, 1]
            row[-1] = -h
            A_ub.append(row)
            b_ub.append(-h)
            
    # Non-overlap constraints
    # We need to guess the separation normal for each pair.
    # Use the initial configuration to determine which normal is "active".
    # For pair (i, j), compute d = c_i - c_j.
    # Check which normal n_k has max |n_k . d|.
    # If n_k . d > 0, constraint is n_k . (c_i - c_j) >= sqrt(3).
    # If n_k . d < 0, constraint is n_k . (c_i - c_j) <= -sqrt(3).
    
    for i in range(n):
        for j in range(i+1, n):
            d = centers_init[i] - centers_init[j]
            dots = np.abs(normals @ d)
            k_max = np.argmax(dots)
            n_k = normals[k_max]
            dot_val = n_k @ d
            
            row = np.zeros(2*n + 1)
            if dot_val > 0:
                # n_k . (c_i - c_j) >= sqrt(3)
                # -n_k . (c_i - c_j) <= -sqrt(3)
                row[i*2] = -n_k[0]
                row[i*2+1] = -n_k[1]
                row[j*2] = n_k[0]
                row[j*2+1] = n_k[1]
                A_ub.append(row)
                b_ub.append(-np.sqrt(3.0))
            else:
                # n_k . (c_i - c_j) <= -sqrt(3)
                # n_k . (c_i - c_j) <= -sqrt(3)
                row[i*2] = n_k[0]
                row[i*2+1] = n_k[1]
                row[j*2] = -n_k[0]
                row[j*2+1] = -n_k[1]
                A_ub.append(row)
                b_ub.append(-np.sqrt(3.0))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: s > 0. Centers can be anything.
    bounds = [(None, None)] * (2*n) + [(0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
            c = x[:2*n].reshape(n, 2)
            s = x[-1]
            # Verify
            if _check_nonoverlap(c, np.zeros(n)):
                return c, s
    except Exception:
        pass
    return None, None

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0
    
    centers, angles_deg = _initial_honeycomb()
    params = np.column_stack([centers, angles_deg]).flatten()
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts)
        best_params = params.copy()
        
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: SA
    T = 0.5
    cooling = 0.99985
    step_size = 0.3
    angle_step = 5.0
    accept_count = 0
    total_count = 0
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 400000
    for it in range(max_iters):
        if time.time() > deadline - 250.0:
            break
        i = random.randint(0, 11)
        old_cx = current_params[i*3]
        old_cy = current_params[i*3+1]
        old_ang = current_params[i*3+2]
        new_cx = old_cx + np.random.randn() * step_size
        new_cy = old_cy + np.random.randn() * step_size
        new_ang = old_ang + np.random.randn() * angle_step
        current_params[i*3] = new_cx
        current_params[i*3+1] = new_cy
        current_params[i*3+2] = new_ang
        centers_new = current_params.reshape(12, 3)[:, :2]
        angles_new = current_params.reshape(12, 3)[:, 2]
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new)
        if not _check_nonoverlap(centers_new, angles_new):
            s_new += 50.0
        delta = s_new - current_s
        total_count += 1
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_s = s_new
            accept_count += 1
            if s_new < best_s:
                best_s = s_new
                best_params = current_params.copy()
        else:
            current_params[i*3] = old_cx
            current_params[i*3+1] = old_cy
            current_params[i*3+2] = old_ang
        T *= cooling
        if it % 500 == 0 and it > 0:
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.15:
                step_size *= 0.85
                angle_step *= 0.85
            elif acc_rate > 0.75:
                step_size *= 1.15
                angle_step *= 1.15
            step_size = np.clip(step_size, 0.01, 1.5)
            angle_step = np.clip(angle_step, 0.5, 15.0)
            accept_count = 0
            total_count = 0
            
    # Phase 2: LP Polish (Fixed Angles)
    # Try to improve by fixing angles to 0 and solving LP
    if time.time() < deadline - 100.0:
        c_best = best_params.reshape(12, 3)[:, :2]
        # Try LP with current centers as init
        c_lp, s_lp = _solve_lp_fixed_angles(c_best)
        if c_lp is not None and s_lp < best_s:
            best_s = s_lp
            best_params = np.column_stack([c_lp, np.zeros(12)]).flatten()
            
        # Try LP with initial honeycomb as init
        c_init, _ = _initial_honeycomb()
        c_lp2, s_lp2 = _solve_lp_fixed_angles(c_init)
        if c_lp2 is not None and s_lp2 < best_s:
            best_s = s_lp2
            best_params = np.column_stack([c_lp2, np.zeros(12)]).flatten()

    # Phase 3: Nelder-Mead on exact objective
    from scipy.optimize import minimize
    if time.time() < deadline - 50.0:
        x0 = best_params.copy()
        try:
            res = minimize(_objective_exact, x0, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            if _check_nonoverlap(res.x.reshape(12,3)[:, :2], res.x.reshape(12,3)[:, 2]):
                s_nm = _min_outer_side(_compute_all_vertices(res.x.reshape(12,3)[:, :2], res.x.reshape(12,3)[:, 2]))
                if s_nm < best_s:
                    best_s = s_nm
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Fine polish
    for _ in range(40000):
        if time.time() > deadline - 30.0:
            break
        trial = best_params + np.random.randn(36) * 0.015
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
            
    # Phase 5: Ultra-fine polish
    for _ in range(25000):
        if time.time() > deadline - 10.0:
            break
        trial = best_params + np.random.randn(36) * 0.004
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
            
    # Phase 6: Very fine polish
    for _ in range(15000):
        if time.time() > deadline - 3.0:
            break
        trial = best_params + np.random.randn(36) * 0.001
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
            
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final)
    
    if not _check_nonoverlap(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
        
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END

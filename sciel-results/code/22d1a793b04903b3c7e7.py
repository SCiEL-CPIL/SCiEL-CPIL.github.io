# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    k = np.arange(6)
    outer_angle = np.radians(outer_angle_deg)
    angles = outer_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    if global_max < 0:
        global_max = 0.0
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _soft_min_outer_side(vertices_all, outer_angle_deg=0.0, beta=50.0):
    k = np.arange(6)
    outer_angle = np.radians(outer_angle_deg)
    angles = outer_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    beta_dots = beta * dots
    max_beta_dots = np.max(beta_dots)
    if max_beta_dots > 700:
        return (2.0 / np.sqrt(3.0)) * (max_beta_dots / beta)
    soft_max = max_beta_dots / beta + np.log(np.sum(np.exp(beta_dots - max_beta_dots))) / beta
    if soft_max < 0:
        soft_max = 0.0
    s_min = (2.0 / np.sqrt(3.0)) * soft_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    n = len(centers)
    all_verts = np.zeros((n, 6, 2))
    all_normals = np.zeros((n, 6, 2))
    
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        all_verts[i] = verts
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                all_normals[i, e] = [nx/norm, ny/norm]
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = all_verts[i]
            v2 = all_verts[j]
            overlap = True
            for hex_idx in range(2):
                normals = all_normals[i] if hex_idx == 0 else all_normals[j]
                proj1 = v1 @ normals.T
                proj2 = v2 @ normals.T
                min1 = np.min(proj1, axis=0)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                separated = (max1 + tol < min2) | (max2 + tol < min1)
                if np.any(separated):
                    overlap = False
                    break
            if overlap:
                return False
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

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
    s = _soft_min_outer_side(all_verts, outer_angle_deg=0.0, beta=beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 610.0
    
    centers, angles_deg = _initial_honeycomb()
    params = np.column_stack([centers, angles_deg]).flatten()
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts, outer_angle_deg=0.0)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts, outer_angle_deg=0.0)
        best_params = params.copy()
    
    # Phase 1: Simulated annealing with adaptive step size
    np.random.seed(42)
    T = 0.5
    cooling = 0.9998
    step_size = 0.3
    accept_count = 0
    total_count = 0
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 300000
    for it in range(max_iters):
        if time.time() > deadline - 200.0:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        
        centers_new = new_params.reshape(12, 3)[:, :2]
        angles_new = new_params.reshape(12, 3)[:, 2]
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new)
        
        if not _check_nonoverlap(centers_new, angles_new):
            s_new += 50.0
        
        delta = s_new - current_s
        total_count += 1
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_params = new_params
            current_s = s_new
            accept_count += 1
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        
        if it % 500 == 0 and it > 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.2:
                step_size *= 0.8
            elif acc_rate > 0.8:
                step_size *= 1.2
            step_size = np.clip(step_size, 0.005, 1.0)
            accept_count = 0
            total_count = 0
    
    # Phase 2: Gradient-based optimization with smooth surrogate
    from scipy.optimize import minimize
    
    n_starts = 8
    for si in range(n_starts):
        if time.time() > deadline - 80.0:
            break
        
        if si == 0:
            x0 = best_params.copy()
        elif si == 1:
            x0 = best_params + np.random.randn(36) * 0.05
        elif si == 2:
            x0 = best_params + np.random.randn(36) * 0.1
        elif si == 3:
            x0 = best_params + np.random.randn(36) * 0.2
        else:
            x0 = best_params + np.random.randn(36) * (0.1 + 0.05 * si)
        
        try:
            result = minimize(
                _objective_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 800, 'maxls': 80, 'ftol': 1e-12, 'gtol': 1e-10},
                tol=1e-12
            )
            
            c = result.x.reshape(12, 3)[:, :2]
            a = result.x.reshape(12, 3)[:, 2]
            if _check_nonoverlap(c, a):
                s_exact = _min_outer_side(_compute_all_vertices(c, a))
                if s_exact < best_s:
                    best_s = s_exact
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine polish with small random steps
    for _ in range(40000):
        if time.time() > deadline - 40.0:
            break
        trial = best_params + np.random.randn(36) * 0.02
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), outer_angle_deg=0.0)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 4: Ultra-fine polish
    for _ in range(30000):
        if time.time() > deadline - 15.0:
            break
        trial = best_params + np.random.randn(36) * 0.005
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), outer_angle_deg=0.0)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 5: Very fine polish
    for _ in range(20000):
        if time.time() > deadline - 5.0:
            break
        trial = best_params + np.random.randn(36) * 0.001
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), outer_angle_deg=0.0)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    
    # Search for best outer angle
    best_outer_angle = 0.0
    best_s_for_angle = _min_outer_side(all_verts_final, outer_angle_deg=0.0)
    for oa in np.linspace(0, 60, 121):
        s_test = _min_outer_side(all_verts_final, outer_angle_deg=oa)
        if s_test < best_s_for_angle:
            best_s_for_angle = s_test
            best_outer_angle = oa
    
    # Fine search around best angle
    for oa in np.linspace(best_outer_angle - 3, best_outer_angle + 3, 61):
        s_test = _min_outer_side(all_verts_final, outer_angle_deg=oa)
        if s_test < best_s_for_angle:
            best_s_for_angle = s_test
            best_outer_angle = oa
    
    s_final = best_s_for_angle
    
    if not _check_nonoverlap(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final, outer_angle_deg=0.0)
        best_outer_angle = 0.0
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END

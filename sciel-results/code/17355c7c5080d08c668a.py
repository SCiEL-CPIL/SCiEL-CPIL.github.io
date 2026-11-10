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

def _min_outer_side_smooth(vertices_all, outer_angle_deg=0.0, beta=50.0):
    k = np.arange(6)
    outer_angle = np.radians(outer_angle_deg)
    angles = outer_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T  # (72, 6)
    
    max_per_vertex = np.max(dots, axis=1)  # (72,)
    
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
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    normals_list = []
    for verts in vertices_list:
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

def _penetration_energy(centers, angles_deg, side=1.0, stiff=100.0):
    """Compute sum of squared penetration depths between all pairs."""
    n = len(centers)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    
    total_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            v1 = verts_list[i]
            v2 = verts_list[j]
            # SAT: find minimum penetration
            min_pen = np.inf
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                for e in range(6):
                    p1 = verts[e]
                    p2 = verts[(e+1) % 6]
                    edge = p2 - p1
                    nx, ny = edge[1], -edge[0]
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm < 1e-12:
                        continue
                    normal = np.array([nx/norm, ny/norm])
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    # Penetration depth = overlap
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap > 0:
                        min_pen = min(min_pen, overlap)
            if min_pen < np.inf and min_pen > 0:
                total_pen += min_pen ** 2
    return stiff * total_pen

def _objective_energy(params, beta=50.0, stiff=100.0):
    """Combined objective: smooth outer size + penetration penalty."""
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side_smooth(all_verts, outer_angle_deg=0.0, beta=beta)
    
    pen = _penetration_energy(centers, angles_deg, stiff=stiff)
    
    return s + pen

def _objective_exact(params):
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg=0.0)
    
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
    
    # Phase 1: Energy-based gradient optimization (pressure model)
    from scipy.optimize import minimize
    
    np.random.seed(42)
    
    # Multiple starts with energy-based objective
    for start_idx in range(10):
        if time.time() > deadline - 150.0:
            break
        
        if start_idx == 0:
            x0 = best_params.copy()
        elif start_idx == 1:
            x0 = best_params + np.random.randn(36) * 0.05
        elif start_idx == 2:
            x0 = best_params + np.random.randn(36) * 0.1
        elif start_idx == 3:
            x0 = best_params + np.random.randn(36) * 0.15
        elif start_idx == 4:
            x0 = best_params + np.random.randn(36) * 0.2
        else:
            x0 = best_params + np.random.randn(36) * (0.1 + 0.03 * start_idx)
        
        try:
            result = minimize(
                _objective_energy,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-12},
                tol=1e-14
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
    
    # Phase 2: Simulated annealing with energy-based acceptance
    T = 0.3
    cooling = 0.99985
    step_size = 0.25
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 400000
    accept_count = 0
    total_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 100.0:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        
        centers_new = new_params.reshape(12, 3)[:, :2]
        angles_new = new_params.reshape(12, 3)[:, 2]
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new)
        
        if not _check_nonoverlap(centers_new, angles_new):
            # Add energy penalty
            pen = _penetration_energy(centers_new, angles_new, stiff=50.0)
            s_new += pen + 20.0
        
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
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.15:
                step_size *= 0.85
            elif acc_rate > 0.7:
                step_size *= 1.15
            step_size = np.clip(step_size, 0.003, 0.8)
            accept_count = 0
            total_count = 0
    
    # Phase 3: High-precision local polish with tight tolerances
    for _ in range(50000):
        if time.time() > deadline - 40.0:
            break
        trial = best_params + np.random.randn(36) * 0.015
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), outer_angle_deg=0.0)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 4: Ultra-fine polish
    for _ in range(40000):
        if time.time() > deadline - 15.0:
            break
        trial = best_params + np.random.randn(36) * 0.004
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), outer_angle_deg=0.0)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 5: Very fine polish
    for _ in range(25000):
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
    
    # Phase 6: Final L-BFGS polish on exact objective from best point
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective_exact,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-13},
                tol=1e-15
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

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _outer_side_length_from_positions(positions, angles_deg):
    n = len(positions)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    s = max_proj / apothem_factor
    return max(s, np.sqrt(12) - 1e-12)

def _smooth_objective(params, beta=50.0):
    """Soft-max surrogate: smooth approximation of the max projection across all
    inner-vertex / outer-normal pairs.  Returns (value, gradient) w.r.t. params."""
    n = 12
    pos = np.column_stack([params[0::3], params[1::3]])
    ang = params[2::3]
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2

    # Collect all (vertex, normal) projection values and their Jacobians.
    # For each inner hex i and each vertex k (6) and each normal m (6):
    #   proj = v_{i,k} . n_m
    #   v_{i,k} = pos_i + (cos(a_i + k*pi/3), sin(a_i + k*pi/3))
    # dproj/dpos_i = n_m
    # dproj/dang_i = -sin(a_i+k*pi/3)*n_m[0] + cos(a_i+k*pi/3)*n_m[1]
    # We accumulate log-sum-exp over all 12*6*6 = 432 terms.
    total_terms = n * 6 * 6
    projs = np.empty(total_terms)
    idx = 0
    for i in range(n):
        a = np.radians(ang[i])
        for k in range(6):
            theta = a + k * np.pi / 3
            vx = pos[i, 0] + np.cos(theta)
            vy = pos[i, 1] + np.sin(theta)
            for m in range(6):
                projs[idx] = vx * normals[m, 0] + vy * normals[m, 1]
                idx += 1

    # log-sum-exp for stability
    pmax = np.max(projs)
    if not np.isfinite(pmax):
        return 1e10, np.zeros_like(params)
    logsum = pmax + np.log(np.sum(np.exp(projs - pmax)))
    # soft-max value = logsum / apothem_factor
    value = logsum / apothem_factor

    # Gradient: softmax weights
    w = np.exp(projs - pmax)
    w /= w.sum()

    grad = np.zeros_like(params)
    idx = 0
    for i in range(n):
        a = np.radians(ang[i])
        for k in range(6):
            theta = a + k * np.pi / 3
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            for m in range(6):
                wi = w[idx] / apothem_factor
                grad[3*i]     += wi * normals[m, 0]
                grad[3*i + 1] += wi * normals[m, 1]
                grad[3*i + 2] += wi * (-sin_t * normals[m, 0] + cos_t * normals[m, 1])
                idx += 1
    return value, grad

def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    # SAT check
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            verts_j = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            separated = False
            # Normals of i
            for k in range(6):
                edge = verts_i[(k + 1) % 6] - verts_i[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                        separated = True
                        break
            if not separated:
                # Normals of j
                for k in range(6):
                    edge = verts_j[(k + 1) % 6] - verts_j[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        proj_i = verts_i @ normal
                        proj_j = verts_j @ normal
                        if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                            separated = True
                            break
            if not separated:
                return False
    # Containment
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -tol:
                    return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Seed arrangement (valid)
    seed_positions = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0, -4]
    ])
    seed_angles = np.zeros(12)
    
    # Hexagonal grid arrangement
    hex_positions = np.array([
        [0, 0],
        [3, 0],
        [1.5, sqrt3],
        [-1.5, sqrt3],
        [-3, 0],
        [-1.5, -sqrt3],
        [1.5, -sqrt3],
        [4.5, sqrt3],
        [3, 2*sqrt3],
        [0, 2*sqrt3],
        [-3, 2*sqrt3],
        [-4.5, sqrt3]
    ])
    hex_angles = np.zeros(12)
    
    # List of starting configs
    configs = []
    configs.append((seed_positions.copy(), seed_angles.copy()))
    configs.append((hex_positions.copy(), hex_angles.copy()))
    
    # Rotated hex grid
    for rot in [15, 30, 45]:
        ang = np.full(12, rot)
        configs.append((hex_positions.copy(), ang))
    
    # Perturbed hex grid
    for t in range(20):
        np.random.seed(t * 100 + 42)
        pos = hex_positions + np.random.randn(12, 2) * 0.1
        ang = np.random.uniform(-30, 30, 12)
        configs.append((pos, ang))
    
    best_params = None
    best_s = float('inf')
    
    def objective(params):
        pos = np.column_stack([params[0::3], params[1::3]])
        ang = params[2::3]
        return _outer_side_length_from_positions(pos, ang)
    
    def smooth_obj(params):
        return _smooth_objective(params, beta=50.0)[0]
    
    def smooth_obj_grad(params):
        val, grad = _smooth_objective(params, beta=50.0)
        return val, grad

    for idx, (pos, ang) in enumerate(configs):
        if time.time() > deadline - 60:
            break
        
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        current_s = _outer_side_length_from_positions(pos, ang)
        
        # Phase 1: BFGS with smooth surrogate (fast, gradient-based)
        if time.time() < deadline - 45:
            try:
                from scipy.optimize import minimize
                res = minimize(smooth_obj_grad, p, method='L-BFGS-B',
                               options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
                if res.fun < current_s:
                    p = res.x
                    current_s = _outer_side_length_from_positions(
                        np.column_stack([p[0::3], p[1::3]]), p[2::3])
            except Exception:
                pass
        
        # Phase 2: Nelder-Mead on the exact (nonsmooth) objective
        if time.time() < deadline - 30:
            try:
                from scipy.optimize import minimize
                res = minimize(objective, p, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if res.fun < current_s:
                    current_s = res.fun
                    p = res.x
            except Exception:
                pass
        
        # Phase 3: Powell on the exact objective
        if time.time() < deadline - 15:
            try:
                from scipy.optimize import minimize
                res2 = minimize(objective, p, method='Powell',
                                options={'maxiter': 1500, 'xtol': 1e-10, 'ftol': 1e-12})
                if res2.fun < current_s:
                    current_s = res2.fun
                    p = res2.x
            except Exception:
                pass
        
        if current_s < best_s:
            best_s = current_s
            best_params = p.copy()
            
    if best_params is None:
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3*i] = seed_positions[i, 0]
            best_params[3*i+1] = seed_positions[i, 1]
            best_params[3*i+2] = seed_angles[i]
        best_s = _outer_side_length_from_positions(seed_positions, seed_angles)
    
    # Final high-precision polish: smooth BFGS then exact Nelder-Mead
    if time.time() < deadline - 15:
        try:
            from scipy.optimize import minimize
            res_final = minimize(smooth_obj_grad, best_params, method='L-BFGS-B',
                                 options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
            p_polish = res_final.x
            s_polish = _outer_side_length_from_positions(
                np.column_stack([p_polish[0::3], p_polish[1::3]]), p_polish[2::3])
            if s_polish < best_s:
                best_s = s_polish
                best_params = p_polish.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            res_final2 = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res_final2.fun < best_s:
                best_s = res_final2.fun
                best_params = res_final2.x.copy()
        except Exception:
            pass
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_params[3*i]
        inner_hex_data[i, 1] = best_params[3*i+1]
        inner_hex_data[i, 2] = best_params[3*i+2]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    s = _outer_side_length_from_positions(
        np.column_stack([best_params[0::3], best_params[1::3]]),
        best_params[2::3]
    )
    
    if not _check_validity(inner_hex_data, outer_hex_data, s):
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END

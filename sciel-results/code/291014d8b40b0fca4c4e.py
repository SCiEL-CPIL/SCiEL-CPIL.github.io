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

def _sat_overlap_penalty(verts_i, verts_j, tol=1e-6):
    """Return max penetration depth (0 if separated)."""
    max_pen = 0.0
    for k in range(6):
        edge = verts_i[(k + 1) % 6] - verts_i[k]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
            proj_i = verts_i @ normal
            proj_j = verts_j @ normal
            pen = np.max(proj_i) - np.min(proj_j)
            if pen > max_pen:
                max_pen = pen
    for k in range(6):
        edge = verts_j[(k + 1) % 6] - verts_j[k]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
            proj_i = verts_i @ normal
            proj_j = verts_j @ normal
            pen = np.max(proj_i) - np.min(proj_j)
            if pen > max_pen:
                max_pen = pen
    return max(0.0, max_pen)

def _containment_violation(verts_inner, outer_verts, tol=1e-6):
    """Return max violation of containment (0 if all inside)."""
    max_viol = 0.0
    for v in verts_inner:
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -tol:
                max_viol = max(max_viol, -cross)
    return max_viol

def _sa_objective(params, s_current):
    """SA energy: outer side length + penalties for overlap and containment."""
    pos = np.column_stack([params[0::3], params[1::3]])
    ang = params[2::3]
    
    s = _outer_side_length_from_positions(pos, ang)
    
    # Compute overlap penalty
    overlap_pen = 0.0
    for i in range(12):
        verts_i = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        for j in range(i + 1, 12):
            verts_j = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, ang[j])
            overlap_pen += _sat_overlap_penalty(verts_i, verts_j)
    
    # Compute containment violation (use current s to define outer hex)
    contain_pen = 0.0
    outer_verts = _hex_vertices(0.0, 0.0, s, 0.0)
    for i in range(12):
        verts_i = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        contain_pen += _containment_violation(verts_i, outer_verts)
    
    return s + 100.0 * overlap_pen + 100.0 * contain_pen

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

def _simulated_annealing(params0, s_init, max_iter=50000, t0=1.0, t_end=0.01, deadline=None):
    """Run SA to minimize the SA objective."""
    np.random.seed(42)
    current = params0.copy()
    current_s = s_init
    current_energy = _sa_objective(current, current_s)
    
    best = current.copy()
    best_s = current_s
    best_energy = current_energy
    
    n = 36
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Adaptive step size
        step_scale = 0.3 * (t0 - current_energy * 0.01)
        step_scale = max(0.05, min(1.0, step_scale))
        
        # Generate neighbor: perturb a random subset of parameters
        n_perturb = np.random.randint(1, 5)
        indices = np.random.choice(n, n_perturb, replace=False)
        neighbor = current.copy()
        for idx in indices:
            if idx % 3 == 2:  # angle
                neighbor[idx] += np.random.randn() * step_scale * 10
            else:  # position
                neighbor[idx] += np.random.randn() * step_scale
        
        neighbor_s = _outer_side_length_from_positions(
            np.column_stack([neighbor[0::3], neighbor[1::3]]), neighbor[2::3])
        neighbor_energy = _sa_objective(neighbor, neighbor_s)
        
        # Temperature schedule
        alpha = t_end / t0
        T = t0 * (alpha ** (it / max_iter))
        
        # Accept if better, or with probability exp(-delta/T)
        delta = neighbor_energy - current_energy
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current = neighbor
            current_s = neighbor_s
            current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best = current.copy()
                best_s = current_s
                best_energy = current_energy
    
    return best, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Honeycomb starting configuration (tighter packing)
    # Center hexagon surrounded by 6, then 5 more around the perimeter
    honeycomb_positions = np.array([
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
    honeycomb_angles = np.zeros(12)
    
    # Seed arrangement (valid fallback)
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
    
    best_params = None
    best_s = float('inf')
    
    # Multiple SA runs with different starting configurations
    starting_configs = []
    
    # 1. Honeycomb
    starting_configs.append((honeycomb_positions.copy(), honeycomb_angles.copy()))
    
    # 2. Seed
    starting_configs.append((seed_positions.copy(), seed_angles.copy()))
    
    # 3. Rotated honeycomb
    for rot in [15, 30, 45, 60]:
        ang = np.full(12, rot)
        starting_configs.append((honeycomb_positions.copy(), ang))
    
    # 4. Perturbed honeycomb
    for t in range(10):
        np.random.seed(t * 100 + 42)
        pos = honeycomb_positions + np.random.randn(12, 2) * 0.2
        ang = np.random.uniform(-20, 20, 12)
        starting_configs.append((pos, ang))
    
    for idx, (pos, ang) in enumerate(starting_configs):
        if time.time() > deadline - 30:
            break
        
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        s_init = _outer_side_length_from_positions(pos, ang)
        
        # Run SA
        if time.time() < deadline - 15:
            try:
                best_p, best_s_sa = _simulated_annealing(p, s_init, max_iter=30000, deadline=deadline - 10)
                if best_s_sa < best_s:
                    best_s = best_s_sa
                    best_params = best_p.copy()
            except Exception:
                pass
        
        # Quick Nelder-Mead polish on the SA result
        if time.time() < deadline - 5 and best_params is not None:
            try:
                from scipy.optimize import minimize
                def objective(params):
                    pos_p = np.column_stack([params[0::3], params[1::3]])
                    ang_p = params[2::3]
                    return _outer_side_length_from_positions(pos_p, ang_p)
                
                res = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_s:
                    best_s = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3*i] = seed_positions[i, 0]
            best_params[3*i+1] = seed_positions[i, 1]
            best_params[3*i+2] = seed_angles[i]
        best_s = _outer_side_length_from_positions(seed_positions, seed_angles)
    
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

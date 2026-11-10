# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all):
    """
    Given array of all inner hexagon vertices (shape (N,2)), compute minimum outer hex side s
    (centered at origin, angle 0) that contains all vertices.
    """
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3  # 30, 90, 150, 210, 270, 330 degrees
    m = np.column_stack([np.cos(angles), np.sin(angles)])  # shape (6,2)
    
    dots = vertices_all @ m.T  # shape (N, 6)
    max_dot = np.max(dots, axis=1)  # shape (N,)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _min_outer_side_smooth(vertices_all, beta=50.0):
    """
    Smooth approximation of _min_outer_side using log-sum-exp for gradient-based optimization.
    """
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])  # shape (6,2)
    
    dots = vertices_all @ m.T  # shape (N, 6)
    
    # Soft max over the 6 directions for each vertex
    max_dots = np.max(dots, axis=1, keepdims=True)
    soft_max_per_vertex = np.sum(np.exp((dots - max_dots) * beta) * dots, axis=1) / np.sum(np.exp((dots - max_dots) * beta), axis=1)
    
    # Global soft max over all vertices
    max_global = np.max(soft_max_per_vertex)
    global_soft_max = np.sum(np.exp((soft_max_per_vertex - max_global) * beta) * soft_max_per_vertex) / np.sum(np.exp((soft_max_per_vertex - max_global) * beta))
    
    if global_soft_max < 0:
        global_soft_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_soft_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
    Returns True if no overlap, False otherwise.
    centers: (12,2), angles_deg: (12,)
    """
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    # Precompute edge normals for each hexagon
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
            # SAT: check all 12 edge normals (6 from each hexagon)
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
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Inner ring: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # Outer ring: 5 more hexagons
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

def _objective_scipy(params, use_smooth=True, beta=50.0):
    """
    Objective for scipy optimizer: minimize outer hex side s, with penalty for overlaps.
    params: (36,) = [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    
    if use_smooth:
        s = _min_outer_side_smooth(all_verts, beta)
    else:
        s = _min_outer_side(all_verts)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _solve_lp_packing(centers, angles_deg):
    """
    Try to solve an LP for the optimal outer hexagon given fixed inner hex positions.
    This uses the fact that the minimum enclosing hexagon can be found via LP.
    """
    # For a regular hexagon with angle 0, the constraint is:
    # For each of 6 directions (30, 90, 150, 210, 270, 330 degrees):
    #   |v · m_i| <= s * sqrt(3)/2 for all vertices v
    # This is equivalent to: s >= (2/sqrt(3)) * max_i max_v |v · m_i|
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts)
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 610.0
    
    # Start with honeycomb arrangement
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
    
    # Simulated annealing phase 1: coarse
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
        
        # Adaptive step size
        if it % 500 == 0 and it > 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.2:
                step_size *= 0.8
            elif acc_rate > 0.8:
                step_size *= 1.2
            step_size = np.clip(step_size, 0.005, 1.0)
            accept_count = 0
            total_count = 0
    
    # Scipy optimization phase with smooth objective: use multiple starting points
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
            x0 = best_params + np.random.randn(36) * (0.1 + 0.1 * si)
        
        try:
            # First try with smooth objective
            result = minimize(
                lambda p: _objective_scipy(p, use_smooth=True, beta=30.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxls': 50},
                tol=1e-12
            )
            if result.fun < best_s and not np.any(np.isnan(result.fun)):
                c = result.x.reshape(12, 3)[:, :2]
                a = result.x.reshape(12, 3)[:, 2]
                if _check_nonoverlap(c, a):
                    s_check = _min_outer_side(_compute_all_vertices(c, a))
                    if s_check < best_s:
                        best_s = s_check
                        best_params = result.x.copy()
            
            # Then refine with exact objective
            result2 = minimize(
                lambda p: _objective_scipy(p, use_smooth=False),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 50},
                tol=1e-14
            )
            if result2.fun < best_s and not np.any(np.isnan(result2.fun)):
                c = result2.x.reshape(12, 3)[:, :2]
                a = result2.x.reshape(12, 3)[:, 2]
                if _check_nonoverlap(c, a):
                    s_check = _min_outer_side(_compute_all_vertices(c, a))
                    if s_check < best_s:
                        best_s = s_check
                        best_params = result2.x.copy()
        except Exception:
            pass
    
    # Final refinement with smaller steps
    for _ in range(30000):
        if time.time() > deadline - 25.0:
            break
        trial = best_params + np.random.randn(36) * 0.02
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Very fine polish
    for _ in range(10000):
        if time.time() > deadline - 5.0:
            break
        trial = best_params + np.random.randn(36) * 0.005
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

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its vertices."""
    normals = []
    center = np.mean(verts, axis=0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def min_gap_between_hexes(v1, v2):
    """
    Compute the separation gap between two hexagons using their 12 edge normals.
    Returns the maximum over all normals of the gap (negative means overlap).
    """
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    max_gap = -np.inf
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap = max(min2 - max1, min1 - max2)
        if gap > max_gap:
            max_gap = gap
    
    return max_gap


def check_separation(hex1, hex2, tol=1e-6):
    """
    Check if two hexagons are separated: on at least one of their 12 edge normals,
    the projected intervals are apart by more than tol.
    """
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    
    v1 = hexagon_vertices(x1, y1, 1.0, a1)
    v2 = hexagon_vertices(x2, y2, 1.0, a2)
    
    gap = min_gap_between_hexes(v1, v2)
    return gap > tol


def calculate_min_side(inner_hex_data, outer_angle_deg):
    """
    Calculate the minimum side length S of an outer regular hexagon centered at origin
    with given angle that contains all inner hexagons.
    """
    normal_angles = [outer_angle_deg + 30 + 60 * k for k in range(6)]
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in normal_angles])
    
    all_vertices = []
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    max_projs = np.max(all_vertices @ normals.T, axis=0)
    apothem = np.max(max_projs)
    side = 2.0 * apothem / math.sqrt(3)
    
    return side


def is_valid_configuration(inner_hex_data, outer_angle_deg, tol=1e-6):
    """Check if the configuration is valid (all hexagons separated)."""
    for i in range(11):
        for j in range(i + 1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol=tol):
                return False
    return True


def objective(params, outer_angle_deg, penalty_weight=100.0):
    """
    Objective function for optimization.
    params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10] (33 values)
    """
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = params[3 * i]
        inner_hex_data[i, 1] = params[3 * i + 1]
        inner_hex_data[i, 2] = params[3 * i + 2]
    
    side = calculate_min_side(inner_hex_data, outer_angle_deg)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol=1e-6):
                v1 = hexagon_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
                v2 = hexagon_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
                gap = min_gap_between_hexes(v1, v2)
                if gap < 1e-6:
                    penalty += (1e-6 - gap) ** 2
    
    return side + penalty_weight * penalty


def objective_2d(params, outer_angle_deg, penalty_weight=100.0):
    """
    Objective function for optimization with only positions (angles fixed at 0).
    params: [x0, y0, x1, y1, ..., x10, y10] (22 values)
    """
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = params[2 * i]
        inner_hex_data[i, 1] = params[2 * i + 1]
        inner_hex_data[i, 2] = 0.0
    
    side = calculate_min_side(inner_hex_data, outer_angle_deg)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol=1e-6):
                v1 = hexagon_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, 0.0)
                v2 = hexagon_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, 0.0)
                gap = min_gap_between_hexes(v1, v2)
                if gap < 1e-6:
                    penalty += (1e-6 - gap) ** 2
    
    return side + penalty_weight * penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 15.0
    
    h = math.sqrt(3) / 2
    
    # Multiple initial configurations
    initial_configs = []
    
    # Config 1: 4-3-4 honeycomb
    coords = []
    for i in range(4):
        coords.append([i * 3.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i * 3.0, 2 * h])
    for i in range(4):
        coords.append([i * 3.0, 4 * h])
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    initial_configs.append(np.array(coords))
    
    # Config 2: 3-5-3 grid
    coords = []
    for i in range(3):
        coords.append([i * 3.0, 0.0])
    for i in range(5):
        coords.append([1.5 + i * 3.0, 2 * h])
    for i in range(3):
        coords.append([i * 3.0, 4 * h])
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    initial_configs.append(np.array(coords))
    
    best_side = float('inf')
    best_outer_angle = 0.0
    best_inner = None
    
    for config_idx, coords in enumerate(initial_configs):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        inner_hex_data = np.zeros((11, 3))
        for i, c in enumerate(coords):
            inner_hex_data[i, 0] = c[0]
            inner_hex_data[i, 1] = c[1]
            inner_hex_data[i, 2] = 0.0
        
        # Find best outer angle
        for outer_angle_deg in np.arange(0, 30.01, 0.5):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(inner_hex_data, outer_angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle_deg
                best_inner = inner_hex_data.copy()
        
        # Refine outer angle
        if time.time() - start_time < deadline - safety_margin:
            for outer_angle_deg in np.arange(max(0, best_outer_angle - 1.0), min(30, best_outer_angle + 1.0) + 0.01, 0.05):
                if time.time() - start_time > deadline - safety_margin:
                    break
                side = calculate_min_side(inner_hex_data, outer_angle_deg)
                if side < best_side:
                    best_side = side
                    best_outer_angle = outer_angle_deg
                    best_inner = inner_hex_data.copy()
    
    if best_inner is None:
        best_inner = initial_configs[0].copy()
        best_inner = np.column_stack([best_inner, np.zeros(11)])
        best_side = calculate_min_side(best_inner, 0.0)
        best_outer_angle = 0.0
    
    # Optimize positions and angles (33 params)
    if time.time() - start_time < deadline - safety_margin:
        x0 = np.zeros(33)
        for i in range(11):
            x0[3 * i] = best_inner[i, 0]
            x0[3 * i + 1] = best_inner[i, 1]
            x0[3 * i + 2] = best_inner[i, 2]
        
        for method in ['Nelder-Mead', 'L-BFGS-B']:
            if time.time() - start_time > deadline - safety_margin:
                break
            
            for trial in range(4):
                if time.time() - start_time > deadline - safety_margin:
                    break
                
                if trial == 0:
                    x_init = x0.copy()
                else:
                    np.random.seed(trial * 1234 + 5678)
                    x_init = x0.copy()
                    x_init[0:22:2] += np.random.randn(11) * 0.05
                    x_init[1:22:2] += np.random.randn(11) * 0.05
                    x_init[2:33:3] += np.random.randn(11) * 5.0
                
                try:
                    if method == 'Nelder-Mead':
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                        )
                    else:
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                    
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    
                    test_inner = np.zeros((11, 3))
                    for i in range(11):
                        test_inner[i, 0] = result.x[3 * i]
                        test_inner[i, 1] = result.x[3 * i + 1]
                        test_inner[i, 2] = result.x[3 * i + 2]
                    
                    if is_valid_configuration(test_inner, best_outer_angle):
                        new_side = calculate_min_side(test_inner, best_outer_angle)
                        if new_side < best_side:
                            best_side = new_side
                            best_inner = test_inner
                            
                except Exception:
                    pass
    
    # Optimize positions only (22 params) with more restarts
    if time.time() - start_time < deadline - safety_margin:
        x0_2d = best_inner[:, :2].flatten()
        
        for trial in range(6):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            if trial == 0:
                x_init = x0_2d.copy()
            else:
                np.random.seed(trial * 4321 + 8765)
                x_init = x0_2d + np.random.randn(22) * 0.03
            
            try:
                result = minimize(
                    objective_2d,
                    x_init,
                    args=(best_outer_angle,),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                if time.time() - start_time > deadline - safety_margin:
                    break
                
                test_inner = np.zeros((11, 3))
                for i in range(11):
                    test_inner[i, 0] = result.x[2 * i]
                    test_inner[i, 1] = result.x[2 * i + 1]
                    test_inner[i, 2] = best_inner[i, 2]
                
                if is_valid_configuration(test_inner, best_outer_angle):
                    new_side = calculate_min_side(test_inner, best_outer_angle)
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = test_inner
                        
            except Exception:
                pass
    
    # Joint optimization of positions and outer angle
    if time.time() - start_time < deadline - safety_margin:
        def objective_joint(params):
            x_pos = params[:22]
            outer_angle = params[22]
            
            inner_hex_data = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data[i, 0] = x_pos[2 * i]
                inner_hex_data[i, 1] = x_pos[2 * i + 1]
                inner_hex_data[i, 2] = best_inner[i, 2]
            
            side = calculate_min_side(inner_hex_data, outer_angle)
            
            penalty = 0.0
            for i in range(11):
                for j in range(i + 1, 11):
                    if not check_separation(inner_hex_data[i], inner_hex_data[j], tol=1e-6):
                        v1 = hexagon_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
                        v2 = hexagon_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
                        gap = min_gap_between_hexes(v1, v2)
                        if gap < 1e-6:
                            penalty += (1e-6 - gap) ** 2
            
            return side + 100 * penalty
        
        x0_joint = np.zeros(23)
        x0_joint[:22] = best_inner[:, :2].flatten()
        x0_joint[22] = best_outer_angle
        
        try:
            result = minimize(
                objective_joint,
                x0_joint,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if time.time() - start_time < deadline - safety_margin:
                test_inner = np.zeros((11, 3))
                for i in range(11):
                    test_inner[i, 0] = result.x[2 * i]
                    test_inner[i, 1] = result.x[2 * i + 1]
                    test_inner[i, 2] = best_inner[i, 2]
                
                test_outer_angle = result.x[22]
                
                if is_valid_configuration(test_inner, test_outer_angle):
                    new_side = calculate_min_side(test_inner, test_outer_angle)
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = test_inner
                        best_outer_angle = test_outer_angle
                        
        except Exception:
            pass
    
    # Final local refinement with small steps
    if time.time() - start_time < deadline - safety_margin:
        current_best = best_side
        current_pos = best_inner.copy()
        current_angle = best_outer_angle
        
        step_sizes = [0.005, 0.001, 0.0001]
        for step in step_sizes:
            if time.time() - start_time > deadline - safety_margin:
                break
            improved = True
            while improved and time.time() - start_time < deadline - safety_margin:
                improved = False
                for i in range(11):
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            test_pos = current_pos.copy()
                            test_pos[i, 0] += dx
                            test_pos[i, 1] += dy
                            
                            if is_valid_configuration(test_pos, current_angle):
                                new_side = calculate_min_side(test_pos, current_angle)
                                if new_side < current_best - 1e-10:
                                    current_pos = test_pos
                                    current_best = new_side
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        if current_best < best_side:
            best_side = current_best
            best_inner = current_pos
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return best_inner, outer_hex_data, best_side


# EVOLVE-BLOCK-END

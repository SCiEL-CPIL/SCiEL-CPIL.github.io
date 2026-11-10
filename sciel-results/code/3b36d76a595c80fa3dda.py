import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _get_all_vertices(positions, angles):
    """
    positions: (11, 2)
    angles: (11,)
    Returns: (66, 2) array of all vertices.
    """
    verts = []
    for i in range(11):
        v = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts.append(v)
    return np.vstack(verts)

def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons (vertex arrays) are separated using all 12 edge normals."""
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    return False

def _is_valid_fast(positions, angles):
    """
    Check validity of packing.
    positions: (11, 2)
    angles: (11,)
    """
    verts_list = []
    for i in range(11):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    
    for i in range(11):
        for j in range(i+1, 11):
            if not _hex_separated(verts_list[i], verts_list[j]):
                return False
    return True

def _compute_side_length(positions, angles, outer_angle):
    """
    Compute the minimum side length of a regular hexagon with given angle
    that contains all inner hexagons.
    """
    all_verts = _get_all_vertices(positions, angles)
    
    a = np.radians(outer_angle)
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i) # Normals are at 30, 90, ...
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n
        max_proj = max(max_proj, np.max(np.abs(proj)))
    
    return 2.0 * max_proj / np.sqrt(3)

def objective(params):
    """
    Objective function for optimizer.
    params: 24 floats (11 x, 11 y, 1 global angle, 1 outer angle)
    """
    pos = params[:22].reshape(11, 2)
    global_angle = params[22]
    outer_angle = params[23]
    
    # We assume all inner hexagons have the same angle for simplicity, 
    # as rotating them individually rarely helps in dense hex packing.
    angles = np.full(11, global_angle)
    
    # Check validity
    if not _is_valid_fast(pos, angles):
        return 1e6
    
    # Compute side length
    side = _compute_side_length(pos, angles, outer_angle)
    
    return side

def _generate_initial_configs():
    """Generate multiple initial configurations for the 11 hexagons."""
    sqrt3 = np.sqrt(3)
    configs = []
    
    # Config 1: Standard hexagonal cluster (7 + 4 corners)
    pos1 = np.zeros((11, 2))
    pos1[0] = [0, 0]
    pos1[1] = [2, 0]
    pos1[2] = [1, sqrt3]
    pos1[3] = [-1, sqrt3]
    pos1[4] = [-2, 0]
    pos1[5] = [-1, -sqrt3]
    pos1[6] = [1, -sqrt3]
    pos1[7] = [3, sqrt3]
    pos1[8] = [-3, sqrt3]
    pos1[9] = [3, -sqrt3]
    pos1[10] = [-3, -sqrt3]
    configs.append(pos1)
    
    # Config 2: Slightly compressed
    pos2 = pos1.copy()
    pos2[7:11] *= 0.95
    configs.append(pos2)
    
    # Config 3: Different arrangement
    pos3 = np.zeros((11, 2))
    pos3[0] = [0, 0]
    pos3[1] = [2, 0]
    pos3[2] = [1, sqrt3]
    pos3[3] = [-1, sqrt3]
    pos3[4] = [-2, 0]
    pos3[5] = [-1, -sqrt3]
    pos3[6] = [1, -sqrt3]
    pos3[7] = [2.5, 1.5*sqrt3]
    pos3[8] = [-2.5, 1.5*sqrt3]
    pos3[9] = [2.5, -1.5*sqrt3]
    pos3[10] = [-2.5, -1.5*sqrt3]
    configs.append(pos3)
    
    # Config 4: Random perturbation of config 1
    pos4 = pos1 + np.random.normal(0, 0.2, size=(11, 2))
    configs.append(pos4)
    
    # Config 5: Another random perturbation
    pos5 = pos1 + np.random.normal(0, 0.3, size=(11, 2))
    configs.append(pos5)
    
    return configs

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_side = float('inf')
    best_pos = None
    best_global_angle = 0.0
    best_outer_angle = 0.0
    
    # Generate initial configurations
    configs = _generate_initial_configs()
    
    # Evaluate all initial configs
    for pos_init in configs:
        if time.time() > deadline:
            break
        
        angles_init = np.zeros(11)
        if _is_valid_fast(pos_init, angles_init):
            current_side = _compute_side_length(pos_init, angles_init, 0.0)
            if current_side < best_side:
                best_side = current_side
                best_pos = pos_init.copy()
                best_global_angle = 0.0
                best_outer_angle = 0.0
    
    # If no valid config found, use a sparse fallback
    if best_pos is None:
        pos_fallback = np.zeros((11, 2))
        for i in range(11):
            pos_fallback[i] = [i * 4, 0]
        best_pos = pos_fallback
        best_global_angle = 0.0
        best_outer_angle = 0.0
        best_side = _compute_side_length(best_pos, np.zeros(11), 0.0)
    
    # Parameters for optimization:
    # 11 x, 11 y, 1 global_angle, 1 outer_angle
    bounds = [(-6, 6)] * 22 + [(0, 360), (0, 360)]
    
    # Use Differential Evolution for global search with multiple restarts
    if time.time() < deadline - 120:
        try:
            # First DE run
            res = differential_evolution(
                objective, 
                bounds=bounds, 
                seed=42, 
                maxiter=50, 
                popsize=12,
                tol=1e-4,
                polish=False
            )
            if res.fun < best_side:
                best_side = res.fun
                params = res.x
                best_pos = params[:22].reshape(11, 2)
                best_global_angle = params[22]
                best_outer_angle = params[23]
        except Exception as e:
            pass
    
    # Second DE run with different seed if time permits
    if time.time() < deadline - 90:
        try:
            res = differential_evolution(
                objective, 
                bounds=bounds, 
                seed=123, 
                maxiter=40, 
                popsize=10,
                tol=1e-4,
                polish=False
            )
            if res.fun < best_side:
                best_side = res.fun
                params = res.x
                best_pos = params[:22].reshape(11, 2)
                best_global_angle = params[22]
                best_outer_angle = params[23]
        except Exception as e:
            pass
    
    # Local refinement using Nelder-Mead
    if time.time() < deadline - 60:
        try:
            initial_params = np.concatenate([best_pos.flatten(), [best_global_angle, best_outer_angle]])
            res_nm = minimize(
                objective, 
                initial_params, 
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_nm.fun < best_side:
                best_side = res_nm.fun
                params = res_nm.x
                best_pos = params[:22].reshape(11, 2)
                best_global_angle = params[22]
                best_outer_angle = params[23]
        except Exception as e:
            pass
    
    # Another local refinement if time permits
    if time.time() < deadline - 30:
        try:
            initial_params = np.concatenate([best_pos.flatten(), [best_global_angle, best_outer_angle]])
            res_nm2 = minimize(
                objective, 
                initial_params, 
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 300, 'maxfev': 500}
            )
            if res_nm2.fun < best_side:
                best_side = res_nm2.fun
                params = res_nm2.x
                best_pos = params[:22].reshape(11, 2)
                best_global_angle = params[22]
                best_outer_angle = params[23]
        except Exception as e:
            pass
            
    # Final check and adjustment
    if best_pos is None:
        best_pos = configs[0]
        best_global_angle = 0.0
        best_outer_angle = 0.0
        best_side = _compute_side_length(best_pos, np.zeros(11), 0.0)

    # Ensure validity
    angles = np.full(11, best_global_angle)
    if not _is_valid_fast(best_pos, angles):
        # If invalid, use the first valid config
        for pos_init in configs:
            if _is_valid_fast(pos_init, np.zeros(11)):
                best_pos = pos_init
                best_global_angle = 0.0
                best_outer_angle = 0.0
                best_side = _compute_side_length(best_pos, np.zeros(11), 0.0)
                break
        else:
            # Fallback to sparse
            best_pos = np.zeros((11, 2))
            for i in range(11):
                best_pos[i] = [i * 4, 0]
            best_global_angle = 0.0
            best_outer_angle = 0.0
            best_side = _compute_side_length(best_pos, np.zeros(11), 0.0)

    # Re-calculate side to be safe
    best_side = _compute_side_length(best_pos, angles, best_outer_angle)
    
    # Format output
    inner_hex_data = np.column_stack([best_pos, np.full(11, best_global_angle)])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, float(best_side)

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.radians(angle_deg)
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    n1 = _hex_edge_normals(hex1[2])
    n2 = _hex_edge_normals(hex2[2])
    
    for normals in [n1, n2]:
        for normal in normals:
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 + tol < min2 or max2 + tol < min1:
                return True
    
    return False

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _validate(inner_data, outer_data, s):
    """Validate the packing."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check inner hexagons
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        
        # Check unit edges
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        # Check inside outer
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    return True

def _min_enclosing_hex_side(centers, inner_angles, outer_angle=0.0):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, given angle)
    that contains all inner hexagons.
    """
    outer_normals = _hex_edge_normals(outer_angle)
    
    max_required_s = 0.0
    for i, center in enumerate(centers):
        inner_angle = inner_angles[i]
        inner_verts = _hex_vertices(0, 0, 1.0, inner_angle)
        
        for n in outer_normals:
            support = np.max(inner_verts @ n)
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _honeycomb_positions():
    """Generate 12 hexagon centers in a 3-4-3 honeycomb arrangement."""
    sqrt3 = np.sqrt(3)
    col_spacing = 1.5
    row_spacing = sqrt3
    
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * col_spacing
            if col % 2 == 0:
                y = row * row_spacing
            else:
                y = (row + 0.5) * row_spacing
            positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return np.array(positions)

def _separation_penalty(centers, angles, outer_angle):
    """
    Compute a smooth penalty for overlapping inner hexagons.
    Returns 0 if all are separated, positive value otherwise.
    """
    penalty = 0.0
    n1 = _hex_edge_normals(0.0)
    
    for i in range(12):
        v1 = _hex_vertices(centers[i,0], centers[i,1], 1.0, 0.0)
        for j in range(i+1, 12):
            v2 = _hex_vertices(centers[j,0], centers[j,1], 1.0, 0.0)
            
            min_gap = np.inf
            separated = False
            
            for normal in n1:
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
                gap = min(min2 - max1, min1 - max2)
                if gap < min_gap:
                    min_gap = gap
            
            if not separated and min_gap < 0:
                penalty += min_gap ** 2
    
    return penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    sqrt3 = np.sqrt(3)
    
    # 1. Initial Placement - multiple diverse starting configurations
    initial_configs = []
    
    # Config 1: Standard honeycomb 4x3
    config1 = _honeycomb_positions()
    initial_configs.append(config1)
    
    # Config 2: Hexagonal cluster (center + ring of 6 + 5 more)
    config2 = np.array([
        [0, 0, 0],
        [2.0, 0, 0],
        [1.0, sqrt3, 0],
        [-1.0, sqrt3, 0],
        [-2.0, 0, 0],
        [-1.0, -sqrt3, 0],
        [1.0, -sqrt3, 0],
        [3.0, sqrt3, 0],
        [3.0, -sqrt3, 0],
        [0, 2*sqrt3, 0],
        [-3.0, sqrt3, 0],
        [-3.0, -sqrt3, 0],
    ])
    # Center it
    cx = (config2[:,0].min() + config2[:,0].max()) / 2
    cy = (config2[:,1].min() + config2[:,1].max()) / 2
    config2[:, 0] -= cx
    config2[:, 1] -= cy
    initial_configs.append(config2)
    
    # Config 3: Compact 3x4 with slight perturbation
    config3 = _honeycomb_positions().copy()
    np.random.seed(42)
    config3[:, 0] += np.random.randn(12) * 0.1
    config3[:, 1] += np.random.randn(12) * 0.1
    initial_configs.append(config3)
    
    best_result = None
    best_value = 0.0
    
    for config_idx, initial_config in enumerate(initial_configs):
        if time.time() > deadline - 100:
            break
        
        inner_data = initial_config.copy()
        centers = inner_data[:, :2].copy()
        inner_angles = np.zeros(12)
        
        # Initial outer hexagon
        s = _min_enclosing_hex_side(centers, inner_angles, 0.0)
        s *= 1.001
        outer_data = np.array([0.0, 0.0, 0.0])
        
        current_best = (inner_data.copy(), outer_data.copy(), s)
        current_value = 1.0 / s
        
        # 2. Optimization Objective
        def objective(params):
            inner_xy = params[:24].reshape(12, 2)
            outer_angle = params[24]
            
            # Calculate required s
            s_req = _min_enclosing_hex_side(inner_xy, inner_angles, outer_angle)
            
            # Penalty for overlaps
            penalty = _separation_penalty(inner_xy, inner_angles, outer_angle)
            
            return s_req + 1000.0 * penalty
        
        # 3. Global Search
        bounds = [(-5, 5)] * 24 + [(-180, 180)]
        
        try:
            if time.time() < deadline - 150:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    seed=42 + config_idx,
                    maxiter=30,
                    tol=1e-4,
                    polish=False,
                    workers=-1
                )
                
                if de_result.fun < current_value * 1.5:
                    inner_xy_opt = de_result.x[:24].reshape(12, 2)
                    outer_angle_opt = de_result.x[24]
                    
                    s_opt = _min_enclosing_hex_side(inner_xy_opt, inner_angles, outer_angle_opt) * 1.0001
                    inner_opt = np.zeros((12, 3))
                    inner_opt[:, 0] = inner_xy_opt[:, 0]
                    inner_opt[:, 1] = inner_xy_opt[:, 1]
                    inner_opt[:, 2] = 0.0
                    outer_opt = np.array([0.0, 0.0, outer_angle_opt])
                    
                    if _validate(inner_opt, outer_opt, s_opt):
                        val = 1.0 / s_opt
                        if val > current_value:
                            current_value = val
                            current_best = (inner_opt, outer_opt, s_opt)
        except Exception:
            pass
        
        # 4. Local Polish with multiple restarts
        current_inner = current_best[0].copy()
        current_outer = current_best[1].copy()
        current_s = current_best[2]
        
        init_params = np.zeros(25)
        init_params[:24] = current_inner[:, :2].flatten()
        init_params[24] = current_outer[2]
        
        # Tolerance progression
        tolerances = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        
        for tol in tolerances:
            if time.time() > deadline - 30:
                break
            
            # Run Nelder-Mead
            try:
                res = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': tol,
                        'fatol': tol,
                        'adaptive': True
                    }
                )
                
                if res.success or res.fun < 100:
                    final_params = res.x
                    inner_xy_final = final_params[:24].reshape(12, 2)
                    outer_angle_final = final_params[24]
                    
                    s_final = _min_enclosing_hex_side(inner_xy_final, inner_angles, outer_angle_final)
                    s_final *= 1.00001
                    
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, outer_angle_final])
                    
                    if _validate(inner_final, outer_final, s_final):
                        val = 1.0 / s_final
                        if val > current_value:
                            current_value = val
                            current_best = (inner_final, outer_final, s_final)
                            init_params = final_params.copy()
                    else:
                        s_final *= 1.001
                        if _validate(inner_final, outer_final, s_final):
                            val = 1.0 / s_final
                            if val > current_value:
                                current_value = val
                                current_best = (inner_final, outer_final, s_final)
                                init_params = final_params.copy()
            except Exception:
                pass
        
        # Update global best
        if current_value > best_value:
            best_value = current_value
            best_result = current_best
    
    # Fallback if no valid result was found
    if best_result is None:
        inner_data = _honeycomb_positions()
        centers = inner_data[:, :2].copy()
        inner_angles = np.zeros(12)
        s = _min_enclosing_hex_side(centers, inner_angles, 0.0) * 1.01
        outer_data = np.array([0.0, 0.0, 0.0])
        best_result = (inner_data, outer_data, s)
    
    return best_result

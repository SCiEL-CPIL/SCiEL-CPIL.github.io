# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _sat_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons (arrays of 2D points) overlap using SAT.
    Returns True if they overlap (separation < eps)."""
    for poly in (v1, v2):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            n_len = np.linalg.norm(n)
            if n_len < 1e-12:
                continue
            n = n / n_len
            # Project both polygons onto normal
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Check separation
            if max1 < min2 - eps or max2 < min1 - eps:
                return False  # Separated on this axis
    return True  # No separating axis found -> overlap

def _point_in_hex(v, outer_verts, eps=1e-6):
    """Check if point v is inside or on the boundary of outer hexagon (convex).
    For each edge (p1, p2) in vertex order, cross(p2-p1, v-p1) must be >= -eps."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_v = v - p1
        cross = edge[0] * to_v[1] - edge[1] * to_v[0]
        if cross < -eps:
            return False
    return True

def _all_inside(v1, outer_verts, eps=1e-6):
    """Check if all vertices of v1 are inside outer hexagon."""
    for i in range(6):
        if not _point_in_hex(v1[i], outer_verts, eps):
            return False
    return True

def _compute_objective(params, n_inner=12):
    """Compute objective: s + penalty for violations.
    params layout: [s, outer_cx, outer_cy, outer_angle, then for each inner: cx, cy, angle]
    Total: 4 + 12*3 = 40 params
    """
    s = params[0]
    outer_cx = params[1]
    outer_cy = params[2]
    outer_angle = params[3]
    
    if s < 1.0:
        s = 1.0
    
    outer_verts = _hex_vertices(outer_cx, outer_cy, s, outer_angle)
    
    penalty = 0.0
    inner_verts_list = []
    
    for i in range(n_inner):
        base = 4 + i * 3
        cx = params[base]
        cy = params[base + 1]
        angle = params[base + 2]
        iv = _hex_vertices(cx, cy, 1.0, angle)
        inner_verts_list.append(iv)
        
        # Check if inside outer
        for j in range(6):
            if not _point_in_hex(iv[j], outer_verts, eps=1e-6):
                # Penalty: distance violation
                # Approximate by checking cross product magnitude
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    edge = p2 - p1
                    to_v = iv[j] - p1
                    cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                    if cross < 0:
                        penalty += 100.0 * (-cross)
    
    # Check pairwise overlap
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], eps=1e-6):
                penalty += 1000.0
    
    return s + penalty

def _compute_objective_fast(params, n_inner=12):
    """Faster objective using vectorized operations where possible."""
    s = params[0]
    outer_cx = params[1]
    outer_cy = params[2]
    outer_angle = params[3]
    
    if s < 1.0:
        s = 1.0
    
    outer_verts = _hex_vertices(outer_cx, outer_cy, s, outer_angle)
    
    penalty = 0.0
    inner_verts_list = []
    
    for i in range(n_inner):
        base = 4 + i * 3
        cx = params[base]
        cy = params[base + 1]
        angle = params[base + 2]
        iv = _hex_vertices(cx, cy, 1.0, angle)
        inner_verts_list.append(iv)
        
        # Check if inside outer - check all 6 vertices
        for j in range(6):
            v = iv[j]
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < 0:
                    penalty += 100.0 * (-cross)
    
    # Check pairwise overlap using SAT
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            v1 = inner_verts_list[i]
            v2 = inner_verts_list[j]
            overlap = False
            for poly in (v1, v2):
                for i2 in range(6):
                    p1 = poly[i2]
                    p2 = poly[(i2 + 1) % 6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_len = np.linalg.norm(n)
                    if n_len < 1e-12:
                        continue
                    n = n / n_len
                    proj1 = np.dot(v1, n)
                    proj2 = np.dot(v2, n)
                    if np.max(proj1) < np.min(proj2) - 1e-6 or np.max(proj2) < np.min(proj1) - 1e-6:
                        break
                else:
                    overlap = True
                    break
            if overlap:
                penalty += 1000.0
    
    return s + penalty

def _validate_result(inner_hex_data, outer_hex_data, s):
    """Validate the packing. Returns True if valid."""
    n = 12
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
    
    inner_verts_list = []
    for i in range(n):
        cx, cy, angle = inner_hex_data[i]
        iv = _hex_vertices(cx, cy, 1.0, angle)
        inner_verts_list.append(iv)
        
        # Check unit edge lengths
        for k in range(6):
            edge_len = np.linalg.norm(iv[(k+1)%6] - iv[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
        
        # Check all vertices inside outer
        for k in range(6):
            if not _point_in_hex(iv[k], outer_verts, eps=1e-6):
                return False
    
    # Check pairwise separation
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], eps=1e-6):
                return False
    
    return True

def _make_initial_config(config_type, s_init):
    """Create initial parameter vector for a given config type."""
    n = 12
    params = np.zeros(4 + n * 3)
    params[0] = s_init  # s
    params[1] = 0.0     # outer_cx
    params[2] = 0.0     # outer_cy
    params[3] = 0.0     # outer_angle
    
    if config_type == 0:
        # Honeycomb-like: center + 6 around + 6 more
        positions = [
            (0, 0),
            (2.5, 0), (-2.5, 0),
            (1.25, 2.165), (-1.25, 2.165),
            (1.25, -2.165), (-1.25, -2.165),
            (3.75, 2.165), (-3.75, 2.165),
            (3.75, -2.165), (-3.75, -2.165),
            (0, 4.33),
            (0, -4.33),
        ]
    elif config_type == 1:
        # Linear arrangement
        positions = []
        for i in range(12):
            x = (i - 5.5) * 2.5
            y = 0 if i % 2 == 0 else 2.165
            positions.append((x, y))
    elif config_type == 2:
        # Two rows of 6
        positions = []
        for i in range(6):
            positions.append((i * 2.5 - 6.25, 0))
            positions.append((i * 2.5 - 6.25 + 1.25, 2.165))
    elif config_type == 3:
        # Random-ish
        np.random.seed(42)
        positions = [(np.random.uniform(-6, 6), np.random.uniform(-6, 6)) for _ in range(12)]
    else:
        positions = [(0, 0)] + [(3.0 * np.cos(2*np.pi*i/11), 3.0 * np.sin(2*np.pi*i/11)) for i in range(11)]
    
    for i, (cx, cy) in enumerate(positions):
        base = 4 + i * 3
        params[base] = cx
        params[base + 1] = cy
        params[base + 2] = 0.0
    
    return params

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0
    
    n_inner = 12
    best_params = None
    best_value = -np.inf
    
    # Try multiple initial configurations
    config_types = [0, 1, 2, 3, 4]
    s_inits = [8.0, 7.0, 6.0, 5.5, 5.0]
    
    attempts = 0
    max_attempts = 15
    
    for ct in config_types:
        for si in s_inits:
            if time.time() > deadline - safety_margin:
                break
            if attempts >= max_attempts:
                break
            attempts += 1
            
            params = _make_initial_config(ct, si)
            
            # Use L-BFGS-B with bounds
            bounds = [(1.0, 20.0), (-10, 10), (-10, 10), (-180, 180)]
            for i in range(n_inner):
                bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
            
            try:
                result = minimize(
                    _compute_objective_fast,
                    params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_value:
                    # Refine with Nelder-Mead
                    if time.time() > deadline - safety_margin:
                        # Just use L-BFGS result
                        if result.fun < best_value:
                            best_params = result.x
                            best_value = result.fun
                    else:
                        result2 = minimize(
                            _compute_objective_fast,
                            result.x,
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                        if result2.fun < best_value:
                            best_params = result2.x
                            best_value = result2.fun
                        elif result.fun < best_value:
                            best_params = result.x
                            best_value = result.fun
            except Exception:
                continue
            
            if time.time() > deadline - safety_margin:
                break
    
    if best_params is None:
        # Fallback to simple valid configuration
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0
    
    # Extract result
    s = best_params[0]
    outer_hex_data = np.array([best_params[1], best_params[2], best_params[3]])
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        base = 4 + i * 3
        inner_hex_data[i] = [best_params[base], best_params[base+1], best_params[base+2]]
    
    # Validate
    if _validate_result(inner_hex_data, outer_hex_data, s):
        return inner_hex_data, outer_hex_data, s
    
    # If validation fails, try to fix by scaling up
    # Find the minimum s that works
    for trial_s in np.arange(s, 15.0, 0.1):
        outer_hex_data = np.array([best_params[1], best_params[2], best_params[3]])
        if _validate_result(inner_hex_data, outer_hex_data, trial_s):
            return inner_hex_data, outer_hex_data, trial_s
    
    # Last resort: return a known valid configuration
    inner_hex_data = np.array([
        [0, 0, 0],
        [-2.5, 0, 0], [2.5, 0, 0],
        [-1.25, 2.17, 0], [1.25, 2.17, 0],
        [-1.25, -2.17, 0], [1.25, -2.17, 0],
        [-3.75, 2.17, 0], [3.75, 2.17, 0],
        [-3.75, -2.17, 0], [3.75, -2.17, 0],
        [0, -4, 0],
    ])
    outer_hex_data = np.array([0, 0, 0])
    return inner_hex_data, outer_hex_data, 8.0

# EVOLVE-BLOCK-END

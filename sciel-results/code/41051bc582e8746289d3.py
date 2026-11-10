# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _min_s_for_all_vertices(all_inner_verts, outer_center, angle_deg):
    """Find minimum s such that all inner vertices fit in outer hexagon."""
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    max_proj = 0
    for verts in all_inner_verts:
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (np.sqrt(3) / 2)

def _compute_s(inner_data, outer_angle):
    """Compute the minimum s needed for given inner positions and outer angle."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return _min_s_for_all_vertices(all_verts, np.array([0, 0]), outer_angle)

def _objective(x):
    """
    Objective: minimize s given positions and angles.
    x = [cx1..cx12, cy1..cy12, angle1..angle12, outer_angle]
    Total: 12*2 + 12 + 1 = 37 variables
    """
    cx = x[0:12]
    cy = x[12:24]
    angles = x[24:36]
    outer_angle = x[36]
    
    inner_data = np.column_stack([cx, cy, angles])
    s = _compute_s(inner_data, outer_angle)
    
    # Check separation and add penalty if violated
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(cx[i], cy[i], 1.0, angles[i])
        inner_verts_list.append(v)
    
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                penalty += 1.0
    
    return s + 100.0 * penalty

def _make_lattice_config(scale=1.0):
    """Create a 3-4-3-2 hexagonal lattice configuration."""
    # Close-packed lattice vectors for unit hexagons
    e1 = np.array([1.5, math.sqrt(3)/2])
    e2 = np.array([0, math.sqrt(3)])
    
    positions = []
    # Row 0: 3 hexagons
    for j in range(3):
        pos = j * e1
        positions.append([pos[0], pos[1], 0.0])
    # Row 1: 4 hexagons
    for j in range(4):
        pos = 0.5 * e1 + j * e1 + e2
        positions.append([pos[0], pos[1], 0.0])
    # Row 2: 3 hexagons
    for j in range(3):
        pos = e1 + j * e1 + 2*e2
        positions.append([pos[0], pos[1], 0.0])
    # Row 3: 2 hexagons
    for j in range(2):
        pos = 1.5 * e1 + j * e1 + 3*e2
        positions.append([pos[0], pos[1], 0.0])
    
    inner = np.array(positions)
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    return inner

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is float.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # Start with lattice configuration
    inner = _make_lattice_config()
    
    # Find best outer angle for initial config
    best_s = float('inf')
    best_outer_angle = 0.0
    for angle_deg in np.linspace(0, 60, 121):
        s_cand = _compute_s(inner, angle_deg)
        if s_cand < best_s:
            best_s = s_cand
            best_outer_angle = angle_deg
    
    best_inner = inner.copy()
    best_s = s_cand if s_cand < best_s else best_s
    
    # Use differential evolution to optimize positions and angles
    # Variables: 12 cx, 12 cy, 12 angles, 1 outer_angle = 37 total
    bounds = [(-2, 2)] * 12 + [(-2, 2)] * 12 + [(-30, 30)] * 12 + [(-30, 30)]
    
    x0 = np.zeros(37)
    x0[0:12] = best_inner[:, 0]
    x0[12:24] = best_inner[:, 1]
    x0[24:36] = best_inner[:, 2]
    x0[36] = best_outer_angle
    
    best_result = (best_inner, np.array([0, 0, best_outer_angle]), best_s)
    
    if time.time() < deadline - 120:
        try:
            result = differential_evolution(
                _objective,
                bounds,
                x0=x0,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True
            )
            
            if result.fun < best_s:
                x_opt = result.x
                cx = x_opt[0:12]
                cy = x_opt[12:24]
                angles = x_opt[24:36]
                outer_angle = x_opt[36]
                
                inner_data = np.column_stack([cx, cy, angles])
                s_opt = _compute_s(inner_data, outer_angle)
                outer_data = np.array([0, 0, outer_angle])
                
                if _is_valid(inner_data, outer_data, s_opt):
                    best_result = (inner_data, outer_data, s_opt)
                    best_s = s_opt
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            current_inner, current_outer, current_s = best_result
            x_current = np.zeros(37)
            x_current[0:12] = current_inner[:, 0]
            x_current[12:24] = current_inner[:, 1]
            x_current[24:36] = current_inner[:, 2]
            x_current[36] = current_outer[2]
            
            res = minimize(
                _objective,
                x_current,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res.fun < best_s:
                x_opt = res.x
                cx = x_opt[0:12]
                cy = x_opt[12:24]
                angles = x_opt[24:36]
                outer_angle = x_opt[36]
                
                inner_data = np.column_stack([cx, cy, angles])
                s_opt = _compute_s(inner_data, outer_angle)
                outer_data = np.array([0, 0, outer_angle])
                
                if _is_valid(inner_data, outer_data, s_opt):
                    best_result = (inner_data, outer_data, s_opt)
                    best_s = s_opt
        except Exception:
            pass
    
    inner_final, outer_final, s_final = best_result
    
    if not _is_valid(inner_final, outer_final, s_final):
        # Fallback to lattice
        inner_final = best_inner
        outer_final = np.array([0, 0, best_outer_angle])
        s_final = best_s
    
    return inner_final, outer_final, s_final

# EVOLVE-BLOCK-END

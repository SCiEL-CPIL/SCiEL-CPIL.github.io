# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

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

def _compute_s_from_centers(centers, angles, outer_angle):
    """
    Compute minimum s for given centers and angles.
    centers: (12, 2)
    angles: (12,) in degrees
    outer_angle: scalar in degrees
    """
    all_verts = []
    for i in range(12):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    a = np.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)
    
    max_proj = 0
    for n in normals:
        projs = all_verts @ n
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (np.sqrt(3) / 2)

def _make_flower_config(scale=1.0):
    """Create a flower configuration: 1 center + 6 ring1 + 5 ring2."""
    centers = []
    centers.append([0.0, 0.0])
    
    r1 = np.sqrt(3) * scale
    for i in range(6):
        angle = 60 * i
        x = r1 * np.cos(np.radians(angle))
        y = r1 * np.sin(np.radians(angle))
        centers.append([x, y])
    
    r2 = 3.0 * scale
    angles_r2 = [30, 90, 150, 210, 270]
    for ang_deg in angles_r2:
        x = r2 * np.cos(np.radians(ang_deg))
        y = r2 * np.sin(np.radians(ang_deg))
        centers.append([x, y])
    
    centers = np.array(centers)
    com = np.mean(centers, axis=0)
    centers -= com
    
    angles = np.full(12, 30.0)
    return centers, angles

def _make_lattice_cluster():
    """Create a hexagonal lattice cluster of 12 points."""
    u1 = np.array([1.5, np.sqrt(3) / 2.0])
    u2 = np.array([-1.5, np.sqrt(3) / 2.0])
    
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            c = i * u1 + j * u2
            points.append(c)
    points = np.array(points)
    points -= points.mean(axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    centers = points[sorted_idx[:12]]
    
    com = np.mean(centers, axis=0)
    centers -= com
    
    angles = np.full(12, 30.0)
    return centers, angles

def _objective(x, centers_fixed=None):
    """
    Objective: minimize s given positions and angles.
    x = [cx1..cx12, cy1..cy12, angle1..angle12, outer_angle]
    Total: 37 variables
    """
    cx = x[0:12]
    cy = x[12:24]
    angles = x[24:36]
    outer_angle = x[36]
    
    centers = np.column_stack([cx, cy])
    s = _compute_s_from_centers(centers, angles, outer_angle)
    
    # Check separation and add penalty
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

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is float.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # Try multiple initial configurations
    configs = []
    
    # Flower configuration
    centers_flower, angles_flower = _make_flower_config(scale=1.0)
    configs.append((centers_flower, angles_flower))
    
    # Lattice cluster
    centers_lattice, angles_lattice = _make_lattice_cluster()
    configs.append((centers_lattice, angles_lattice))
    
    # Find best initial configuration
    best_initial = None
    best_initial_s = float('inf')
    
    for centers, angles in configs:
        best_angle = 0.0
        best_s = float('inf')
        for angle_deg in np.linspace(0, 60, 121):
            s_cand = _compute_s_from_centers(centers, angles, angle_deg)
            if s_cand < best_s:
                best_s = s_cand
                best_angle = angle_deg
        
        if best_s < best_initial_s:
            best_initial_s = best_s
            best_initial = (centers.copy(), angles.copy(), best_angle)
    
    centers_init, angles_init, outer_angle_init = best_initial
    best_inner = np.column_stack([centers_init, angles_init])
    best_outer = np.array([0.0, 0.0, outer_angle_init])
    best_s = best_initial_s
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 60:
        try:
            x_current = np.zeros(37)
            x_current[0:12] = centers_init[:, 0]
            x_current[12:24] = centers_init[:, 1]
            x_current[24:36] = angles_init
            x_current[36] = outer_angle_init
            
            res = minimize(
                _objective,
                x_current,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if res.fun < best_s + 1e-6:
                x_opt = res.x
                cx = x_opt[0:12]
                cy = x_opt[12:24]
                angles = x_opt[24:36]
                outer_angle = x_opt[36]
                
                centers_opt = np.column_stack([cx, cy])
                s_opt = _compute_s_from_centers(centers_opt, angles, outer_angle)
                inner_opt = np.column_stack([cx, cy, angles])
                outer_opt = np.array([0.0, 0.0, outer_angle])
                
                if _is_valid(inner_opt, outer_opt, s_opt):
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt
        except Exception:
            pass
    
    # Second round of refinement with tighter tolerance
    if time.time() < deadline - 30:
        try:
            x_current = np.zeros(37)
            x_current[0:12] = best_inner[:, 0]
            x_current[12:24] = best_inner[:, 1]
            x_current[24:36] = best_inner[:, 2]
            x_current[36] = best_outer[2]
            
            res = minimize(
                _objective,
                x_current,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            
            if res.fun < best_s + 1e-6:
                x_opt = res.x
                cx = x_opt[0:12]
                cy = x_opt[12:24]
                angles = x_opt[24:36]
                outer_angle = x_opt[36]
                
                centers_opt = np.column_stack([cx, cy])
                s_opt = _compute_s_from_centers(centers_opt, angles, outer_angle)
                inner_opt = np.column_stack([cx, cy, angles])
                outer_opt = np.array([0.0, 0.0, outer_angle])
                
                if _is_valid(inner_opt, outer_opt, s_opt):
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt
        except Exception:
            pass
    
    inner_final, outer_final, s_final = best_inner, best_outer, best_s
    
    # Final validation: if invalid, use fallback
    if not _is_valid(inner_final, outer_final, s_final):
        # Use the initial configuration
        s_lattice = _compute_s_from_centers(centers_init, angles_init, outer_angle_init)
        outer_lattice = np.array([0.0, 0.0, outer_angle_init])
        if _is_valid(best_inner, outer_lattice, s_lattice):
            inner_final = best_inner
            outer_final = outer_lattice
            s_final = s_lattice
        else:
            # Absolute fallback: spread out more
            inner_abs = np.zeros((12, 3))
            for i in range(12):
                row = i // 4
                col = i % 4
                inner_abs[i, 0] = (col - 1.5) * 4.0
                inner_abs[i, 1] = (row - 1.5) * 4.0 * math.sqrt(3) / 2
                inner_abs[i, 2] = 30.0
            s_abs = _compute_s_from_centers(inner_abs[:, :2], inner_abs[:, 2], 30.0)
            outer_abs = np.array([0.0, 0.0, 30.0])
            inner_final = inner_abs
            outer_final = outer_abs
            s_final = s_abs
    
    return inner_final, outer_final, s_final

# EVOLVE-BLOCK-END

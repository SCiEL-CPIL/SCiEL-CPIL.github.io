<<<<<<< SEARCH
"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses differential evolution with a vectorized constraint penalty.
"""
N = 12

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_half_width(cx, cy, side, angle_deg, dx, dy):
    """
    For a regular hexagon centered at (cx,cy) with side s and rotation angle,
    return the signed distance from center to the supporting line in direction (dx,dy),
    i.e., the maximum of dot((v - center), d) over vertices v.
    This is the support function in direction d.
    """
    # For a regular hexagon, the support function h(d) = s * max over 6 edge normals
    # Simpler: compute dot products with all 6 vertices
    a = np.radians(angle_deg)
    d = np.array([dx, dy])
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        return 0.0
    d = d / dn
    # 6 vertices
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _containing_hex_side_from_centers(centers, outer_angle_deg):
    """
    Given N center points (each hex has circumradius 1), find the minimum side
    of a regular hexagon with given rotation angle that contains all hexagons.
    
    A hexagon of side S contains a unit hexagon centered at p if:
      for all 6 edges of the outer hex, the signed distance from p to the edge
      is >= 1 (the inradius of the inner hex in that direction).
    
    Actually, the condition is: for each edge direction of the outer hex,
      dot(p - center_outer, n_i) + 1 <= S (where n_i is outward normal)
    Wait, let me think more carefully.
    
    Outer hex: center O, side S, rotation alpha.
    Its 6 edges have outward unit normals n_i = (cos(alpha + 60i), sin(alpha + 60i)).
    The edge i is at distance S from center O (apothem = S * sqrt(3)/2 ... no).
    
    For a regular hexagon with side S:
    - circumradius = S
    - inradius (apothem) = S * sqrt(3) / 2
    
    The supporting line in direction n_i is at distance apothem from center.
    So a point p is inside if: dot(p - O, n_i) <= apothem for all i.
    
    But we need the entire unit hexagon centered at p to be inside.
    The unit hexagon extends up to distance 1 from p in any direction (circumradius=1).
    Actually, the support function of a unit hexagon in direction n is h(n) = cos(30°) = sqrt(3)/2
    if n aligns with an edge normal of the inner hex... but the inner hex can rotate.
    
    To be safe (and since we're optimizing positions/angles), let's use:
    The condition for inner hex (center p, rotation beta, side 1) to be inside
    outer hex (center O, rotation alpha, side S) is:
      For each of the 6 outer edge directions n_i:
        max over inner hex vertices v of dot(v - O, n_i) <= apothem_outer
    
    This is complex. Let me use a simpler sufficient condition:
      For each outer edge direction n_i:
        dot(p - O, n_i) + 1 <= S  (using circumradius=1 as max extent)
    
    This is conservative but valid. The actual minimum extent of a unit hex in
    any direction is between sqrt(3)/2 and 1.
    
    Let me just compute the required S directly:
      S >= max_i [ dot(p - O, n_i) + h_inner(n_i) ]
    where h_inner(n_i) is the support function of the inner hex in direction n_i.
    
    For simplicity and correctness, I'll compute the support function properly.
    """
    outer_angle = np.radians(outer_angle_deg)
    # 6 outward normals of outer hex
    n_out = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    # For each center and its angle, compute max support in each outer normal direction
    # Support function of unit hex in direction d: 
    #   If the hex has vertices at angles beta + 60k, then
    #   h(d) = max_k dot(v_k, d) = max_k cos(beta + 60k - theta_d)
    #   where theta_d = atan2(d_y, d_x)
    
    max_ext = 0.0
    for k in range(N):
        cx, cy, ang = centers[k]
        a = np.radians(ang)
        # Support in each of the 6 outer normal directions
        for ni in range(6):
            dx = n_out[ni, 0]
            dy = n_out[ni, 1]
            theta_d = np.arctan2(dy, dx)
            # Support function of unit hex: max over 6 vertex directions
            # Vertex k is at angle a + 60*k, unit vector (cos, sin)
            # dot = cos(a + 60k - theta_d)
            angles_v = a + np.radians(60 * np.arange(6))
            dots = np.cos(angles_v - theta_d)
            h = np.max(dots)  # support of unit hex in direction d
            # The center is at distance from O
            dot_center = (cx - 0.0) * dx + (cy - 0.0) * dy  # O = (0,0)
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    # apothem = S * sqrt(3)/2, so S = max_ext / (sqrt(3)/2) = 2*max_ext/sqrt(3)
    # Wait: the condition is dot(p-O, n_i) + h_inner <= apothem = S*sqrt(3)/2
    # So S >= 2*(dot + h) / sqrt(3)
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _objective(params, outer_angle_deg):
    """
    params: array of shape (N*3,) = [x0,y0,a0, x1,y1,a1, ...]
    Returns the required outer side length (lower is better).
    """
    centers = np.zeros((N, 3))
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        centers[i, 2] = params[3*i+2]
    
    S = _containing_hex_side_from_centers(centers, outer_angle_deg)
    return S

def _pairwise_overlap_penalty(params):
    """Penalty if any two hexagons overlap. Use center distance check."""
    centers = np.zeros((N, 3))
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        centers[i, 2] = params[3*i+2]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = centers[i]
            dj = centers[j]
            # Two unit hexagons don't overlap if center distance >= 2
            # (this is sufficient when aligned, but not always sufficient with rotation)
            # For safety, use distance >= 2
            dist = np.sqrt((di[0]-dj[0])**2 + (di[1]-dj[1])**2)
            if dist < 2.0:
                penalty += (2.0 - dist)
    return penalty

def _full_objective(params, outer_angle_deg):
    """Combined objective: outer side + penalty for overlaps."""
    S = _objective(params, outer_angle_deg)
    overlap_pen = _pairwise_overlap_penalty(params)
    return S + 100.0 * overlap_pen

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Try different outer hex rotations
    best_S = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    for outer_angle in [0.0, 30.0]:
        if time.time() > deadline - 30.0:
            break
        
        # Define bounds for differential evolution
        # Positions: within a reasonable range, say [-6, 6] x [-6, 6]
        # Angles: [0, 60] degrees (periodic)
        bounds = []
        for i in range(N):
            bounds.append((-6.0, 6.0))  # x
            bounds.append((-6.0, 6.0))  # y
            bounds.append((0.0, 60.0))  # angle
        
        # Initial population: use hex lattice
        # Hex lattice positions for 12 hexagons
        # 3 rows x 4 cols
        hex_positions = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                hex_positions.append((x, y, 0.0))
        
        # Center it
        cx_mean = np.mean([p[0] for p in hex_positions])
        cy_mean = np.mean([p[1] for p in hex_positions])
        
        init_params = []
        for (x, y, a) in hex_positions:
            init_params.extend([x - cx_mean, y - cy_mean, a])
        init_params = np.array(init_params)
        
        # Run differential evolution
        try:
            result = differential_evolution(
                _full_objective,
                bounds,
                args=(outer_angle,),
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=15,
                polish=True,
                init='latinhypercube'
            )
            opt_params = result.x
            S_val = _objective(opt_params, outer_angle)
            overlap_pen = _pairwise_overlap_penalty(opt_params)
            
            if overlap_pen < 1e-6 and S_val < best_S:
                best_S = S_val
                best_params = opt_params.copy()
                best_outer_angle = outer_angle
        except Exception:
            continue
        
        # Try to polish with L-BFGS-B
        if time.time() < deadline - 10.0:
            try:
                result2 = minimize(
                    _full_objective,
                    best_params,
                    args=(best_outer_angle,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                opt2 = result2.x
                S2 = _objective(opt2, best_outer_angle)
                op2 = _pairwise_overlap_penalty(opt2)
                if op2 < 1e-6 and S2 < best_S:
                    best_S = S2
                    best_params = opt2.copy()
            except Exception:
                pass
    
    # If no good result, use fallback
    if best_params is None:
        # Safe fallback: hex lattice in a large container
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 6.0, 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation: check containment and non-overlap
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(0.0, 0.0, best_S, best_outer_angle)
    
    # Check containment: all inner vertices inside outer
    def point_in_hex(pt, outer_verts):
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i+1) % 6]
            edge = v2 - v1
            rel = pt - v1
            cross = edge[0]*rel[1] - edge[1]*rel[0]
            if cross < -1e-10:
                return False
        return True
    
    valid = True
    for iv in inner_verts_list:
        for pt in iv:
            if not point_in_hex(pt, outer_verts):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Increase S slightly
        best_S *= 1.05
    
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
=======
"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses Nelder-Mead with SAT-based overlap and containment penalties.
"""
N = 12

import numpy as np
import time
from scipy.optimize import minimize
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _point_in_polygon(pt, verts):
    """Check if point is inside polygon using winding number."""
    n = len(verts)
    inside = False
    x, y = pt
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def _sat_overlap_depth(verts1, verts2):
    """
    Compute the minimum translation distance to separate two convex polygons.
    Returns 0 if they don't overlap, positive value = overlap depth.
    """
    n1 = len(verts1)
    n2 = len(verts2)
    min_depth = float('inf')
    
    all_axes = []
    for i in range(n1):
        v1 = verts1[i]
        v2 = verts1[(i + 1) % n1]
        edge = v2 - v1
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm > 1e-12:
            all_axes.append(axis / norm)
    for i in range(n2):
        v1 = verts2[i]
        v2 = verts2[(i + 1) % n2]
        edge = v2 - v1
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm > 1e-12:
            all_axes.append(axis / norm)
    
    for axis in all_axes:
        proj1 = verts1 @ axis
        proj2 = verts2 @ axis
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        min_depth = min(min_depth, overlap)
    
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """
    Compute how much inner_verts violate containment in outer_verts.
    Returns 0 if fully contained, positive value = violation.
    """
    max_violation = 0.0
    for pt in inner_verts:
        if not _point_in_polygon(pt, outer_verts):
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % 6]
                edge = v2 - v1
                norm = np.linalg.norm(edge)
                if norm < 1e-12:
                    continue
                edge_unit = edge / norm
                center = np.mean(outer_verts, axis=0)
                edge_to_center = center - v1
                cross = edge_unit[0] * edge_to_center[1] - edge_unit[1] * edge_to_center[0]
                if cross > 0:
                    normal = np.array([-edge_unit[1], edge_unit[0]])
                else:
                    normal = np.array([edge_unit[1], -edge_unit[0]])
                
                dist = np.dot(pt - v1, normal)
                if dist < 0:
                    max_violation = max(max_violation, -dist)
    return max_violation

def _total_penalty(params, outer_side, outer_angle_deg, outer_center=(0.0, 0.0)):
    """
    Compute total penalty: overlap between inner hexagons + containment violation.
    """
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        verts = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(verts)
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle_deg)
    
    containment_pen = 0.0
    for iv in inner_verts_list:
        cv = _containment_violation(iv, outer_verts)
        containment_pen = max(containment_pen, cv)
    
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            overlap_pen = max(overlap_pen, depth)
    
    return 10.0 * containment_pen + 5.0 * overlap_pen

def _try_pack(outer_side, outer_angle_deg, initial_params, max_time=5.0):
    """Try to find a valid packing with the given outer hexagon."""
    start = time.time()
    
    try:
        result = minimize(
            _total_penalty,
            initial_params,
            args=(outer_side, outer_angle_deg),
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        opt_params = result.x
        pen = _total_penalty(opt_params, outer_side, outer_angle_deg)
        
        if pen < 1e-6:
            return True, opt_params
    except Exception:
        pass
    
    return False, initial_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    outer_angles = [0.0, 30.0]
    
    def generate_hex_lattice():
        positions = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y, 0.0))
        
        cx_mean = np.mean([p[0] for p in positions])
        cy_mean = np.mean([p[1] for p in positions])
        
        params = []
        for (x, y, a) in positions:
            params.extend([x - cx_mean, y - cy_mean, a])
        return np.array(params)
    
    def generate_circular():
        params = []
        for i in range(12):
            angle = 2 * np.pi * i / 12
            r = 2.5
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            params.extend([x, y, 0.0])
        return np.array(params)
    
    def generate_grid():
        params = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 - 3.0
                y = r * np.sqrt(3) - np.sqrt(3)
                params.extend([x, y, 0.0])
        return np.array(params)
    
    initial_configs = [generate_hex_lattice(), generate_circular(), generate_grid()]
    
    S_low = 3.0
    S_high = 8.0
    
    best_S = S_high
    best_params = None
    best_outer_angle = 0.0
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 30.0:
            break
        
        for init_config in initial_configs:
            if time.time() > deadline - 20.0:
                break
            
            success, opt_params = _try_pack(S_high, outer_angle, init_config, max_time=5.0)
            
            if success:
                S_lo = S_low
                S_hi = S_high
                best_params_found = opt_params.copy()
                
                for _ in range(10):
                    if time.time() > deadline - 15.0:
                        break
                    
                    S_mid = (S_lo + S_hi) / 2.0
                    success_mid, params_mid = _try_pack(S_mid, outer_angle, best_params_found, max_time=3.0)
                    
                    if success_mid:
                        S_hi = S_mid
                        best_params_found = params_mid.copy()
                        if S_mid < best_S:
                            best_S = S_mid
                            best_params = params_mid.copy()
                            best_outer_angle = outer_angle
                    else:
                        S_lo = S_mid
                
                break
    
    if best_params is None:
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 6.0, 0.0
    
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2])
        inner.append([x, y, ang])
    
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(0.0, 0.0, best_S, best_outer_angle)
    
    valid = True
    for iv in inner_verts_list:
        for pt in iv:
            if not _point_in_polygon(pt, outer_verts):
                valid = False
                break
        if not valid:
            break
    
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
                if depth > 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        best_S *= 1.02
        if time.time() < deadline - 5.0:
            success, new_params = _try_pack(best_S, best_outer_angle, best_params, max_time=3.0)
            if success:
                best_params = new_params
                inner = []
                for i in range(N):
                    x = float(best_params[3*i])
                    y = float(best_params[3*i+1])
                    ang = float(best_params[3*i+2])
                    inner.append([x, y, ang])
    
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
>>>>>>> REPLACE

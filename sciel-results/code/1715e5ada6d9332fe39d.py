"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = math.radians(angle_deg)
    verts = []
    for k in range(6):
        phi = theta + math.pi * k / 3.0
        x = cx + side * math.cos(phi)
        y = cy + side * math.sin(phi)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside (or on boundary of) the hexagon."""
    # Transform to local coords
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = px - cx
    dy = py - cy
    # Rotate by -theta
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    
    # In local coords, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # with distance = side
    # The hexagon in local coords: |x| <= side, |y| <= side*sqrt(3)/2, and 
    # for the top/bottom parts: |y| <= sqrt(3)*(side - |x|) for |x| <= side
    # Actually let me use the standard form:
    # A regular hexagon with "flat top" (vertices at 0, 60, 120, 180, 240, 300 degrees)
    # has equations:
    # |x| <= side
    # |y| <= sqrt(3) * side
    # |x| + |y|/sqrt(3) <= side
    # Wait, let me reconsider. With vertices at angle 0, 60, 120, 180, 240, 300:
    # Vertex at 0: (side, 0)
    # Vertex at 60: (side/2, side*sqrt(3)/2)
    # The edges connect these. The hexagon is the intersection of 6 half-planes.
    # For edge between vertex at 0 and vertex at 60:
    # Midpoint: (3*side/4, side*sqrt(3)/4)
    # Normal pointing inward: perpendicular to edge, pointing toward center
    # Edge direction: (side/2 - side, side*sqrt(3)/2 - 0) = (-side/2, side*sqrt(3)/2)
    # Normal (inward): rotate edge by -90: (side*sqrt(3)/2, side/2) normalized -> (sqrt(3)/2, 1/2)
    # So: sqrt(3)/2 * x + 1/2 * y <= sqrt(3)/2 * side + 1/2 * 0 = sqrt(3)*side/2
    # Hmm, this is getting complex. Let me just check distance to each edge.
    
    # Simpler: check if point is inside by checking all 6 edge constraints
    # Edge k connects vertex k and vertex k+1
    # For edge from vertex at angle a_k to a_{k+1}, the inward normal is at angle (a_k + a_{k+1})/2
    # The distance from center to edge = side * cos(30°) = side * sqrt(3)/2 = apothem
    
    apothem = side * math.sqrt(3) / 2.0
    
    for k in range(6):
        phi_edge = theta + math.pi * (k + 0.5) / 3.0  # direction of outward normal
        # Distance from center to this edge along normal direction = apothem
        # Point is inside if its projection onto this normal <= apothem
        proj = lx * math.cos(phi_edge) + ly * math.sin(phi_edge)
        if proj > apothem + 1e-9:
            return False
    return True

def _polygons_disjoint(hex1, hex2, eps=1e-9):
    """
    Check if two hexagons (each as list of 6 vertices) are disjoint (no interior overlap).
    They may touch at edges/vertices.
    Uses: if no vertex of one is inside the other AND no edge intersection, they're disjoint.
    """
    # Check if any vertex of hex1 is inside hex2
    for v in hex1:
        if _point_in_hex_simple(v[0], v[1], hex2):
            return False
    # Check if any vertex of hex2 is inside hex1
    for v in hex2:
        if _point_in_hex_simple(v[0], v[1], hex1):
            return False
    # Check edge intersections
    for i in range(6):
        v1 = hex1[i]
        v2 = hex1[(i + 1) % 6]
        for j in range(6):
            v3 = hex2[j]
            v4 = hex2[(j + 1) % 6]
            if _segments_intersect(v1, v2, v3, v4, eps):
                # They might just be touching - need to check if it's proper intersection
                # If they intersect at interior points, they overlap
                # For touching at a single point, that's OK
                # Let's use a more careful check
                pass
    return True

def _point_in_hex_simple(px, py, hex_verts):
    """Ray casting point in polygon test for hexagon."""
    inside = False
    n = len(hex_verts)
    j = n - 1
    for i in range(n):
        xi, yi = hex_verts[i]
        xj, yj = hex_verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4, eps=1e-9):
    """Check if segment p1-p2 intersects segment p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    return False

def _outer_side_length(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum side length of a regular hexagon (centered at outer_center, rotated by outer_angle)
    that contains all inner hexagons.
    
    inner_hexes: list of (cx, cy, angle_deg) for unit hexagons
    """
    ox, oy = outer_center
    oa = outer_angle
    
    # For a regular hexagon with side s, centered at (ox,oy), rotated by oa:
    # A point (px,py) is inside if for all 6 edge normals n_k:
    #   dot((px-ox, py-oy), n_k) <= s * apothem_ratio
    # where apothem = s * sqrt(3)/2, so the constraint is:
    #   dot((px-ox, py-oy), n_k) / apothem <= s
    #   dot((px-ox, py-oy), n_k) * 2/(sqrt(3)*s) <= s
    #   s^2 * 2/(sqrt(3)) >= dot((px-ox, py-oy), n_k)
    # Wait, let me redo this.
    # 
    # Regular hexagon with side s, center (ox,oy), rotation theta:
    # Vertices at distance s from center, at angles theta + k*60°
    # Apothem = s * sqrt(3)/2
    # The 6 edge outward normals are at angles theta + (k+0.5)*60°
    # A point (px,py) is inside iff for all k:
    #   dot((px-ox, py-oy), n_k) <= apothem = s * sqrt(3)/2
    # where n_k = (cos(theta + (k+0.5)*pi/3), sin(theta + (k+0.5)*pi/3))
    #
    # So: s >= dot((px-ox, py-oy), n_k) * 2/sqrt(3) for all k
    # The minimum s that works for a single point is:
    #   s_point = max_k [ dot((px-ox, py-oy), n_k) ] * 2/sqrt(3)
    # But we also need s >= 0, and the dot product can be negative (point on opposite side).
    # Actually, since the hexagon is symmetric, we need to consider absolute values? No.
    # The constraint dot(p - o, n_k) <= apothem is one-sided. If the point is on the "inside"
    # relative to edge k, the dot product is less than apothem.
    # But we also need the point to not be outside on the other side... 
    # Actually for a convex polygon, being inside all half-planes is sufficient.
    # Each edge defines a half-plane: dot(p - o, n_k) <= apothem
    # So for all k: dot(p - o, n_k) <= apothem
    # This means: apothem >= max_k dot(p - o, n_k)
    # So: s * sqrt(3)/2 >= max_k dot(p - o, n_k)
    # s >= max_k dot(p - o, n_k) * 2/sqrt(3)
    # But we also need s > 0.
    
    theta = math.radians(oa)
    apothem_ratio = math.sqrt(3) / 2.0  # apothem / s
    
    max_required_s = 0.0
    
    for (icx, icy, iangle) in inner_hexes:
        # Get vertices of this inner hexagon
        verts = _hex_vertices(icx, icy, 1.0, iangle)
        for (vx, vy) in verts:
            dx = vx - ox
            dy = vy - oy
            max_dot = 0.0
            for k in range(6):
                phi = theta + math.pi * (k + 0.5) / 3.0
                dot_val = dx * math.cos(phi) + dy * math.sin(phi)
                if dot_val > max_dot:
                    max_dot = dot_val
            # s >= max_dot / apothem_ratio
            required_s = max_dot / apothem_ratio
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _all_pairs_disjoint(inner_hexes, eps=1e-7):
    """Check if all pairs of inner hexagons are disjoint."""
    n = len(inner_hexes)
    hex_verts = []
    for (cx, cy, ang) in inner_hexes:
        hex_verts.append(_hex_vertices(cx, cy, 1.0, ang))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check vertex containment
            for v in hex_verts[i]:
                if _point_in_hex_simple(v[0], v[1], hex_verts[j]):
                    return False
            for v in hex_verts[j]:
                if _point_in_hex_simple(v[0], v[1], hex_verts[i]):
                    return False
            # Check edge crossings (proper intersections only)
            for a in range(6):
                p1 = hex_verts[i][a]
                p2 = hex_verts[i][(a + 1) % 6]
                for b in range(6):
                    p3 = hex_verts[j][b]
                    p4 = hex_verts[j][(b + 1) % 6]
                    if _proper_segment_intersect(p1, p2, p3, p4):
                        return False
    return True

def _proper_segment_intersect(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 properly intersect (cross in their interiors)."""
    def orient(a, b, c):
        val = (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

def _objective(params, N=12):
    """
    Objective: minimize outer side length, with large penalty if hexagons overlap.
    params: array of length 3*N + 3
      first 3*N: for each inner hex, (x, y, angle_deg)
      last 3: outer center (x, y) and outer angle
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    ox = params[3*N]
    oy = params[3*N+1]
    oa = params[3*N+2]
    
    # Check disjointness
    if not _all_pairs_disjoint(inner):
        return 1e10
    
    side = _outer_side_length(inner, (ox, oy), oa)
    return side

def _objective_smooth(params, N=12, log_scale=100.0):
    """
    Smooth surrogate: use log-sum-exp for the max in the outer side computation.
    Also smooth the disjointness constraint.
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    ox = params[3*N]
    oy = params[3*N+1]
    oa = params[3*N+2]
    
    theta = math.radians(oa)
    apothem_ratio = math.sqrt(3) / 2.0
    
    # Collect all dot products from all vertices of all inner hexagons
    all_dots = []
    for (icx, icy, iangle) in inner:
        verts = _hex_vertices(icx, icy, 1.0, iangle)
        for (vx, vy) in verts:
            dx = vx - ox
            dy = vy - oy
            for k in range(6):
                phi = theta + math.pi * (k + 0.5) / 3.0
                dot_val = dx * math.cos(phi) + dy * math.sin(phi)
                all_dots.append(dot_val)
    
    # Log-sum-exp for smooth max
    all_dots = np.array(all_dots)
    max_dot = np.max(all_dots)
    # Use a softer version: just use max but with smooth min for disjointness
    side = max(max_dot, 0.0) / apothem_ratio
    
    # Penalty for overlaps: check minimum distance between hexagons
    # This is expensive, so use a simplified penalty
    overlap_penalty = 0.0
    hex_verts = []
    for (cx, cy, ang) in inner:
        hex_verts.append(_hex_vertices(cx, cy, 1.0, ang))
    
    for i in range(N):
        for j in range(i + 1, N):
            # Simple penalty: if any vertex is inside the other hex, add penalty
            for v in hex_verts[i]:
                if _point_in_hex_simple(v[0], v[1], hex_verts[j]):
                    overlap_penalty += 1e6
            for v in hex_verts[j]:
                if _point_in_hex_simple(v[0], v[1], hex_verts[i]):
                    overlap_penalty += 1e6
    
    return side + overlap_penalty

def _initial_config():
    """Generate an initial feasible configuration."""
    # Place hexagons in a pattern that's likely to be non-overlapping
    # Use a honeycomb-like arrangement
    inner = []
    # For 12 hexagons, use a 3x4 or similar grid
    # Let's try placing them in rows
    dx = 3.0  # horizontal spacing for non-overlapping hexagons
    dy = 2.6  # vertical spacing
    
    positions = [
        (0, 0), (dx, 0), (2*dx, 0),
        (-dx/2, dy), (dx/2, dy), (3*dx/2, dy),
        (dx, 2*dy), (2*dx, 2*dy),
        (-dx/2, 3*dy), (dx/2, 3*dy), (3*dx/2, 3*dy),
        (dx, 4*dy)
    ]
    
    for (px, py) in positions[:N]:
        inner.append([px, py, 0.0])
    
    return inner

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initial configuration
    inner_init = _initial_config()
    
    # Compute initial side length
    cx_init = np.mean([h[0] for h in inner_init])
    cy_init = np.mean([h[1] for h in inner_init])
    side_init = _outer_side_length(inner_init, (cx_init, cy_init), 0.0)
    
    best_side = side_init
    best_params = None
    
    # Build initial params
    params_init = []
    for h in inner_init:
        params_init.extend([h[0], h[1], h[2]])
    params_init.extend([cx_init, cy_init, 0.0])
    params_init = np.array(params_init)
    
    # Define bounds
    bounds = []
    for i in range(N):
        bounds.append((-15, 15))  # x
        bounds.append((-15, 15))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-15, 15))  # outer x
    bounds.append((-15, 15))  # outer y
    bounds.append((-180, 180))  # outer angle
    
    # Try differential evolution
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            def de_objective(p):
                if time.time() > deadline:
                    return 1e10
                return _objective(p, N)
            
            result_de = differential_evolution(
                de_objective, bounds, maxiter=20, popsize=15,
                seed=42, tol=1e-6, polish=False,
                updating='deferred', init='latinhypercube',
                mutation=(0.5, 1.0), recombination=0.8
            )
            
            if result_de.fun < best_side:
                best_side = result_de.fun
                best_params = result_de.x.copy()
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Polish with L-BFGS-B
        if best_params is not None:
            start_p = best_params
        else:
            start_p = params_init
        
        try:
            def lbfgs_objective(p):
                if time.time() > deadline:
                    return 1e10
                return _objective(p, N)
            
            result_lbfgs = minimize(
                lbfgs_objective, start_p, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result_lbfgs.fun < best_side:
                best_side = result_lbfgs.fun
                best_params = result_lbfgs.x.copy()
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Fine-tuning pass: coordinate descent on angles with finer resolution
        if best_params is not None:
            p = best_params.copy()
            # Try small perturbations of each parameter
            for iteration in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(len(p)):
                    for delta in [0.1, -0.1, 0.01, -0.01]:
                        p_trial = p.copy()
                        p_trial[i] += delta
                        # Check bounds
                        if p_trial[i] < bounds[i][0] or p_trial[i] > bounds[i][1]:
                            continue
                        try:
                            val = _objective(p_trial, N)
                            if val < best_side - 1e-12:
                                best_side = val
                                p = p_trial
                                improved = True
                                break
                        except:
                            continue
                    if improved:
                        break
                if not improved:
                    break
            best_params = p
    
    # Extract results
    if best_params is not None:
        inner = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            a = best_params[3*i+2]
            inner.append([x, y, a])
        
        cx = best_params[3*N]
        cy = best_params[3*N+1]
        ca = best_params[3*N+2]
        
        side = _outer_side_length(inner, (cx, cy), ca)
        
        return inner, (cx, cy), side, ca
    else:
        # Fallback to initial config
        side = _outer_side_length(inner_init, (cx_init, cy_init), 0.0)
        return inner_init, (cx_init, cy_init), side, 0.0

# EVOLVE-BLOCK-END

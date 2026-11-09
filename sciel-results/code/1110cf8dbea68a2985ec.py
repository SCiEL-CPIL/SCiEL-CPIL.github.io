# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def _get_hex_vertices(center, side, angle_deg):
    """
    Get vertices of a regular hexagon.
    center: (x, y)
    side: float
    angle_deg: float (degrees)
    Returns: (6, 2) array of vertices.
    """
    cx, cy = center
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _check_validity(inner, outer, side):
    """
    Check if the packing is valid.
    inner: (11, 3) array (x, y, angle_deg)
    outer: (3,) array (x, y, angle_deg)
    side: float
    Returns: (is_valid, min_sep, min_cont)
    min_sep: minimum separation gap (should be > 1e-6)
    min_cont: minimum containment margin (should be >= -1e-6)
    """
    # Shift coordinates so outer center is at (0,0)
    outer_cx, outer_cy, outer_angle = outer
    shifted_inner = np.copy(inner)
    shifted_inner[:, 0] -= outer_cx
    shifted_inner[:, 1] -= outer_cy
    
    # Check edge lengths of inner hexagons (must be 1)
    for i in range(11):
        v = _get_hex_vertices(shifted_inner[i, :2], 1.0, shifted_inner[i, 2])
        for j in range(6):
            p1 = v[j]
            p2 = v[(j+1)%6]
            length = np.linalg.norm(p2 - p1)
            if abs(length - 1.0) > 1e-8:
                return False, 0, 0

    # Check containment
    # Outer hexagon centered at (0,0) with side 'side' and angle 'outer_angle'
    # Vertices of outer hexagon
    outer_verts = _get_hex_vertices((0,0), side, outer_angle)
    
    min_cont = np.inf
    for i in range(11):
        inner_verts = _get_hex_vertices(shifted_inner[i, :2], 1.0, shifted_inner[i, 2])
        for p in inner_verts:
            # Check against each edge of outer hexagon
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                # Cross product to check side
                # Edge vector
                e = p2 - p1
                # Vector from p1 to p
                v = p - p1
                # Cross product z-component
                cross = e[0]*v[1] - e[1]*v[0]
                # For a CCW polygon (which our vertices are, assuming standard orientation),
                # points inside should have cross >= 0 (if e is CCW).
                # Let's verify orientation of outer_verts.
                # Angles: angle, angle+60... CCW.
                # So cross >= 0 means inside.
                if cross < min_cont:
                    min_cont = cross
            if min_cont < -1e-6:
                return False, 0, min_cont

    # Check separation
    min_sep = np.inf
    # Precompute normals for each inner hexagon
    # Normals are perpendicular to edges.
    # For a hexagon with angle a, edges are at angles a+30+60k?
    # Vertex i is at a+60i. Edge i connects v_i to v_{i+1}.
    # Midpoint angle is a + 60i + 30.
    # Normal direction is a + 60i + 30 + 90 = a + 60i + 120?
    # Let's just calculate normals from vertices.
    
    for i in range(11):
        v1 = _get_hex_vertices(shifted_inner[i, :2], 1.0, shifted_inner[i, 2])
        normals1 = []
        for k in range(6):
            p1 = v1[k]
            p2 = v1[(k+1)%6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            n = n / np.linalg.norm(n)
            normals1.append(n)
            
    for i in range(11):
        for j in range(i+1, 11):
            v1 = _get_hex_vertices(shifted_inner[i, :2], 1.0, shifted_inner[i, 2])
            v2 = _get_hex_vertices(shifted_inner[j, :2], 1.0, shifted_inner[j, 2])
            
            # Get normals for both
            n1 = []
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k+1)%6]
                e = p2 - p1
                n = np.array([-e[1], e[0]])
                n = n / np.linalg.norm(n)
                n1.append(n)
                
            n2 = []
            for k in range(6):
                p1 = v2[k]
                p2 = v2[(k+1)%6]
                e = p2 - p1
                n = np.array([-e[1], e[0]])
                n = n / np.linalg.norm(n)
                n2.append(n)
                
            all_normals = n1 + n2
            
            max_gap = -np.inf
            for n in all_normals:
                # Project all vertices of hex 1 and 2 onto n
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                
                min1 = np.min(proj1)
                max1 = np.max(proj1)
                min2 = np.min(proj2)
                max2 = np.max(proj2)
                
                # Gap is distance between intervals
                # If max1 < min2, gap = min2 - max1
                # If max2 < min1, gap = min1 - max2
                # Else gap = -inf (overlap)
                gap = -np.inf
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    # Overlap on this axis
                    # But we need to check if they are separated on ANY axis.
                    # If they overlap on this axis, gap is effectively negative/zero for separation.
                    # The condition is: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
                    # So if we find ANY axis with gap > 1e-6, they are separated.
                    # If we check all 12 and none have gap > 1e-6, they are intersecting.
                    pass
                
                if gap > max_gap:
                    max_gap = gap
            
            if max_gap < min_sep:
                min_sep = max_gap
                
            if min_sep <= 1e-6:
                return False, min_sep, min_cont
                
    return True, min_sep, min_cont

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590
    
    # 1. Construct a known valid "flower" packing
    # Center at (0,0)
    # 6 hexagons in a ring.
    # Distance between centers of touching hexagons (aligned) is 2 * apothem = sqrt(3) ~ 1.732.
    # But we can rotate them to fit better?
    # Let's stick to aligned hexagons for simplicity and guaranteed validity if spaced correctly.
    # Spacing: If we place them at distance D from center, and they touch the center hex, D = sqrt(3).
    # If they touch each other, the distance between adjacent ring hexagons is also D (equilateral triangle of centers).
    # So a regular hexagon of centers with side length sqrt(3) works.
    # Ring radius = sqrt(3).
    
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1: 6 hexagons at radius sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
        
    # We need 4 more.
    # The "gaps" in the ring are at angles 30, 90, 150, 210, 270, 330.
    # Distance from center to gap vertex?
    # The ring hexagons touch at vertices?
    # Center of ring hex k: (r1 cos(60k), r1 sin(60k)).
    # Vertex of ring hex k pointing outwards: r1 + 1? No.
    # The vertex of the ring hexagon that is furthest from origin is at distance r1 + 1 (if aligned radially).
    # But they are not aligned radially.
    # Let's just place the 4 outer hexagons further out.
    # To be safe, place them at radius r2 = r1 + 2.0 = 3.732.
    # And angles 30, 90, 150, 210? Or 0, 60, 120, 180?
    # If we place them at 30, 90, 150, 210, they are in the gaps.
    # Let's try r2 = 3.8.
    
    r2 = 3.8
    for ang_deg in [30, 90, 150, 210]:
        ang = np.deg2rad(ang_deg)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
        
    inner = np.array([list(c) + [a] for c, a in zip(centers, angles)])
    outer = np.array([0.0, 0.0, 0.0])
    
    # 2. Calculate minimal enclosing hexagon side length
    # The outer hexagon is centered at (0,0) with angle 0.
    # We need to find the minimum S such that all inner vertices are inside.
    # For a regular hexagon with side S and angle 0, the apothem is S * sqrt(3)/2.
    # The condition for a point p to be inside is:
    # For each of the 6 normals n_k (at angles 30, 90, 150, 210, 270, 330):
    # dot(p, n_k) <= S * sqrt(3)/2
    # So S >= max_k ( dot(p, n_k) / (sqrt(3)/2) ) for all p in all inner vertices.
    
    max_proj = 0.0
    for i in range(11):
        v = _get_hex_vertices(inner[i, :2], 1.0, inner[i, 2])
        for p in v:
            for k in range(6):
                ang = np.deg2rad(30 + 60 * k)
                n = np.array([np.cos(ang), np.sin(ang)])
                proj = np.dot(p, n)
                if proj > max_proj:
                    max_proj = proj
                    
    side = max_proj / (np.sqrt(3)/2)
    
    # Add a small buffer to ensure validity due to floating point
    side *= 1.0001
    
    # Verify
    valid, sep, cont = _check_validity(inner, outer, side)
    if not valid:
        # If invalid, expand side further
        for _ in range(10):
            side *= 1.01
            valid, sep, cont = _check_validity(inner, outer, side)
            if valid:
                break
    
    # 3. Local Refinement (Optional, if time permits)
    # We can try to optimize the positions of the 4 outer hexagons to reduce side.
    # But this is risky. The current solution is valid.
    # Let's try a simple local search on the 4 outer positions.
    
    if time.time() < deadline - 10:
        # Define objective: side length
        # Variables: x, y for the 4 outer hexagons (indices 7, 8, 9, 10)
        # We keep the first 7 fixed.
        
        def objective(params):
            test_inner = np.copy(inner)
            test_inner[7, 0] = params[0]
            test_inner[7, 1] = params[1]
            test_inner[8, 0] = params[2]
            test_inner[8, 1] = params[3]
            test_inner[9, 0] = params[4]
            test_inner[9, 1] = params[5]
            test_inner[10, 0] = params[6]
            test_inner[10, 1] = params[7]
            
            # Recalculate minimal side
            max_proj = 0.0
            for i in range(11):
                v = _get_hex_vertices(test_inner[i, :2], 1.0, test_inner[i, 2])
                for p in v:
                    for k in range(6):
                        ang = np.deg2rad(30 + 60 * k)
                        n = np.array([np.cos(ang), np.sin(ang)])
                        proj = np.dot(p, n)
                        if proj > max_proj:
                            max_proj = proj
            new_side = max_proj / (np.sqrt(3)/2)
            
            # Check validity
            valid, sep, cont = _check_validity(test_inner, outer, new_side)
            if not valid:
                return 1e9
            return new_side
            
        x0 = np.array([inner[7, 0], inner[7, 1],
                       inner[8, 0], inner[8, 1],
                       inner[9, 0], inner[9, 1],
                       inner[10, 0], inner[10, 1]])
                       
        res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead',
                                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < side and res.fun < 1e8:
            inner[7, 0] = res.x[0]
            inner[7, 1] = res.x[1]
            inner[8, 0] = res.x[2]
            inner[8, 1] = res.x[3]
            inner[9, 0] = res.x[4]
            inner[9, 1] = res.x[5]
            inner[10, 0] = res.x[6]
            inner[10, 1] = res.x[7]
            
            # Recalculate side
            max_proj = 0.0
            for i in range(11):
                v = _get_hex_vertices(inner[i, :2], 1.0, inner[i, 2])
                for p in v:
                    for k in range(6):
                        ang = np.deg2rad(30 + 60 * k)
                        n = np.array([np.cos(ang), np.sin(ang)])
                        proj = np.dot(p, n)
                        if proj > max_proj:
                            max_proj = proj
            side = max_proj / (np.sqrt(3)/2) * 1.0001
            
            # Final validity check
            valid, sep, cont = _check_validity(inner, outer, side)
            if not valid:
                # Revert to previous valid side
                side = side / 1.0001 * 1.01

    return inner, outer, side

# EVOLVE-BLOCK-END

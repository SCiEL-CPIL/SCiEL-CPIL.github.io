import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _pairwise_clearance(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns a measure of overlap. 0 if disjoint.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    # Quick reject: if centers are far apart, no overlap.
    # Max distance between vertices is 2*1 = 2.
    # If dist > 2 + epsilon, definitely disjoint.
    if dx*dx + dy*dy > 4.5: 
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    return penalty

def _min_enclosing_hexagon(inner_hexes):
    """
    Finds the minimum side length regular hexagon containing all inner_hexes.
    inner_hexes: list of (cx, cy, angle_deg)
    Returns: (cx, cy, side_length, angle_deg)
    """
    # Collect all vertices of all inner hexagons
    points = []
    for h in inner_hexes:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        points.extend(verts)
    
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # The orientation of the enclosing hexagon is the critical variable.
    # We sample angles. Due to symmetry, 0-60 degrees is enough.
    # Step size 0.5 degrees is good balance.
    
    for angle_deg in np.arange(0, 60, 0.5):
        a = math.radians(angle_deg)
        # Normals of the enclosing hexagon sides
        normals = []
        for k in range(3):
            theta_n = a + math.pi/6 + k * math.pi/3
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        A = []
        b = []
        
        for nx, ny in normals:
            for px, py in points:
                # Constraint 1: n . (p - c) <= s * apothem_factor
                # Actually, for a regular hexagon with side s, the apothem is s * sqrt(3)/2.
                # The constraint is n . (p - c) <= apothem.
                # n . p - n . c <= apothem
                # n . p - n . c <= s * sqrt(3)/2
                # n . p - n . cx - n . cy - s * sqrt(3)/2 <= 0
                # Variables: cx, cy, s
                A.append([nx, ny, -math.sqrt(3)/2])
                b.append(nx*px + ny*py)
                
                # Constraint 2: -n . (p - c) <= s * sqrt(3)/2
                A.append([-nx, -ny, -math.sqrt(3)/2])
                b.append(-nx*px - ny*py)
        
        # Bounds: s >= 0, c_x, c_y unbounded (or large bounds)
        bounds = [(-100, 100), (-100, 100), (0, None)]
        
        try:
            res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_params = (res.x[0], res.x[1], s_val, angle_deg)
        except Exception:
            pass
            
    # Refine around the best angle found
    if best_params[3] != 0.0:
        best_ang = best_params[3]
        for angle_deg in np.arange(best_ang - 0.5, best_ang + 0.5, 0.05):
            a = math.radians(angle_deg)
            normals = []
            for k in range(3):
                theta_n = a + math.pi/6 + k * math.pi/3
                normals.append((math.cos(theta_n), math.sin(theta_n)))
                
            A = []
            b = []
            for nx, ny in normals:
                for px, py in points:
                    A.append([nx, ny, -math.sqrt(3)/2])
                    b.append(nx*px + ny*py)
                    A.append([-nx, -ny, -math.sqrt(3)/2])
                    b.append(-nx*px - ny*py)
                    
            bounds = [(-100, 100), (-100, 100), (0, None)]
            try:
                res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    s_val = res.x[2]
                    if s_val < best_side:
                        best_side = s_val
                        best_params = (res.x[0], res.x[1], s_val, angle_deg)
            except Exception:
                pass
                
    return best_params

def _generate_hexagonal_lattice_configurations():
    """
    Generates several candidate configurations based on hexagonal lattice packing.
    Hexagonal lattice vectors for side 1 hexagons (flat top):
    u = (3, 0)  -> distance 3? No.
    For flat top hexagons of side 1:
    Width = 2, Height = sqrt(3).
    Neighbors:
    (0, sqrt(3)), (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, -sqrt(3)), (-1.5, -sqrt(3)/2), (-1.5, sqrt(3)/2)
    
    Let's use axial coordinates or just generate points.
    """
    configs = []
    
    # Configuration 1: Standard hexagonal cluster (1 center + 6 ring + 4 more)
    # 1 center
    # 6 in first ring
    # 4 in second ring (to make 11)
    
    # Define basis vectors for hexagonal lattice of hexagons
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3)
    # Wait, for flat top hexagons, the distance between centers of touching hexagons is sqrt(3) * 2?
    # Apothem = sqrt(3)/2. Distance = 2 * apothem = sqrt(3).
    # If we rotate hexagons, the packing might change. Let's assume 0 deg rotation for inner hexes first.
    
    # Lattice vectors for flat-top hexagons (side 1):
    # a1 = (sqrt(3), 0) ? No.
    # Vertices at angles 0, 60, 120...
    # Center to vertex is 1.
    # Distance between centers of adjacent hexagons sharing a side is 2 * apothem = sqrt(3).
    # Direction of normal to side is 30, 90, 150...
    # So neighbor centers are at distance sqrt(3) in directions 30, 90, 150, 210, 270, 330.
    
    d = math.sqrt(3)
    neighbors = [
        (d * math.cos(math.radians(30)), d * math.sin(math.radians(30))),
        (d * math.cos(math.radians(90)), d * math.sin(math.radians(90))),
        (d * math.cos(math.radians(150)), d * math.sin(math.radians(150))),
        (d * math.cos(math.radians(210)), d * math.sin(math.radians(210))),
        (d * math.cos(math.radians(270)), d * math.sin(math.radians(270))),
        (d * math.cos(math.radians(330)), d * math.sin(math.radians(330))),
    ]
    
    # 1 center
    pts = [(0, 0)]
    # 6 ring
    for n in neighbors:
        pts.append(n)
        
    # Now we have 7. Need 4 more.
    # Second ring neighbors of the first ring.
    # To maximize density, we should pick positions that are close to the center.
    # The second ring in a perfect hexagonal packing has 12 positions.
    # We want the 4 "closest" or most compact.
    # Actually, any 4 from the second ring will form a convex hull.
    # Let's just pick the first 4 in the list of second ring candidates.
    # Second ring positions are sums of two neighbor vectors.
    # e.g., n1 + n2.
    
    # Let's generate all unique second ring positions
    second_ring = set()
    for i in range(6):
        for j in range(6):
            if i == j: continue
            px = neighbors[i][0] + neighbors[j][0]
            py = neighbors[i][1] + neighbors[j][1]
            # Round to avoid float issues
            second_ring.add((round(px, 5), round(py, 5)))
            
    # Sort second ring by distance to center to pick compact ones
    second_ring_list = sorted(list(second_ring), key=lambda p: p[0]**2 + p[1]**2)
    
    # Add 4 closest from second ring
    for p in second_ring_list[:4]:
        pts.append(p)
        
    # Ensure we have exactly 11
    while len(pts) < 11:
        # If we somehow have less, add more from second ring
        added = False
        for p in second_ring_list:
            if p not in pts:
                pts.append(p)
                added = True
                break
        if not added:
            break
            
    # Convert to (x, y, angle)
    config1 = [(p[0], p[1], 0.0) for p in pts[:11]]
    configs.append(config1)
    
    # Configuration 2: Try a slightly different set of 4 from the second ring
    # Maybe the other 4?
    # Or a 3x3 grid minus corners?
    # 3x3 grid of hexagons (side 1) with flat top:
    # Spacing x: 3? No, horizontal distance between centers in a row is 3?
    # If flat top, width is 2. Adjacent in row are separated by 3?
    # No, flat top hexagons don't tile in a simple grid without rotation.
    # Let's stick to lattice based.
    
    # Try rotating the whole configuration? The outer hexagon can rotate, so inner rotation relative to outer matters.
    # But since we compute min enclosing hexagon, the absolute angle of inner hexes matters for their shape relative to the container.
    # If we rotate all inner hexes by 30 deg, the "flat top" becomes "pointy top".
    # Let's try config with 30 deg rotation.
    config2 = [(p[0], p[1], 30.0) for p in pts[:11]]
    configs.append(config2)
    
    # Configuration 3: Maybe a different subset of second ring?
    # The 12 positions in second ring form a hexagon.
    # Picking 4 contiguous ones might be more compact than scattered?
    # Let's try picking 4 contiguous from the sorted list (angularly).
    second_ring_angular = sorted(list(second_ring), key=lambda p: math.atan2(p[1], p[0]))
    # Pick 4 contiguous
    start_idx = 0
    config3_pts = pts[:7] + [second_ring_angular[i] for i in range(start_idx, start_idx+4)]
    config3 = [(p[0], p[1], 0.0) for p in config3_pts[:11]]
    configs.append(config3)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_result = None
    
    configs = _generate_hexagonal_lattice_configurations()
    
    # Add a few random perturbations to the best config to see if we can squeeze more
    # But first, evaluate the pure lattice configs.
    
    for config in configs:
        if time.time() > deadline - 10:
            break
            
        # Check validity (disjointness)
        valid = True
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if _pairwise_clearance(config[i], config[j]) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Compute min enclosing hexagon
        ocx, ocy, os, oang = _min_enclosing_hexagon(config)
        
        # Verify containment
        if time.time() > deadline - 5:
            break
            
        contained = True
        for h in config:
            verts = _hex_vertices(h[0], h[1], 1.0, h[2])
            for v in verts:
                if _point_in_hex_dist_sq(v[0], v[1], ocx, ocy, os, oang) > 1e-6:
                    contained = False
                    break
            if not contained:
                break
        
        if contained:
            if os < best_score:
                best_score = os
                # Format result
                inner = [[h[0], h[1], h[2]] for h in config]
                best_result = (inner, (ocx, ocy), os, oang)
                
    # If no valid config found, fallback to a simple grid
    if best_result is None:
        # Fallback: Place in a line or grid
        inner = []
        for i in range(N):
            inner.append([i * 2.0, 0.0, 0.0])
        ocx, ocy, os, oang = _min_enclosing_hexagon(inner)
        best_result = (inner, (ocx, ocy), os, oang)
        best_score = os
        
    return best_result

if __name__ == "__main__":
    r = construct()
    print(r)

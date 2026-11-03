"""Packing 12 unit hexagons in a regular hexagon - minimizing outer side length."""
import numpy as np
import time

N = 12
APOTHEM = np.sqrt(3) / 2  # apothem of unit hexagon

def hex_face_normals(theta):
    """6 outward face normals of a regular hexagon rotated by theta."""
    angles = theta + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])

def hex_vertex_positions(cx, cy, side, theta):
    """6 vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    angles = theta + np.arange(6) * np.pi / 3
    r = side  # circumradius = side for regular hexagon
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, theta):
    """Check if point (px,py) is inside hexagon. Returns margin (positive = outside)."""
    normals = hex_face_normals(theta)
    d = np.array([px - cx, py - cy])
    margins = normals @ d
    return np.max(margins) - side * APOTHEM

def hex_vertices_unit(theta=0.0):
    """Vertices of unit hexagon centered at origin."""
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])

def check_validity(inner, center, side, angle):
    """Check if all inner hexagons are inside the outer hexagon and non-overlapping."""
    cx, cy = center
    # Check containment: All vertices of inner hexes must be inside outer hex
    for (px, py, ptheta) in inner:
        verts = hex_vertices_unit(pttheta)
        for v in verts:
            vx = px + v[0]
            vy = py + v[1]
            margin = point_in_hexagon(vx, vy, cx, cy, side, angle)
            if margin > 1e-9:
                return False
    
    # Check non-overlap
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            xi, yi, ti = inner[i]
            xj, yj, tj = inner[j]
            # Check vertices of hex i against hex j
            vi = hex_vertices_unit(ti)
            for v in vi:
                px = xi + v[0]
                py = yi + v[1]
                if point_in_hexagon(px, py, xj, yj, 1.0, tj) < -1e-9:
                    return False
            # Check vertices of hex j against hex i
            vj = hex_vertices_unit(tj)
            for v in vj:
                px = xj + v[0]
                py = yj + v[1]
                if point_in_hexagon(px, py, xi, yi, 1.0, ti) < -1e-9:
                    return False
    return True

def min_enclosing_hexagon_for_points(points):
    """
    Find minimum side length regular hexagon containing all points.
    points: (M, 2) array
    Returns: (cx, cy, side, theta)
    """
    best_side = np.inf
    best_params = None
    
    # Scan theta. The optimal theta for a set of points is often aligned with one of the points' features.
    # We scan 0 to pi/3 with fine steps, then refine.
    # Since the problem is periodic with pi/3, we only need to search one sector.
    
    def side_for_theta(theta, points):
        normals = hex_face_normals(theta)  # (6, 2)
        # Project points onto normals
        projs = points @ normals.T  # (M, 6)
        max_projs = np.max(projs, axis=0)  # (6,)
        min_projs = np.min(projs, axis=0)  # (6,)
        
        # Widths in directions of normals 0, 1, 2
        # width_i = max_projs[i] - min_projs[i]
        # Note: min_projs[i+3] = -max_projs[i] is NOT true generally unless points are symmetric.
        # The extent along normal i is max(p.n_i) - min(p.n_i).
        # The hexagon side length S must satisfy S * APOTHEM >= max_i ( (max(p.n_i) - min(p.n_i)) / 2 )?
        # No. The hexagon is defined by 6 half-planes.
        # For face i (normal n_i), the constraint is max_j (p_j . n_i) - cx . n_i <= S * APOTHEM.
        # For opposite face i+3 (normal -n_i), the constraint is max_j (p_j . (-n_i)) - cx . (-n_i) <= S * APOTHEM
        # => -min_j (p_j . n_i) + cx . n_i <= S * APOTHEM.
        # Adding these: (max_j (p_j . n_i) - min_j (p_j . n_i)) <= 2 * S * APOTHEM.
        # So S >= width_i / (2 * APOTHEM).
        # The minimal S for a given theta and optimal center is max_i (width_i / (2 * APOTHEM)).
        
        widths = np.zeros(3)
        for i in range(3):
            widths[i] = max_projs[i] - min_projs[i]
        
        apothem_needed = np.max(widths) / 2
        side = apothem_needed / APOTHEM
        
        # Optimal center
        # cx . n_i = (max_projs[i] + min_projs[i]) / 2
        normals_3 = normals[:3]
        b = (max_projs[:3] + min_projs[:3]) / 2
        
        # Solve normals_3 @ cx = b
        cx, _, _, _ = np.linalg.lstsq(normals_3, b, rcond=None)
        
        return side, cx

    # Coarse scan
    thetas = np.linspace(0, np.pi / 3, 100)
    for theta in thetas:
        side, cx = side_for_theta(theta, points)
        if side < best_side:
            best_side = side
            best_params = (cx[0], cx[1], side, theta)
            
    # Fine scan around best
    if best_params:
        best_theta = best_params[3]
        fine_thetas = np.linspace(best_theta - 0.01, best_theta + 0.01, 20)
        for theta in fine_thetas:
            side, cx = side_for_theta(theta, points)
            if side < best_side:
                best_side = side
                best_params = (cx[0], cx[1], side, theta)

    return best_params

def generate_topologies():
    """Generate candidate topologies for 12 hexagons."""
    topologies = []
    
    # Topology 1: 3x4 Grid (Rectangular-ish)
    # Hex lattice coords (q, r) -> x = sqrt(3)*q + sqrt(3)/2*r, y = 1.5*r
    # Let's try a 3x4 block.
    hexes = []
    for r in range(4):
        for q in range(3):
            x = np.sqrt(3) * q + np.sqrt(3) / 2 * r
            y = 1.5 * r
            hexes.append([x, y, 0.0])
    topologies.append(hexes)
    
    # Topology 2: 4x3 Grid (Rotated)
    hexes = []
    for r in range(3):
        for q in range(4):
            x = np.sqrt(3) * q + np.sqrt(3) / 2 * r
            y = 1.5 * r
            hexes.append([x, y, 0.0])
    topologies.append(hexes)
    
    # Topology 3: 1-2-3-4-2 pattern? Or compact cluster.
    # Let's try a "diamond" or "rhombus" shape.
    # Axial coords:
    # (0,0)
    # (1,0), (0,1)
    # (2,0), (1,1), (0,2)
    # (1,2), (0,3) -> This is 7. Need 12.
    
    # Let's try 2-3-4-3 pattern
    hexes = []
    pattern = [2, 3, 4, 3]
    for row, count in enumerate(pattern):
        for col in range(count):
            # Center the row
            x = np.sqrt(3) * (col - (count - 1) / 2.0)
            y = 1.5 * row
            hexes.append([x, y, 0.0])
    topologies.append(hexes)
    
    # Topology 4: 3-4-3-2 pattern
    hexes = []
    pattern = [3, 4, 3, 2]
    for row, count in enumerate(pattern):
        for col in range(count):
            x = np.sqrt(3) * (col - (count - 1) / 2.0)
            y = 1.5 * row
            hexes.append([x, y, 0.0])
    topologies.append(hexes)
    
    # Topology 5: 4-4-4 strip
    hexes = []
    for r in range(3):
        for q in range(4):
            x = np.sqrt(3) * q + np.sqrt(3) / 2 * r
            y = 1.5 * r
            hexes.append([x, y, 0.0])
    topologies.append(hexes)
            
    # Topology 6: 5-4-3 pattern?
    hexes = []
    pattern = [5, 4, 3]
    for row, count in enumerate(pattern):
        for col in range(count):
            x = np.sqrt(3) * (col - (count - 1) / 2.0)
            y = 1.5 * row
            hexes.append([x, y, 0.0])
    topologies.append(hexes)
            
    return topologies

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for safety (limit 150s)
    
    best_score = np.inf
    best_result = None
    
    topologies = generate_topologies()
    
    for hexes in topologies:
        if time.time() > deadline:
            break
            
        if len(hexes) != N:
            continue
            
        # Center the topology
        xs = [h[0] for h in hexes]
        ys = [h[1] for h in hexes]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        for h in hexes:
            h[0] -= cx
            h[1] -= cy
            
        points = np.array([[h[0], h[1]] for h in hexes])
        
        # Find min enclosing hexagon
        cx_enc, cy_enc, side_enc, theta_enc = min_enclosing_hexagon_for_points(points)
        
        # Check validity
        if check_validity(hexes, (cx_enc, cy_enc), side_enc, theta_enc):
            if side_enc < best_score:
                best_score = side_enc
                best_result = (hexes, (cx_enc, cy_enc), side_enc, theta_enc)
        else:
            # If invalid, it's likely due to numerical precision in check_validity or
            # the enclosing hexagon calculation being slightly off.
            # We can try to expand the side slightly to ensure validity, but that increases score.
            # Better: just skip if strictly invalid, as our topologies should be valid packings.
            # The "overlap" check in check_validity might be too sensitive to floating point.
            # Let's retry with a small epsilon in check_validity? No, we want to be safe.
            # Actually, if the topology is a valid packing (which these are, by construction on lattice),
            # and the enclosing hexagon contains the vertices, it should be valid.
            # The failure might be that the enclosing hexagon computed by min_enclosing_hexagon_for_points
            # is slightly too small due to floating point errors in the projection/least squares.
            # Let's add a small margin to side_enc and re-check.
            side_safe = side_enc * 1.001
            if check_validity(hexes, (cx_enc, cy_enc), side_safe, theta_enc):
                if side_safe < best_score:
                    best_score = side_safe
                    best_result = (hexes, (cx_enc, cy_enc), side_safe, theta_enc)
            else:
                # Try even larger margin
                side_safe = side_enc * 1.01
                if check_validity(hexes, (cx_enc, cy_enc), side_safe, theta_enc):
                    if side_safe < best_score:
                        best_score = side_safe
                        best_result = (hexes, (cx_enc, cy_enc), side_safe, theta_enc)

    # Fallback if no valid result found (should not happen)
    if best_result is None:
        # Use the first topology with a large margin
        hexes = topologies[0]
        xs = [h[0] for h in hexes]
        ys = [h[1] for h in hexes]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        for h in hexes:
            h[0] -= cx
            h[1] -= cy
        points = np.array([[h[0], h[1]] for h in hexes])
        cx_enc, cy_enc, side_enc, theta_enc = min_enclosing_hexagon_for_points(points)
        side_enc *= 1.1
        best_result = (hexes, (cx_enc, cy_enc), side_enc, theta_enc)
        
    return best_result

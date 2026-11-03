import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform to hexagon's local frame
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate back
    x = dx * np.cos(-angle) - dy * np.sin(-angle)
    y = dx * np.sin(-angle) + dy * np.cos(-angle)
    # For a regular hexagon with side s, the vertices are at distance s from center
    # The hexagon can be described by |x| <= s and |y| <= s*sqrt(3)/2 in a rotated frame
    # Actually, for a regular hexagon with "radius" (center to vertex) = side,
    # in the local frame where one vertex is at (s, 0), the constraints are:
    # |x| <= s, |y| <= s*sqrt(3)/2, and |x| + |y|/sqrt(3) <= s... 
    # Let me use the support function approach instead.
    
    # A regular hexagon with circumradius R=side has 6 faces.
    # The face normals are at angles 0, 60, 120, 180, 240, 300 degrees (in local frame)
    # The apothem (distance from center to face) = R * cos(pi/6) = R * sqrt(3)/2
    R = side
    apothem = R * np.sqrt(3) / 2
    for i in range(6):
        theta = i * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        # The constraint is: x*nx + y*ny <= apothem
        if x * nx + y * ny > apothem + 1e-9:
            return False
    return True

def all_points_in_hexagon(points, cx, cy, side, angle_deg):
    """Check if all points are inside the hexagon."""
    for p in points:
        if not point_in_hexagon(p[0], p[1], cx, cy, side, angle_deg):
            return False
    return True

def compute_min_outer_side(inner_hexes):
    """
    Compute the minimum side length of a regular hexagon containing all inner hexagons.
    inner_hexes: list of [x, y, angle_deg] for each inner hexagon (side=1)
    Returns (best_side, best_cx, best_cy, best_angle_deg)
    """
    # Get all vertices of all inner hexagons
    all_verts = []
    for hx, hy, ha in inner_hexes:
        verts = hex_vertices(hx, hy, 1.0, ha)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # The outer hexagon is defined by (cx, cy, S, theta)
    # We want to minimize S such that all vertices are inside.
    
    def objective(params):
        cx, cy, S, theta = params
        if S <= 0:
            return 1e10
        # Check all points
        angle = np.radians(theta)
        apothem = S * np.sqrt(3) / 2
        max_violation = 0
        for p in all_verts:
            dx = p[0] - cx
            dy = p[1] - cy
            x = dx * np.cos(-angle) - dy * np.sin(-angle)
            y = dx * np.sin(-angle) + dy * np.cos(-angle)
            for i in range(6):
                phi = i * np.pi / 3
                val = x * np.cos(phi) + y * np.sin(phi)
                if val > apothem:
                    violation = val - apothem
                    if violation > max_violation:
                        max_violation = violation
        # Penalize if any violation
        if max_violation > 0:
            return S + 100 * max_violation
        return S
    
    # Try multiple initial guesses
    best_result = None
    best_obj = np.inf
    
    # Initial guess: center at centroid of all vertices
    cx0 = np.mean(all_verts[:, 0])
    cy0 = np.mean(all_verts[:, 1])
    
    # Estimate initial side
    max_dist = np.max(np.sqrt((all_verts[:, 0] - cx0)**2 + (all_verts[:, 1] - cy0)**2))
    S0 = max_dist * 1.2
    
    # Try different rotations
    for theta0 in [0, 30, 60, 90, 120, 150]:
        x0 = np.array([cx0, cy0, S0, theta0])
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except:
            pass
    
    if best_result is None:
        return S0, cx0, cy0, 0.0
    
    cx, cy, S, theta = best_result
    return S, cx, cy, theta

def try_arrangement(arrangement):
    """Try a specific arrangement and return the outer hexagon parameters."""
    inner = arrangement
    S, cx, cy, theta = compute_min_outer_side(inner)
    return inner, (cx, cy), S, theta

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Define several candidate arrangements for 12 unit hexagons
    arrangements = []
    
    # Arrangement 1: Hexagonal close packing - center + 6 around + 5 more
    # Using hex grid coordinates
    # In a hex grid with spacing sqrt(3) between centers for touching hexagons
    d = np.sqrt(3)  # distance between centers of touching unit hexagons
    
    # Hexagonal arrangement: positions in a hexagonal pattern
    # Ring 0: (0,0)
    # Ring 1: 6 positions at distance d
    # Ring 2: 12 positions, we need 5 more for total 12
    
    # Let me use axial coordinates for hex grid
    # Position in Cartesian: x = d * (q + r/2), y = d * (sqrt(3)/2) * r
    
    def axial_to_cart(q, r):
        x = d * (q + r / 2)
        y = d * (np.sqrt(3) / 2) * r
        return x, y
    
    # Compact hexagonal arrangement for 12 hexagons
    # Try: a hexagon of side 2 would have 1+6+12=19, too many
    # A smaller cluster: let's try various patterns
    
    # Pattern A: 4-4-4 in three rows (offset)
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset by d/2 in x)
    # Row 2: 4 hexagons
    arr_a = []
    for row in range(3):
        for col in range(4):
            x = col * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_a.append([x, y, 0.0])
    arrangements.append(arr_a)
    
    # Pattern B: Hexagonal shape - center + ring of 6 + 5 from next ring
    # Axial coords for a hex of "radius" 1: all (q,r) with max(|q|,|r|,|q+r|) <= 1
    # That gives 7 hexagons. Add 5 more from radius 2.
    hex_coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 1:
                hex_coords.append((q, r))
    # That's 7. Add 5 more from the next ring
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) == 2:
                ring2.append((q, r))
    # Take first 5 from ring2
    selected = hex_coords + ring2[:5]
    arr_b = []
    for q, r in selected:
        x, y = axial_to_cart(q, r)
        arr_b.append([x, y, 0.0])
    arrangements.append(arr_b)
    
    # Pattern C: 3-4-3-2 arrangement
    arr_c = []
    row_counts = [3, 4, 3, 2]
    for row, count in enumerate(row_counts):
        offset = (4 - count) / 2  # Center each row
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_c.append([x, y, 0.0])
    arrangements.append(arr_c)
    
    # Pattern D: 2-4-4-2
    arr_d = []
    row_counts = [2, 4, 4, 2]
    for row, count in enumerate(row_counts):
        offset = (4 - count) / 2
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_d.append([x, y, 0.0])
    arrangements.append(arr_d)
    
    # Pattern E: 5-3-4 (three rows)
    arr_e = []
    row_counts = [5, 3, 4]
    for row, count in enumerate(row_counts):
        offset = (5 - count) / 2
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_e.append([x, y, 0.0])
    arrangements.append(arr_e)
    
    # Pattern F: 4-5-3
    arr_f = []
    row_counts = [4, 5, 3]
    for row, count in enumerate(row_counts):
        offset = (5 - count) / 2
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_f.append([x, y, 0.0])
    arrangements.append(arr_f)
    
    # Pattern G: 6-6 (two rows of 6)
    arr_g = []
    for row in range(2):
        for col in range(6):
            x = col * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_g.append([x, y, 0.0])
    arrangements.append(arr_g)
    
    # Pattern H: 6-3-3
    arr_h = []
    row_counts = [6, 3, 3]
    for row, count in enumerate(row_counts):
        offset = (6 - count) / 2
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_h.append([x, y, 0.0])
    arrangements.append(arr_h)
    
    # Pattern I: 3-6-3
    arr_i = []
    row_counts = [3, 6, 3]
    for row, count in enumerate(row_counts):
        offset = (6 - count) / 2
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_i.append([x, y, 0.0])
    arrangements.append(arr_i)
    
    # Pattern J: 2-5-5
    arr_j = []
    row_counts = [2, 5, 5]
    for row, count in enumerate(row_counts):
        offset = (5 - count) / 2
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_j.append([x, y, 0.0])
    arrangements.append(arr_j)
    
    # Pattern K: 5-5-2
    arr_k = []
    row_counts = [5, 5, 2]
    for row, count in enumerate(row_counts):
        offset = (5 - count) / 2
        for col in range(count):
            x = (col + offset) * d + (row % 2) * d / 2
            y = row * d * np.sqrt(3) / 2
            arr_k.append([x, y, 0.0])
    arrangements.append(arr_k)
    
    # Pattern L: 4-4-4 with different offset
    arr_l = []
    for row in range(3):
        for col in range(4):
            x = col * d
            y = row * d * np.sqrt(3) / 2
            arr_l.append([x, y, 0.0])
    arrangements.append(arr_l)
    
    # Evaluate all arrangements
    best_S = np.inf
    best_result = None
    
    for i, arr in enumerate(arrangements):
        if time.time() > deadline - 5:
            break
        try:
            result = try_arrangement(arr)
            S = result[2]
            if S < best_S:
                best_S = S
                best_result = result
        except Exception as e:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the seed arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

"""Packing 12 unit hexagons in a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side and rotation."""
    a = np.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    # Rotate by -angle
    cos_a, sin_a = np.cos(-a), np.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Hexagon with side s has vertices at angles 0, 60, 120, 180, 240, 300
    # In local frame, check against 3 pairs of parallel sides
    # The hexagon can be described by |x| <= s, |x/2 + y*sqrt(3)/2| <= s, |-x/2 + y*sqrt(3)/2| <= s
    # Actually for a regular hexagon with side s centered at origin with flat top:
    # Vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # The constraints are: |x| <= s, |y| <= s*sqrt(3)/2, |x/2 + y*sqrt(3)/2| <= s, etc.
    # Simpler: use the 6 half-plane tests
    sqrt3 = np.sqrt(3)
    # Normal vectors to the 6 sides (outward)
    # Side normals point at angles 30, 90, 150, 210, 270, 330 degrees for flat-top hex
    # For a hex with vertices at 0,60,...,300 deg, the side normals are at 30,90,...,330
    # Distance from center to each side = s * sqrt(3) / 2
    apothem = side * sqrt3 / 2
    normal_angles = np.radians(30 + np.arange(6) * 60)
    nx = apothem * np.cos(normal_angles)
    ny = apothem * np.sin(normal_angles)
    # Point is inside if dot(point, normal) <= apothem for all 6 sides
    dots = lx * nx + ly * ny
    return np.all(dots <= apothem + 1e-9)

def all_points_in_hex(points, cx, cy, side, angle_deg):
    """Check if all points are inside the hexagon."""
    a = np.radians(angle_deg)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    cos_a, sin_a = np.cos(-a), np.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    sqrt3 = np.sqrt(3)
    apothem = side * sqrt3 / 2
    normal_angles = np.radians(30 + np.arange(6) * 60)
    nx = apothem * np.cos(normal_angles)
    ny = apothem * np.sin(normal_angles)
    # For each point, check all 6 half-planes
    # dots[i, j] = lx[i]*nx[j] + ly[i]*ny[j]
    dots = lx[:, np.newaxis] * np.newaxis * nx[np.newaxis, :] + ly[:, np.newaxis] * np.newaxis * ny[np.newaxis, :]
    # Actually let me redo this more carefully
    dots = lx[:, np.newaxis] * nx[np.newaxis, :] + ly[:, np.newaxis] * ny[np.newaxis, :]
    return np.all(dots <= apothem + 1e-9)

def get_all_hex_points(inner, outer_cx, outer_cy, outer_side, outer_angle):
    """Get all vertices of all inner hexagons."""
    all_pts = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, iang)
        all_pts.append(verts)
    return np.vstack(all_pts)

def compute_required_side(inner, cx, cy, angle_deg):
    """Compute minimum side length of outer hex centered at (cx,cy) with given angle to contain all inner hex vertices."""
    points = get_all_hex_points(inner, cx, cy, 0, 0)
    a = np.radians(angle_deg)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    cos_a, sin_a = np.cos(-a), np.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    sqrt3 = np.sqrt(3)
    # For the outer hex with side s, apothem = s*sqrt(3)/2
    # Constraint: for each point, max over 6 directions of (lx*n_x + ly*n_y) <= apothem
    # n_x, n_y are unit normals at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.radians(30 + np.arange(6) * 60)
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    # dots[i,j] = lx[i]*nx[j] + ly[i]*ny[j]
    dots = lx[:, np.newaxis] * nx[np.newaxis, :] + ly[:, np.newaxis] * ny[np.newaxis, :]
    max_dot = np.max(dots)
    # apothem >= max_dot, so s*sqrt(3)/2 >= max_dot, s >= max_dot * 2/sqrt(3)
    return max_dot * 2.0 / sqrt3

def objective(params, inner):
    """Objective: minimize outer side length."""
    cx, cy, angle_deg = params
    return compute_required_side(inner, cx, cy, angle_deg)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Create a hexagonal cluster of 12 unit hexagons
    # Natural arrangement: center hex + ring of 6 + partial second ring
    # For 12 hexagons: 1 center + 6 in first ring + 5 in second ring
    # Or: 2 rows of 4, 2 rows of 3, 1 row of 2? Let me think about hex grid coords.
    
    # Use axial coordinates for hex grid
    # A hex with side 1 has distance between centers = sqrt(3) for adjacent hexes
    # Let's place them in a compact cluster
    
    # Hex grid: use offset coordinates
    # Row 0: 4 hexes
    # Row 1: 4 hexes (offset by half spacing)
    # Row 2: 4 hexes
    # Total: 12
    
    # Actually, let me use a more natural hex packing:
    # Center at origin, then 6 around it, then 5 more in the next ring
    # Distance between adjacent hex centers = 2 * apothem = sqrt(3)
    
    d = np.sqrt(3)  # distance between adjacent hex centers
    
    # Place 12 hexes in a hexagonal pattern
    # Using cube coordinates converted to 2D
    # Let's just manually place them in a compact arrangement
    
    # Arrangement: 3-4-3-2 pattern or similar
    # Let me use a simple grid:
    # Row 0 (y=0): x = -1.5d, -0.5d, 0.5d, 1.5d  (4 hexes)
    # Row 1 (y=d*1.5): x = -d, 0, d  (3 hexes) -- wait, that's only 7
    
    # Better: use a proper hex cluster
    # 12 hexes: center + 6 neighbors + 5 from second ring
    # Center: (0,0)
    # Ring 1 (6 hexes at distance d): angles 0, 60, 120, 180, 240, 300
    # Ring 2: pick 5 positions at distance 2d or sqrt(3)*d... 
    
    # Actually for hex tiling, the centers form a triangular lattice.
    # Let me use basis vectors:
    # v1 = (d, 0), v2 = (d/2, d*sqrt(3)/2)
    
    # Place hexes at lattice points to form a compact shape
    # For 12 hexes, a good shape is a "hexagon" of side 2 minus corners, or
    # a 3x4 rectangular patch in the hex grid
    
    # Let me use axial coordinates (q, r) and convert:
    # x = d * (q + r/2), y = d * sqrt(3)/2 * r
    
    # A 3x4 block: q in [0,2], r in [0,3] gives 12 hexes
    # But that's a parallelogram. Let me use a more symmetric arrangement.
    
    # Symmetric hexagonal cluster of 12:
    # Use all (q,r) with |q|<=2, |r|<=2, |q+r|<=2 but that's 19... too many
    # 
    # Let me just do a 2x6 or 3x4 arrangement and optimize.
    
    # Simple approach: 4 columns of 3
    inner = []
    for col in range(4):
        for row in range(3):
            x = col * d
            y = row * d * np.sqrt(3) / 2
            if col % 2 == 1:
                y += d * np.sqrt(3) / 4  # offset odd columns
            inner.append([x, y, 0.0])
    
    # Center the cluster
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx_mean = (min(xs) + max(xs)) / 2
    cy_mean = (min(ys) + max(ys)) / 2
    inner = [[p[0] - cx_mean, p[1] - cy_mean, 0.0] for p in inner]
    
    # Initial outer hex parameters
    best_inner = inner
    best_cx = 0.0
    best_cy = 0.0
    best_side = 100.0
    best_angle = 0.0
    
    # Try multiple initial angles and optimize
    for init_angle in [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > deadline:
            break
        
        x0 = [0.0, 0.0, float(init_angle)]
        try:
            result = minimize(objective, x0, args=(inner,), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < best_side:
                cx, cy, angle = result.x
                side = result.fun
                if side < best_side:
                    best_cx, best_cy, best_side, best_angle = cx, cy, side, angle
        except:
            pass
    
    # Also try different inner arrangements
    # Arrangement 2: 3x4 with different offset
    inner2 = []
    for row in range(3):
        for col in range(4):
            x = col * d
            y = row * d * np.sqrt(3) / 2
            if row % 2 == 1:
                x += d / 2
            inner2.append([x, y, 0.0])
    xs = [p[0] for p in inner2]
    ys = [p[1] for p in inner2]
    cx_mean = (min(xs) + max(xs)) / 2
    cy_mean = (min(ys) + max(ys)) / 2
    inner2 = [[p[0] - cx_mean, p[1] - cy_mean, 0.0] for p in inner2]
    
    for init_angle in [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > deadline:
            break
        x0 = [0.0, 0.0, float(init_angle)]
        try:
            result = minimize(objective, x0, args=(inner2,), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < best_side:
                cx, cy, angle = result.x
                side = result.fun
                if side < best_side:
                    best_inner = inner2
                    best_cx, best_cy, best_side, best_angle = cx, cy, side, angle
        except:
            pass
    
    # Arrangement 3: hexagonal cluster (center + 6 + 5)
    inner3 = [[0.0, 0.0, 0.0]]
    for i in range(6):
        ang = np.radians(60 * i)
        inner3.append([d * np.cos(ang), d * np.sin(ang), 0.0])
    # Add 5 more from second ring
    for i in range(6):
        if i == 5:  # skip one to get 12 total
            continue
        ang = np.radians(60 * i)
        inner3.append([2 * d * np.cos(ang), 2 * d * np.sin(ang), 0.0])
    # That gives 1 + 6 + 5 = 12
    xs = [p[0] for p in inner3]
    ys = [p[1] for p in inner3]
    cx_mean = (min(xs) + max(xs)) / 2
    cy_mean = (min(ys) + max(ys)) / 2
    inner3 = [[p[0] - cx_mean, p[1] - cy_mean, 0.0] for p in inner3]
    
    for init_angle in [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > deadline:
            break
        x0 = [0.0, 0.0, float(init_angle)]
        try:
            result = minimize(objective, x0, args=(inner3,), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < best_side:
                cx, cy, angle = result.x
                side = result.fun
                if side < best_side:
                    best_inner = inner3
                    best_cx, best_cy, best_side, best_angle = cx, cy, side, angle
        except:
            pass
    
    # Final refinement with L-BFGS-B on the best arrangement
    if time.time() < deadline - 5:
        x0 = [best_cx, best_cy, best_angle]
        try:
            result = minimize(objective, x0, args=(best_inner,), method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-12})
            if result.fun < best_side:
                best_cx, best_cy, best_angle = result.x
                best_side = result.fun
        except:
            pass
    
    return best_inner, (best_cx, best_cy), best_side, best_angle


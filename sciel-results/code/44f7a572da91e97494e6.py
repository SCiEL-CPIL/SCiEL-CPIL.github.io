import numpy as np

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.radians(np.arange(0, 360, 60))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    for k in range(6):
        mid_angle = np.radians(k * 60 + 30)
        nx = np.cos(mid_angle)
        ny = np.sin(mid_angle)
        if nx * lx + ny * ly > side * np.sqrt(3) / 2 + 1e-9:
            return False
    return True

def all_points_in_hexagon(points, cx, cy, side, angle_deg):
    for px, py in points:
        if not point_in_hexagon(px, py, cx, cy, side, angle_deg):
            return False
    return True

def min_side_for_hexagon(points, cx, cy, angle_deg):
    lo = 0.0
    hi = 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if all_points_in_hexagon(points, cx, cy, mid, angle_deg):
            hi = mid
        else:
            lo = mid
    return hi

def hex_spiral_positions(n):
    positions = [(0, 0)]
    ring = 1
    while len(positions) < n:
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        q, r = 0, -ring
        for d in range(6):
            for step in range(ring):
                if len(positions) >= n:
                    break
                positions.append((q, r))
                q += directions[d][0]
                r += directions[d][1]
            if len(positions) >= n:
                break
        ring += 1
    return positions[:n]

def axial_to_cartesian(q, r):
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def construct():
    axial_pos = hex_spiral_positions(N)
    inner = []
    for q, r in axial_pos:
        x, y = axial_to_cartesian(q, r)
        inner.append([float(x), float(y), 0.0])
    
    all_pts = []
    for x, y, ang in inner:
        verts = hexagon_vertices(x, y, 1.0, ang)
        all_pts.extend(verts.tolist())
    
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    
    best_side = float('inf')
    best_angle = 0.0
    
    for angle_deg in np.linspace(0, 30, 61):
        side = min_side_for_hexagon(all_pts, cx, cy, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return inner, (float(cx), float(cy)), float(best_side), float(best_angle)

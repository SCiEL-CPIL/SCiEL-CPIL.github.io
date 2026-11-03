import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon (flat-top when angle=0)."""
    a = np.radians(angle_deg)
    # Flat-top: vertices at 30, 90, 150, 210, 270, 330 degrees
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_sdf(px, py, cx, cy, side, angle_deg):
    """Signed distance from point to hexagon boundary. Negative if inside."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate into local frame
    ca, sa = np.cos(-a), np.sin(-a)
    rx = dx * ca - dy * sa
    ry = dx * sa + dy * ca
    
    apothem = side * np.sqrt(3) / 2
    # Edge normals at 0, 60, 120, 180, 240, 300 degrees for flat-top
    edge_normals = np.array([
        [1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
        [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
    ])
    # Project point onto each normal
    projections = edge_normals @ np.array([rx, ry])
    # The hexagon is where all projections <= apothem
    # SDF approximation: max(projections) - apothem
    return max(projections) - apothem

def _min_enclosing_hexagon(points):
    """Find minimum side length regular hexagon containing all points."""
    points = np.array(points)
    
    def objective(params):
        cx, cy, side, angle = params
        if side <= 0:
            return 1e10
        # Compute max violation
        max_viol = 0
        for px, py in points:
            v = _hex_sdf(px, py, cx, cy, side, angle)
            if v > max_viol:
                max_viol = v
        return side + 1000 * max(0, max_viol)
    
    xs = points[:, 0]
    ys = points[:, 1]
    cx0 = (xs.min() + xs.max()) / 2
    cy0 = (ys.min() + ys.max()) / 2
    side0 = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.7 + 1
    angle0 = 0.0
    
    x0 = np.array([cx0, cy0, side0, angle0])
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
    cx, cy, side, angle = res.x
    return cx, cy, max(0.1, side), angle

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    sqrt3 = np.sqrt(3)
    
    best_side = float('inf')
    best_result = None
    
    # Try multiple grid arrangements
    arrangements = []
    
    # 4x3 honeycomb
    arr1 = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            arr1.append([x, y, 0.0])
    arrangements.append(arr1)
    
    # 3x4 honeycomb
    arr2 = []
    for row in range(4):
        for col in range(3):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            arr2.append([x, y, 0.0])
    arrangements.append(arr2)
    
    # 2x6 honeycomb
    arr3 = []
    for row in range(6):
        for col in range(2):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            arr3.append([x, y, 0.0])
    arrangements.append(arr3)
    
    # 6x2 honeycomb
    arr4 = []
    for row in range(2):
        for col in range(6):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            arr4.append([x, y, 0.0])
    arrangements.append(arr4)
    
    # Compact hexagonal cluster: center + 6 around + 5 more
    # Hexagonal numbers: 1, 7, 19... 12 is not a hexagonal number
    # Try: 1 center, 6 in first ring, 5 in second ring (partial)
    arr5 = []
    # Center
    arr5.append([0, 0, 0.0])
    # First ring (6 hexagons)
    for i in range(6):
        ang = i * np.pi / 3
        x = sqrt3 * np.cos(ang)
        y = 1.5 * np.sin(ang)
        arr5.append([x, y, 0.0])
    # Second ring (5 hexagons) - place them in gaps
    # Second ring positions at distance ~2*sqrt3 or so
    for i in range(5):
        ang = i * 2 * np.pi / 5 + np.pi / 10  # offset to avoid overlap
        r = 2 * sqrt3
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        arr5.append([x, y, 0.0])
    arrangements.append(arr5)
    
    for arr in arrangements:
        if time.time() > deadline - 20:
            break
        
        # Center the arrangement
        xs = [p[0] for p in arr]
        ys = [p[1] for p in arr]
        cx_off = (min(xs) + max(xs)) / 2
        cy_off = (min(ys) + max(ys)) / 2
        arr_centered = [[p[0] - cx_off, p[1] - cy_off, p[2]] for p in arr]
        
        # Get all vertices
        all_pts = []
        for x, y, ang in arr_centered:
            verts = _hex_vertices(x, y, 1.0, ang)
            all_pts.extend(verts.tolist())
        
        # Try several outer hexagon rotations
        for outer_rot in [0, 15, 30, 45, 60, 75, 90]:
            if time.time() > deadline - 10:
                break
            
            # Rotate inner arrangement
            a = np.radians(outer_rot)
            arr_rot = []
            for x, y, ang in arr_centered:
                nx = x * np.cos(a) - y * np.sin(a)
                ny = x * np.sin(a) + y * np.cos(a)
                arr_rot.append([nx, ny, ang + outer_rot])
            
            all_pts_rot = []
            for x, y, ang in arr_rot:
                verts = _hex_vertices(x, y, 1.0, ang)
                all_pts_rot.extend(verts.tolist())
            
            cx, cy, side, angle = _min_enclosing_hexagon(all_pts_rot)
            
            if side < best_side:
                best_side = side
                best_result = (arr_rot, (cx, cy), side, angle)
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, angle = best_result
    return inner, center, side, angle

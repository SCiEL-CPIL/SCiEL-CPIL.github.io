import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon with given center, side, and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform to hexagon's local frame
    angle_rad = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point by -angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    
    # In local frame, hexagon has flat top (vertices at 30°, 90°, 150°, 210°, 270°, 330°)
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the apothem is s * sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    
    # The 6 normal directions (pointing inward) are at angles 0, 60, 120, 180, 240, 300 in local frame
    # Actually, for vertices at 30, 90, 150, 210, 270, 330, the face normals are at 0, 60, 120, 180, 240, 300
    for k in range(6):
        theta = k * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        # The face is at distance apothem from center in direction theta
        if nx * lx + ny * ly > apothem:
            return False
    return True

def polygon_contains_hexagon(outer_cx, outer_cy, outer_side, outer_angle, 
                              inner_cx, inner_cy, inner_side, inner_angle):
    """Check if outer hexagon contains inner hexagon."""
    # Check all vertices of inner hexagon are inside outer hexagon
    ix, iy = hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for i in range(6):
        if not point_in_hexagon(ix[i], iy[i], outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def compute_min_outer_hexagon(inner_positions, inner_angles, inner_side=1.0):
    """Find minimal outer hexagon containing all inner hexagons.
    Returns (cx, cy, side, angle_deg) and the side length."""
    
    def objective(params):
        cx, cy, side, angle = params
        if side <= 0:
            return 1e10
        for i in range(len(inner_positions)):
            ix, iy = inner_positions[i]
            ia = inner_angles[i]
            if not polygon_contains_hexagon(cx, cy, side, angle, ix, iy, inner_side, ia):
                # Penalty for violation
                # Compute max violation
                pen = 0
                vx, vy = hex_vertices(ix, iy, inner_side, ia)
                for v in range(6):
                    if not point_in_hexagon(vx[v], vy[v], cx, cy, side, angle):
                        # Distance outside
                        dx = vx[v] - cx
                        dy = vy[v] - cy
                        angle_rad = np.radians(angle)
                        cos_a = np.cos(-angle_rad)
                        sin_a = np.sin(-angle_rad)
                        lx = cos_a * dx - sin_a * dy
                        ly = sin_a * dx + cos_a * dy
                        apothem = side * np.sqrt(3) / 2
                        for k in range(6):
                            theta = k * np.pi / 3
                            nx = np.cos(theta)
                            ny = np.sin(theta)
                            val = nx * lx + ny * ly
                            if val > apothem:
                                pen += (val - apothem)
                if pen > 0:
                    return side + 100 * pen
        return side
    
    # Initial guess
    # Compute bounding box of all inner hexagon vertices
    all_vx, all_vy = [], []
    for i in range(len(inner_positions)):
        ix, iy = inner_positions[i]
        ia = inner_angles[i]
        vx, vy = hex_vertices(ix, iy, inner_side, ia)
        all_vx.extend(vx)
        all_vy.extend(vy)
    
    all_vx = np.array(all_vx)
    all_vy = np.array(all_vy)
    
    cx0 = (all_vx.min() + all_vx.max()) / 2
    cy0 = (all_vy.min() + all_vy.max()) / 2
    # Rough estimate of side length
    max_dist = np.max(np.sqrt((all_vx - cx0)**2 + (all_vy - cy0)**2))
    side0 = max_dist * 2 / np.sqrt(3) * 1.1  # some margin
    
    best_result = None
    best_side = 1e10
    
    # Try multiple initial angles
    for angle0 in [0, 30, 60, 90, 120, 150]:
        x0 = [cx0, cy0, side0, angle0]
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < best_side:
                if res.fun < best_side:
                    best_side = res.fun
                    best_result = res.x
        except:
            pass
    
    if best_result is None:
        return (cx0, cy0, side0, 0.0), side0
    
    return tuple(best_result), best_side

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    # Place 11 hexagons on a hexagonal lattice
    # Hex lattice basis vectors (60 degrees apart)
    d = np.sqrt(3)  # center-to-center distance for edge-to-edge hexagons
    
    # Hexagonal coordinates to Cartesian
    def hex_to_cart(q, r):
        x = d * (q + r / 2)
        y = d * (np.sqrt(3) / 2) * r
        return x, y
    
    # 11 hexagons: center + 6 in first ring + 4 in second ring
    # Using axial coordinates (q, r)
    positions_axial = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # first ring
        (2, 0), (2, -1), (1, -2), (0, -2),  # 4 from second ring
    ]
    
    inner_positions = []
    for q, r in positions_axial:
        x, y = hex_to_cart(q, r)
        inner_positions.append((x, y))
    
    inner_angles = [0.0] * N
    
    # Convert to list of [x, y, angle]
    inner = [[p[0], p[1], a] for p, a in zip(inner_positions, inner_angles)]
    
    # Compute minimal outer hexagon
    outer_center, outer_side, outer_angle = compute_min_outer_hexagon(
        inner_positions, inner_angles, 1.0
    )
    
    result_side = outer_side
    result = (inner, (outer_center[0], outer_center[1]), outer_side, outer_angle)
    
    # Check time
    if time.time() > deadline - 10:
        return result
    
    # Try alternative arrangements if time permits
    # Try rotating the whole configuration
    best = result
    
    for rot_deg in [30, 60, 90, 120, 150]:
        if time.time() > deadline - 15:
            break
        rot_rad = np.radians(rot_deg)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        
        rotated_positions = []
        for x, y in inner_positions:
            nx = cos_r * x - sin_r * y
            ny = sin_r * x + cos_r * y
            rotated_positions.append((nx, ny))
        
        rotated_angles = [(a + rot_deg) % 360 for a in inner_angles]
        
        oc, os, oa = compute_min_outer_hexagon(rotated_positions, rotated_angles, 1.0)
        if os < result_side:
            inner_rot = [[p[0], p[1], a] for p, a in zip(rotated_positions, rotated_angles)]
            result = (inner_rot, (oc[0], oc[1]), os, oa)
            result_side = os
    
    return result

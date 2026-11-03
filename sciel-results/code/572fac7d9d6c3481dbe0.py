import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, rot_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation in degrees."""
    rot = np.radians(rot_deg)
    verts = []
    for i in range(6):
        angle = rot + np.radians(60 * i)
        x = cx + s * np.cos(angle)
        y = cy + s * np.sin(angle)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, s, rot_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotation rot_deg."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    rot = np.radians(rot_deg)
    # Rotate point by -rot to align hexagon
    cos_r = np.cos(-rot)
    sin_r = np.sin(-rot)
    lx = dx * cos_r - dy * sin_r
    ly = dx * sin_r + dy * cos_r
    # In local frame, hexagon has vertices at angles 0,60,...,300 with radius s
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with "pointy top" at angle 0, the constraints are:
    # |lx| <= s (for the pair of opposite edges at 90 and 270... actually let me think)
    # Actually for a regular hexagon with vertices at angles 0, 60, ..., 300:
    # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330
    # The apothem (distance from center to edge) = s * sqrt(3)/2
    # For edge normal at angle theta, constraint: lx*cos(theta) + ly*sin(theta) <= s*sqrt(3)/2
    apothem = s * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(60 * i + 30)  # edge normals at 30, 90, 150, ...
        if lx * np.cos(theta) + ly * np.sin(theta) > apothem + 1e-9:
            return False
    return True

def all_vertices_inside(inner_hexes, outer_cx, outer_cy, outer_s, outer_rot):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for (ix, iy, irot) in inner_hexes:
        verts = hex_vertices(ix, iy, 1.0, irot)
        for (vx, vy) in verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_s, outer_rot):
                return False
    return True

def initial_cluster():
    """Create a compact initial arrangement of 11 unit hexagons."""
    # Center hexagon
    hexes = [[0.0, 0.0, 0.0]]
    # 6 neighbors at distance 2*sqrt(3)/2 = sqrt(3) ≈ 1.732, at angles 0,60,...,300
    # Actually, for two touching unit hexagons, center distance = 2 * apothem = sqrt(3)
    d = np.sqrt(3)
    for i in range(6):
        angle = np.radians(60 * i)
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        hexes.append([x, y, 0.0])
    # 4 second-ring hexagons: place at positions that fill gaps
    # Second ring positions: at distance 2*sqrt(3) from center, at angles 0, 90, 180, 270 (or similar)
    # Actually, let's place them at the "corners" of the hexagonal arrangement
    # The 6 first-ring hexagons are at angles 0,60,120,180,240,300
    # Second ring can go at angles 30, 90, 150, 210, 270, 330 at distance 2*sqrt(3)
    # But we only need 4, so pick 4 of these 6 positions
    d2 = 2 * np.sqrt(3)
    second_ring_angles = [30, 90, 150, 210, 270, 330]
    # Pick 4 that give a more symmetric/compact shape
    # Let's pick angles 30, 150, 270, 330 (or 90, 210, 330, 30)
    for angle in [30, 150, 270, 330]:
        rad = np.radians(angle)
        x = d2 * np.cos(rad)
        y = d2 * np.sin(rad)
        hexes.append([x, y, 0.0])
    return hexes

def objective(params):
    """Objective function: minimize outer side length with penalty for containment violations."""
    # params: [x0,y0,rot0, x1,y1,rot1, ..., x10,y10,rot10, outer_cx, outer_cy, outer_s, outer_rot]
    inner_hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        rot = params[3*i+2]
        inner_hexes.append([x, y, rot])
    outer_cx = params[3*N]
    outer_cy = params[3*N+1]
    outer_s = params[3*N+2]
    outer_rot = params[3*N+3]
    
    # Penalty for containment violations
    penalty = 0.0
    for (ix, iy, irot) in inner_hexes:
        verts = hex_vertices(ix, iy, 1.0, irot)
        for (vx, vy) in verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            rot_rad = np.radians(outer_rot)
            cos_r = np.cos(-rot_rad)
            sin_r = np.sin(-rot_rad)
            lx = dx * cos_r - dy * sin_r
            ly = dx * sin_r + dy * cos_r
            apothem = outer_s * np.sqrt(3) / 2
            for j in range(6):
                theta = np.radians(60 * j + 30)
                val = lx * np.cos(theta) + ly * np.sin(theta)
                if val > apothem:
                    penalty += (val - apothem) ** 2
    
    return outer_s + 100.0 * penalty

def construct():
    start_time = time.time()
    
    # Create initial cluster
    inner = initial_cluster()
    
    # Initial outer hexagon: large enough to contain everything
    # Compute bounding box of initial cluster
    max_r = 0
    for (x, y, rot) in inner:
        r = np.sqrt(x*x + y*y) + np.sqrt(3)  # max extent of unit hexagon
        max_r = max(max_r, r)
    outer_s_init = max_r / (np.sqrt(3)/2) + 0.5  # apothem = s*sqrt(3)/2, so s = apothem/(sqrt(3)/2)
    outer_cx_init = 0.0
    outer_cy_init = 0.0
    outer_rot_init = 0.0
    
    # Build initial params
    params = []
    for (x, y, rot) in inner:
        params.extend([x, y, rot])
    params.extend([outer_cx_init, outer_cy_init, outer_s_init, outer_rot_init])
    params = np.array(params)
    
    best_params = params.copy()
    best_score = objective(params)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            params,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < best_score:
            best_params = result.x
            best_score = result.fun
        
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return best found
        inner_hexes = []
        for i in range(N):
            inner_hexes.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        outer_cx = best_params[3*N]
        outer_cy = best_params[3*N+1]
        outer_s = max(best_params[3*N+2], 1.0)
        outer_rot = best_params[3*N+3]
        return inner_hexes, (outer_cx, outer_cy), outer_s, outer_rot
    
    # Verify and refine: check if all vertices are inside
    inner_hexes = []
    for i in range(N):
        inner_hexes.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N+1]
    outer_s = max(best_params[3*N+2], 1.0)
    outer_rot = best_params[3*N+3]
    
    # If containment is violated, increase outer_s slightly
    if not all_vertices_inside(inner_hexes, outer_cx, outer_cy, outer_s, outer_rot):
        # Increase outer side until valid
        for _ in range(100):
            outer_s *= 1.01
            if all_vertices_inside(inner_hexes, outer_cx, outer_cy, outer_s, outer_rot):
                break
    
    return inner_hexes, (outer_cx, outer_cy), outer_s, outer_rot

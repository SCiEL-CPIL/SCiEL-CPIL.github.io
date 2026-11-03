"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _min_dist_polygons(v1, v2):
    """Minimum distance between two convex polygons (arrays of vertices)."""
    min_d = np.inf
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        a1 = v1[i]
        a2 = v1[(i+1) % n1]
        for j in range(n2):
            b1 = v2[j]
            b2 = v2[(j+1) % n2]
            # Check if segments intersect (for overlap detection)
            # Also compute min distance between segments
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_d:
                min_d = d
    return min_d

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    elif a < 1e-12:
        t = f / e
        t = np.clip(t, 0, 1)
        return np.linalg.norm(r + d2 * t)
    elif e < 1e-12:
        t = -np.dot(d1, r) / a
        t = np.clip(t, 0, 1)
        return np.linalg.norm(r - d1 * t)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if abs(denom) < 1e-12:
        t = 0.0
        s = np.clip(-np.dot(d1, r) / a, 0, 1)
    else:
        s = np.clip((b * f - c * e) / denom, 0, 1)
        t = (b * s + f) / e
        t = np.clip(t, 0, 1)
        if s < 0:
            s = 0
            t = np.clip(-np.dot(d1, r) / a, 0, 1)
        elif s > 1:
            s = 1
            t = np.clip((b - c) / a, 0, 1)
    
    return np.linalg.norm(r + d1 * t - d2 * s)

def _polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    for verts in [v1, v2]:
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            # Normal to edge
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            
            # Project all vertices onto normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 + tol or max2 < min1 + tol:
                return False  # Separating axis found
    return True  # No separating axis, polygons overlap

def _outer_hex_side(inner_positions, outer_center, outer_angle):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    ocx, ocy = outer_center
    o_angle_rad = np.radians(outer_angle)
    
    # For each inner hexagon, find the maximum "radius" in directions of outer hex sides
    # The outer hexagon with side s has inradius s*sqrt(3)/2
    # A point (px,py) is inside outer hex centered at (ocx,ocy) with side s and angle theta
    # if for each of the 6 face normals, the dot product is <= inradius
    
    max_required_inradius = 0
    for (ix, iy, iangle) in inner_positions:
        verts = _hex_vertices(ix, iy, iangle, 1.0)
        for vx, vy in verts:
            # Relative to outer center
            dx = vx - ocx
            dy = vy - ocy
            # Project onto each of 6 face normal directions
            for k in range(6):
                theta = o_angle_rad + k * np.pi / 3 + np.pi / 6  # face normals
                nx = np.cos(theta)
                ny = np.sin(theta)
                dist = dx * nx + dy * ny
                if dist > max_required_inradius:
                    max_required_inradius = dist
    
    # inradius = side * sqrt(3)/2, so side = inradius / (sqrt(3)/2) = 2*inradius/sqrt(3)
    return 2.0 * max_required_inradius / np.sqrt(3)

def _check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Check if packing is valid (no overlaps, all inside outer hex)."""
    # Check no overlaps between inner hexagons
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            v1 = _hex_vertices(inner_positions[i][0], inner_positions[i][1], inner_positions[i][2])
            v2 = _hex_vertices(inner_positions[j][0], inner_positions[j][1], inner_positions[j][2])
            if _polygons_overlap(v1, v2, tol=-1e-7):
                return False
    # Check all inside outer hex
    ocx, ocy = outer_center
    o_angle_rad = np.radians(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2
    for (ix, iy, iangle) in inner_positions:
        verts = _hex_vertices(ix, iy, iangle, 1.0)
        for vx, vy in verts:
            dx = vx - ocx
            dy = vy - ocy
            for k in range(6):
                theta = o_angle_rad + k * np.pi / 3 + np.pi / 6
                nx = np.cos(theta)
                ny = np.sin(theta)
                dist = dx * nx + dy * ny
                if dist > inradius + 1e-7:
                    return False
    return True

def _objective(params):
    """Objective function: outer hexagon side length."""
    n = N
    # params: 2*n for inner positions (x,y) + n for angles + 2 for outer center + 1 for outer angle
    # Total: 3*n + 3
    inner_x = params[0:n]
    inner_y = params[n:2*n]
    inner_angles = params[2*n:3*n]
    outer_cx = params[3*n]
    outer_cy = params[3*n+1]
    outer_angle = params[3*n+2]
    
    inner_positions = list(zip(inner_x, inner_y, inner_angles))
    outer_center = (outer_cx, outer_cy)
    
    side = _outer_hex_side(inner_positions, outer_center, outer_angle)
    
    # Penalty for overlaps
    penalty = 0
    for i in range(n):
        for j in range(i+1, n):
            v1 = _hex_vertices(inner_x[i], inner_y[i], inner_angles[i])
            v2 = _hex_vertices(inner_x[j], inner_y[j], inner_angles[j])
            if _polygons_overlap(v1, v2, tol=-1e-6):
                # Estimate overlap penalty
                d = np.sqrt((inner_x[i]-inner_x[j])**2 + (inner_y[i]-inner_y[j])**2)
                penalty += max(0, 2.0 - d) * 100  # 2.0 is roughly min center distance
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial arrangement: hexagonal cluster
    # Use axial coordinates for a compact 12-hex arrangement
    # A 3-4-3 or similar arrangement
    
    sqrt3 = np.sqrt(3)
    
    # Hex grid positions (flat-top, side=1)
    # Axial to pixel: x = 1.5*q, y = sqrt3*(r + q/2)
    # For 12 hexagons, use a compact arrangement
    
    # Let's try a 2-3-4-3 arrangement (rows)
    # Or use a more symmetric approach
    
    # Start with a hexagonal lattice cluster
    # Centers for a cluster of 12 hexagons
    # Use axial coords: (q, r)
    # A compact arrangement:
    # Row r=0: q = -1, 0, 1, 2  (4 hexes)
    # Row r=1: q = -1, 0, 1      (3 hexes)  
    # Row r=-1: q = -1, 0, 1     (3 hexes)
    # Row r=2: q = 0, 1          (2 hexes)
    # Total: 4+3+3+2 = 12
    
    axial_coords = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2), (1, 2)
    ]
    
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    init_x = []
    init_y = []
    init_angles = []
    for q, r in axial_coords:
        x, y = axial_to_pixel(q, r)
        init_x.append(x)
        init_y.append(y)
        init_angles.append(0.0)
    
    # Center the arrangement
    cx0 = np.mean(init_x)
    cy0 = np.mean(init_y)
    init_x = [x - cx0 for x in init_x]
    init_y = [y - cy0 for y in init_y]
    
    # Initial outer hexagon
    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle = 0.0
    
    best_side = _outer_hex_side(list(zip(init_x, init_y, init_angles)), (outer_cx, outer_cy), outer_angle)
    best_params = np.array(init_x + init_y + init_angles + [outer_cx, outer_cy, outer_angle])
    
    # Try multiple outer angles
    for oa in [0, 30, 60, 90, 120, 150]:
        s = _outer_hex_side(list(zip(init_x, init_y, init_angles)), (0, 0), oa)
        if s < best_side:
            best_side = s
            best_params = np.array(init_x + init_y + init_angles + [0, 0, oa])
    
    # Local optimization
    for attempt in range(3):
        if time.time() > deadline - 30:
            break
        
        x0 = best_params.copy()
        # Add small perturbation
        if attempt > 0:
            x0 = x0 + np.random.normal(0, 0.1, size=x0.shape)
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_side:
                # Verify validity
                n = N
                rx = result.x
                inner_x = rx[0:n]
                inner_y = rx[n:2*n]
                inner_angles = rx[2*n:3*n]
                ocx, ocy = rx[3*n], rx[3*n+1]
                oa = rx[3*n+2]
                inner_positions = list(zip(inner_x, inner_y, inner_angles))
                side = _outer_hex_side(inner_positions, (ocx, ocy), oa)
                if _check_validity(inner_positions, (ocx, ocy), side, oa):
                    best_side = side
                    best_params = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 15:
            break
    
    # Extract best result
    n = N
    bx = best_params
    inner_x = bx[0:n]
    inner_y = bx[n:2*n]
    inner_angles = bx[2*n:3*n]
    ocx, ocy = bx[3*n], bx[3*n+1]
    oa = bx[3*n+2]
    
    inner = [[float(inner_x[i]), float(inner_y[i]), float(inner_angles[i])] for i in range(n)]
    side = float(best_side)
    
    return inner, (float(ocx), float(ocy)), side, float(oa)
# EVOLVE-BLOCK-END

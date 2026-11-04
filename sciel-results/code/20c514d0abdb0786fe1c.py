"""Packing 12 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # Regular hexagon with flat sides at angle 0: |ly| <= side*sqrt(3)/2 and 
    # for the 6 sides, we need to check all half-planes
    # Actually for a regular hexagon with vertices at angles 0,60,...:
    # The 6 supporting lines are at distance side*sqrt(3)/2 from center
    # Normal directions: 0, 60, 120, 180, 240, 300 degrees
    normals = np.column_stack([np.cos(np.arange(6) * np.pi / 3), 
                                np.sin(np.arange(6) * np.pi / 3)])
    dists = np.abs(normals[:, 0] * lx + normals[:, 1] * ly)
    return np.all(dists <= side * np.sqrt(3) / 2)

def hexagon_overlap(v1, v2):
    """Check if two convex polygons (each 6 vertices) overlap using separating axis theorem."""
    edges = []
    for v in [v1, v2]:
        for i in range(6):
            p1, p2 = v[i], v[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-12:
                normal /= np.linalg.norm(normal)
                edges.append(normal)
    for normal in edges:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False
    return True

def outer_hexagon_contains_point(px, py, ocx, ocy, oside, oangle):
    """Check if point is inside outer hexagon."""
    return point_in_hexagon(px, py, ocx, ocy, oside, oangle)

def build_initial_config():
    """Build initial hexagonal lattice arrangement of 12 unit hexagons."""
    # Honeycomb pattern: rows with offsets
    # Try: 4 rows of 3, with alternating offsets
    # Horizontal spacing: sqrt(3), vertical: 1.5, row offset: sqrt(3)/2
    s3 = np.sqrt(3)
    positions = []
    # Row 0: 3 hexagons
    for i in range(3):
        positions.append((i * s3, 0.0))
    # Row 1: 3 hexagons, offset
    for i in range(3):
        positions.append((i * s3 + s3/2, 1.5))
    # Row 2: 3 hexagons
    for i in range(3):
        positions.append((i * s3, 3.0))
    # Row 3: 3 hexagons, offset
    for i in range(3):
        positions.append((i * s3 + s3/2, 4.5))
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy) for x, y in positions]
    
    return positions

def compute_min_outer_side(positions, angles, ocx, ocy, oangle):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    # For each inner hexagon, find the maximum "distance" from outer center in each of 6 directions
    # The outer hexagon with center (ocx, ocy), side s, angle a contains point p iff
    # for all 6 normal directions n_k: n_k . (p - c) <= s * sqrt(3)/2
    # So s >= max over all points p and all 6 directions of n_k . (p - c) / (sqrt(3)/2)
    
    normals = np.column_stack([np.cos(oangle + k * np.pi/3), 
                                np.sin(oangle + k * np.pi/3)] for k in range(6))
    
    max_val = 0
    for (px, py), ang in zip(positions, angles):
        verts = hex_vertices(px, py, 1.0, ang)
        for v in verts:
            rel = v - np.array([ocx, ocy])
            for n in normals:
                d = np.abs(np.dot(n, rel))
                if d > max_val:
                    max_val = d
    
    return max_val / (np.sqrt(3) / 2)

def objective(params, positions_init, angles_init):
    """Objective: minimize outer side length.
    params: [ocx, ocy, oangle, x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    """
    t = time.time()
    if t > DEADLINE:
        return 1e10
    
    ocx, ocy, oangle = params[0], params[1], params[2]
    inner_params = params[3:]
    
    positions = []
    angles = []
    for i in range(N):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        positions.append((x, y))
        angles.append(a)
    
    # Compute required outer side
    s = compute_min_outer_side(positions, angles, ocx, ocy, oangle)
    
    # Check non-overlap constraints (soft penalty)
    penalty = 0
    for i in range(N):
        v1 = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for j in range(i+1, N):
            v2 = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if hexagon_overlap(v1, v2):
                # Compute minimum separation needed
                penalty += 0.1
                if penalty > 10:
                    return s + penalty
    
    return s + penalty

def construct():
    start_time = time.time()
    
    # Build initial configuration
    positions = build_initial_config()
    angles = [0.0] * N
    
    # Initial outer hexagon parameters
    # Estimate bounding box
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ox = (min(xs) + max(xs)) / 2
    oy = (min(ys) + max(ys)) / 2
    # Rough estimate of side length
    max_x = max(xs) - min(xs)
    max_y = max(ys) - min(ys)
    init_side = max(max_x, max_y) / np.sqrt(3) + 1.0
    
    # Parameters: [ocx, ocy, oangle, x1, y1, a1, ..., x12, y12, a12]
    x0 = [ox, oy, 0.0]
    for p, a in zip(positions, angles):
        x0.extend([p[0], p[1], a])
    
    # Try optimization
    best_result = None
    best_side = compute_min_outer_side(positions, angles, ox, oy, 0.0)
    
    # Run SLSQP optimization
    try:
        result = minimize(
            objective, x0, 
            args=(positions, angles),
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.success or result.fun < best_side:
            params = result.x
            ocx, ocy, oangle = params[0], params[1], params[2]
            inner_params = params[3:]
            new_positions = []
            new_angles = []
            for i in range(N):
                x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
                new_positions.append((x, y))
                new_angles.append(a)
            
            side = compute_min_outer_side(new_positions, new_angles, ocx, ocy, oangle)
            
            # Verify no overlaps
            valid = True
            for i in range(N):
                v1 = hex_vertices(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i])
                for j in range(i+1, N):
                    v2 = hex_vertices(new_positions[j][0], new_positions[j][1], 1.0, new_angles[j])
                    if hexagon_overlap(v1, v2):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and side < best_side:
                best_positions = new_positions
                best_angles = new_angles
                best_ocx, best_ocy, best_oangle = ocx, ocy, oangle
                best_side = side
        else:
            best_positions = positions
            best_angles = angles
            best_ocx, best_ocy, best_oangle = ox, oy, 0.0
    except Exception:
        best_positions = positions
        best_angles = angles
        best_ocx, best_ocy, best_oangle = ox, oy, 0.0
    
    # Build output
    inner = []
    for p, a in zip(best_positions, best_angles):
        inner.append([p[0], p[1], np.degrees(a)])
    
    return inner, (best_ocx, best_ocy), best_side, np.degrees(best_oangle)


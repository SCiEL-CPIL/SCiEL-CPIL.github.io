"""Packing unit hexagons in a regular hexagon via multi-start Nelder-Mead with smooth objective."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, angle):
    """Vertices of regular hexagon centered at (cx,cy), circumradius r, rotation angle (rad)."""
    verts = np.zeros((6, 2))
    for i in range(6):
        a = angle + i * math.pi / 3.0
        verts[i, 0] = cx + r * math.cos(a)
        verts[i, 1] = cy + r * math.sin(a)
    return verts

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap (using SAT or edge intersection)."""
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        for j in range(len(v2)):
            p3 = v2[j]
            p4 = v2[(j + 1) % len(v2)]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    if _point_in_polygon(v1[0], v2) or _point_in_polygon(v2[0], v1):
        return True
    return False

def _point_in_polygon(pt, poly):
    """Ray casting point in polygon."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _hex_in_hex(v_inner, v_outer):
    """Check if inner hexagon is fully inside outer hexagon."""
    for pt in v_inner:
        if not _point_in_polygon(pt, v_outer):
            return False
    return True

def _min_overlap_penalty(positions, angles, outer_center, outer_side, outer_angle):
    """Compute penalty for overlaps and containment violations."""
    R_outer = outer_side
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], R_outer, outer_angle)
    penalty = 0.0
    
    for i in range(len(positions)):
        x, y = positions[i]
        verts = _hex_vertices(x, y, 1.0, angles[i])
        if not _hex_in_hex(verts, outer_verts):
            for pt in verts:
                if not _point_in_polygon(pt, outer_verts):
                    penalty += 10.0
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if _polygons_overlap(v1, v2):
                penalty += 10.0
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = float('inf')
    best_x = None
    
    # Objective function for Nelder-Mead
    # x = [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, cx, cy, outer_angle]
    # Total: 3*N + 3 = 36 variables
    
    def objective(x):
        positions = []
        angles = []
        for i in range(N):
            positions.append((x[3*i], x[3*i+1]))
            angles.append(x[3*i+2])
        center = (x[3*N], x[3*N+1])
        outer_angle = x[3*N+2]
        
        # Compute enclosing side
        min_side = 0.0
        for i in range(N):
            verts = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for pt in verts:
                rx = pt[0] - center[0]
                ry = pt[1] - center[1]
                for k in range(6):
                    a = outer_angle + k * math.pi / 3.0
                    proj = rx * math.cos(a) + ry * math.sin(a)
                    required_R = proj / (math.sqrt(3) / 2.0)
                    if required_R > min_side:
                        min_side = required_R
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                if _polygons_overlap(v1, v2):
                    penalty += 1.0
        
        return max(min_side, 1.0) + penalty * 5.0
    
    # Multiple starts
    n_starts = 20
    start_interval = (deadline - start_time) / max(n_starts, 1)
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Create initial layout
        if s == 0:
            # Hexagonal lattice
            positions = []
            angles = []
            dx = 2.0
            dy = 2.0 * math.sqrt(3) / 2.0
            n_per_row = int(math.ceil(math.sqrt(N * 2 / math.sqrt(3))))
            for i in range(N):
                row = i // n_per_row
                col = i % n_per_row
                x = col * dx + (row % 2) * dx / 2.0
                y = row * dy
                positions.append((x, y))
                angles.append(0.0)
            cx = sum(p[0] for p in positions) / N
            cy = sum(p[1] for p in positions) / N
            positions = [(p[0] - cx, p[1] - cy) for p in positions]
        else:
            # Random perturbation of hexagonal lattice
            positions = []
            angles = []
            dx = 2.0
            dy = 2.0 * math.sqrt(3) / 2.0
            n_per_row = int(math.ceil(math.sqrt(N * 2 / math.sqrt(3))))
            for i in range(N):
                row = i // n_per_row
                col = i % n_per_row
                x = col * dx + (row % 2) * dx / 2.0 + random.gauss(0, 0.5)
                y = row * dy + random.gauss(0, 0.5)
                positions.append((x, y))
                angles.append(random.gauss(0, 0.3))
            cx = sum(p[0] for p in positions) / N
            cy = sum(p[1] for p in positions) / N
            positions = [(p[0] - cx, p[1] - cy) for p in positions]
        
        # Build initial x vector
        x0 = []
        for i in range(N):
            x0.extend([positions[i][0], positions[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])  # center and outer_angle
        
        # Run Nelder-Mead
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.fun < best_score:
            best_score = result.fun
            best_x = result.x
    
    # Extract best solution
    if best_x is not None:
        positions = []
        angles = []
        for i in range(N):
            positions.append((best_x[3*i], best_x[3*i+1]))
            angles.append(best_x[3*i+2])
        center = (best_x[3*N], best_x[3*N+1])
        outer_angle = best_x[3*N+2]
        
        # Compute final enclosing side
        min_side = 0.0
        for i in range(N):
            verts = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for pt in verts:
                rx = pt[0] - center[0]
                ry = pt[1] - center[1]
                for k in range(6):
                    a = outer_angle + k * math.pi / 3.0
                    proj = rx * math.cos(a) + ry * math.sin(a)
                    required_R = proj / (math.sqrt(3) / 2.0)
                    if required_R > min_side:
                        min_side = required_R
        
        # Verify no overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                if _polygons_overlap(v1, v2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            inner = [[positions[i][0], positions[i][1], math.degrees(angles[i])] for i in range(N)]
            return inner, center, max(min_side, 1.0), math.degrees(outer_angle)
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END

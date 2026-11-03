"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses multi-start local search with COBYLA and hard overlap penalties.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def overlap_depth(A, B):
    """
    Compute the minimum translation depth to separate two convex polygons.
    Returns 0 if they are separated or touching.
    Returns positive value if they overlap (depth of overlap).
    """
    min_depth = 0.0
    for poly in (A, B):
        m = len(poly)
        for k in range(m):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % m]
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-12:
                continue
            nx, ny = -dy / length, dx / length
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            
            # Check for separation
            if maxA < minB or maxB < minA:
                return 0.0 # Separated on this axis, so separated overall
            
            # Overlap on this axis
            # The amount of overlap on this axis is:
            # min(maxA, maxB) - max(minA, minB)
            depth = min(maxA, maxB) - max(minA, minB)
            if depth < min_depth:
                min_depth = depth
                
    return min_depth


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            # Point in convex polygon check
            n = len(outer)
            sign = 0
            inside = True
            for i in range(n):
                x1, y1 = outer[i]
                x2, y2 = outer[(i + 1) % n]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if abs(cross) < 1e-9:
                    continue
                s = 1 if cross > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    inside = False
                    break
            if not inside:
                return False
    # pairwise non-overlap
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if overlap_depth(polys[i], polys[j]) > 1e-7:
                return False
    return True


def objective_coby(params, outer_center=(0.0, 0.0)):
    """
    Objective for COBYLA optimization.
    params: 36-dim vector [x0,y0,a0, ..., x11,y11,a11] + [outer_angle]
    Actually, let's include outer_angle in params to optimize it too.
    Total 37 params.
    """
    if time.time() > DEADLINE:
        return 1e6
    
    inner_params = params[:36]
    outer_angle = params[36]
    
    inner = []
    for i in range(N):
        x = inner_params[i * 3]
        y = inner_params[i * 3 + 1]
        ang = inner_params[i * 3 + 2]
        inner.append((x, y, ang))
    
    # Compute minimum outer side needed
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Compute overlap penalty
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = overlap_depth(polys[i], polys[j])
            if d > 0:
                penalty += d * d
    
    return outer_side + 100.0 * penalty


def build_initial_config(type_id):
    """Generate initial configurations."""
    configs = []
    
    if type_id == 0:
        # Hexagonal lattice 1+6+5
        a = 2.0
        positions = [(0.0, 0.0)]
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(5):
            theta = math.radians(k * 60.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
        inner = [(x, y, 0.0) for (x, y) in positions]
        configs.append(inner)
        
    elif type_id == 1:
        # Hexagonal lattice 1+6+6 (full ring 2)
        a = 2.0
        positions = [(0.0, 0.0)]
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
        inner = [(x, y, 0.0) for (x, y) in positions]
        configs.append(inner)
        
    elif type_id == 2:
        # Rectangular 3x4
        a = 2.0
        row_spacing = a * math.sqrt(3) / 2.0
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * a
                y = row * row_spacing
                if row % 2 == 1:
                    x += a / 2.0
                positions.append((x - 1.5 * a, y - row_spacing))
        inner = [(x, y, 0.0) for (x, y) in positions]
        configs.append(inner)
        
    elif type_id == 3:
        # Random
        np.random.seed(42 + type_id)
        inner = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            ang = np.random.uniform(-15, 15)
            inner.append((x, y, ang))
        configs.append(inner)
        
    return configs


def construct():
    t0 = time.time()
    best_side = 1e6
    best_result = None
    
    # Define bounds for COBYLA
    # x, y in [-5, 5], angle in [-60, 60], outer_angle in [-30, 30]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((-60.0, 60.0))
    bounds.append((-30.0, 30.0))
    
    # Try multiple initial configurations
    for init_type in range(4):
        if time.time() > DEADLINE - 20:
            break
            
        configs = build_initial_config(init_type)
        for inner in configs:
            if time.time() > DEADLINE - 10:
                break
                
            # Flatten inner to params
            params = []
            for (x, y, ang) in inner:
                params.extend([x, y, ang])
            params.append(0.0) # outer angle
            
            try:
                res = minimize(
                    objective_coby,
                    params,
                    method='COBYLA',
                    options={'maxiter': 1000, 'rhobeg': 1.0, 'catol': 1e-6}
                )
                
                if res.fun < best_side and res.fun < 1e5:
                    # Check validity
                    res_params = res.x
                    inner_res = []
                    for i in range(N):
                        inner_res.append((res_params[i*3], res_params[i*3+1], res_params[i*3+2]))
                    outer_angle = res_params[36]
                    outer_side = compute_min_outer_side(inner_res, (0.0, 0.0), outer_angle)
                    
                    if check_validity(inner_res, (0.0, 0.0), outer_side, outer_angle):
                        best_side = outer_side
                        best_result = (inner_res, (0.0, 0.0), outer_side, outer_angle)
                        
            except Exception as e:
                pass
                
    if best_result is not None:
        return best_result
    
    # Fallback
    a = 2.0
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    inner = [(x, y, 0.0) for (x, y) in positions]
    outer_side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
    return inner, (0.0, 0.0), outer_side, 0.0

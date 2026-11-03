"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses free-placement optimization with smooth penalties and multiple seeds.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    return [(cx + s * math.cos(a + i * math.pi / 3.0), cy + s * math.sin(a + i * math.pi / 3.0)) for i in range(6)]

def point_in_convex_polygon(px, py, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = [(math.cos(a + (i + 0.5) * math.pi / 3.0), math.sin(a + (i + 0.5) * math.pi / 3.0)) for i in range(6)]
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

def objective_free(params):
    """
    params: array of 36 floats [x0, y0, th0, x1, y1, th1, ...]
    Returns: outer_side + penalty
    """
    inner = [(params[i], params[i+1], params[i+2]) for i in range(0, 36, 3)]
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute min outer side
    side = compute_min_outer_side(inner, center, outer_angle)
    
    # Compute overlap penalty
    # We use a smooth approximation of the SAT violation.
    # For speed, we only check a subset of axes or use a simplified metric.
    # However, for Nelder-Mead, a hard penalty is often fine if the landscape is smooth enough.
    # Let's use a soft penalty based on center distance and orientation difference? 
    # No, that's not accurate. 
    # Let's just do the SAT check but it's expensive. 
    # Alternative: use a pre-computed list of critical pairs? No, topology changes.
    
    # To make this fast, we will use a "soft" SAT where we project onto the 6 axes of hex A and 6 of hex B.
    # If any axis separates, no penalty. If all overlap, penalty is the minimum overlap depth.
    penalty = 0.0
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    
    for i in range(N):
        for j in range(i + 1, N):
            A = polys[i]
            B = polys[j]
            min_overlap = float('inf')
            for poly in (A, B):
                for k in range(6):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k + 1) % 6]
                    dx, dy = x2 - x1, y2 - y1
                    nx, ny = -dy, dx
                    norm = math.sqrt(nx*nx + ny*ny)
                    if norm == 0: continue
                    nx, ny = nx/norm, ny/norm
                    minA = min(nx * px + ny * py for px, py in A)
                    maxA = max(nx * px + ny * py for px, py in A)
                    minB = min(nx * px + ny * py for px, py in B)
                    maxB = max(nx * px + ny * py for px, py in B)
                    if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                        min_overlap = 0.0
                        break
                    else:
                        overlap = min(maxA, maxB) - max(minA, minB)
                        if overlap < min_overlap:
                            min_overlap = overlap
                if min_overlap == 0.0:
                    break
            if min_overlap > 0:
                penalty += min_overlap * min_overlap
                # Early exit if penalty is huge
                if penalty > 10.0:
                    return side + penalty
    
    # Containment penalty
    outer = hex_vertices(0.0, 0.0, side, 0.0)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                # Estimate penetration depth
                # Simplified: distance to closest edge
                # For now, just add a large penalty
                penalty += 1.0
                if penalty > 10.0:
                    return side + penalty
    
    return side + penalty

def generate_seed_honeycomb():
    """Generate a honeycomb-like seed."""
    params = []
    # Center
    params.extend([0.0, 0.0, 0.0])
    # Ring 1 (6 hexes)
    for k in range(6):
        theta = math.radians(30 + k * 60)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        params.extend([x, y, 0.0])
    # Ring 2 (5 hexes, leaving a gap)
    for k in range(5):
        theta = math.radians(60 + k * 60) # Offset
        x = 3.8 * math.cos(theta)
        y = 3.8 * math.sin(theta)
        params.extend([x, y, 0.0])
    return np.array(params)

def generate_seed_ring12():
    """Generate a ring of 12."""
    params = []
    for k in range(12):
        theta = math.radians(k * 30)
        x = 2.5 * math.cos(theta)
        y = 2.5 * math.sin(theta)
        params.extend([x, y, 0.0])
    return np.array(params)

def generate_seed_grid():
    """Generate a 3x4 grid."""
    params = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * 2.0
            y = (j - 1.5) * 2.0
            params.extend([x, y, 0.0])
    return np.array(params)

def optimize():
    best_side = 40.0
    best_params = None
    best_valid = False
    
    seeds = [generate_seed_honeycomb(), generate_seed_ring12(), generate_seed_grid()]
    
    for seed in seeds:
        if time.time() > DEADLINE - 10.0:
            break
        
        try:
            # Use Nelder-Mead
            result = minimize(objective_free, seed, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
            
            if result.fun < best_side:
                # Verify validity
                params = result.x
                inner = [(params[i], params[i+1], params[i+2]) for i in range(0, 36, 3)]
                side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
                if check_validity(inner, (0.0, 0.0), side, 0.0):
                    best_side = side
                    best_params = params
                    best_valid = True
        except Exception:
            pass
            
    return best_side, best_params, best_valid

def construct():
    t0 = time.time()
    try:
        side, params, valid = optimize()
        if valid and params is not None:
            inner = [(params[i], params[i+1], params[i+2]) for i in range(0, 36, 3)]
            return inner, (0.0, 0.0), side, 0.0
    except Exception:
        pass

    # Fallback: linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

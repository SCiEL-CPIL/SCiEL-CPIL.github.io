"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses direct geometric optimization with Nelder-Mead on 36 parameters
(12 positions + 12 rotations), with multiple restarts from diverse
initial configurations and a smooth penalty for constraint violations.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize
import random

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill
_EPS = 1e-9


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def hex_vertices_array(cx, cy, s, angle_deg):
    """Vectorized hex vertices as numpy array (6,2)."""
    a = math.radians(angle_deg)
    thetas = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(thetas), cy + s * np.sin(thetas)])


def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
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
            if maxA < minB - _EPS or maxB < minA - _EPS:
                return False
    return True


def sat_gap(A, B):
    """Return minimum separation gap between two convex polygons.
    Positive gap means separated; <= 0 means overlapping/touching."""
    min_gap = float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                continue
            nx, ny = nx / norm, ny / norm
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            gap = min(minA - maxB, minB - maxA)
            if gap > 0:
                return gap  # found separating axis, they are disjoint
            min_gap = min(min_gap, gap)
    return min_gap


def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon (CCW or CW)."""
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
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def params_to_inner(params):
    """Convert 36 params (12 x, 12 y, 12 angles) to inner list."""
    inner = []
    for i in range(N):
        x = params[3 * i]
        y = params[3 * i + 1]
        ang = params[3 * i + 2]
        inner.append((x, y, ang))
    return inner


def objective(params, outer_angle=0.0, outer_center=(0.0, 0.0)):
    """Objective: outer side length + penalties for overlaps and containment."""
    if time.time() > DEADLINE:
        return 1e6
    
    inner = params_to_inner(params)
    
    # Compute outer side needed
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Compute overlap penalty using SAT gap
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0:
                overlap_penalty += (-gap) ** 2
    
    # Containment check: points outside the hexagon of side outer_side
    outer_poly = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    contain_penalty = 0.0
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer_poly):
                # Distance to nearest edge (approximate)
                min_d = float('inf')
                for k in range(6):
                    x1, y1 = outer_poly[k]
                    x2, y2 = outer_poly[(k + 1) % 6]
                    # Point-to-segment distance
                    dx, dy = x2 - x1, y2 - y1
                    len2 = dx * dx + dy * dy
                    if len2 < 1e-12:
                        d = math.sqrt((px - x1)**2 + (py - y1)**2)
                    else:
                        t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / len2))
                        proj_x = x1 + t * dx
                        proj_y = y1 + t * dy
                        d = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
                    min_d = min(min_d, d)
                contain_penalty += min_d ** 2
                break  # only count once per polygon
    
    total = outer_side + 50.0 * overlap_penalty + 20.0 * contain_penalty
    return total


def make_initial_config(strategy, rng):
    """Generate initial configuration based on strategy.
    Returns 36-element param vector."""
    params = np.zeros(36)
    
    if strategy == 0:
        # Hexagonal lattice: center + ring of 6 + partial outer ring
        # Center at origin
        params[0:3] = [0.0, 0.0, 0.0]
        # Ring of 6 at radius ~2
        for k in range(6):
            theta = math.radians(k * 60.0)
            i = k + 1
            params[3*i:3*i+3] = [2.0 * math.cos(theta), 2.0 * math.sin(theta), 0.0]
        # 5 more in outer ring
        for k in range(5):
            theta = math.radians(k * 72.0 + 30.0)
            i = k + 7
            params[3*i:3*i+3] = [4.0 * math.cos(theta), 4.0 * math.sin(theta), 0.0]
            
    elif strategy == 1:
        # Ring of 12
        for k in range(12):
            theta = math.radians(k * 30.0)
            i = k
            params[3*i:3*i+3] = [2.5 * math.cos(theta), 2.5 * math.sin(theta), 0.0]
            
    elif strategy == 2:
        # Two rows of 6
        for k in range(6):
            i = k
            params[3*i:3*i+3] = [(k - 2.5) * 2.1, -1.0, 0.0]
            j = k + 6
            params[3*j:3*j+3] = [(k - 2.5) * 2.1, 1.0, 0.0]
            
    elif strategy == 3:
        # Random scatter
        for i in range(N):
            r = rng.uniform(0.5, 3.5)
            theta = rng.uniform(0, 2 * math.pi)
            params[3*i] = r * math.cos(theta)
            params[3*i+1] = r * math.sin(theta)
            params[3*i+2] = rng.uniform(0, 30)
            
    else:
        # Staggered grid
        for i in range(N):
            row = i // 4
            col = i % 4
            x = (col - 1.5) * 2.1
            y = (row - 1.0) * 1.9
            if row % 2 == 1:
                x += 1.05
            params[3*i] = x
            params[3*i+1] = y
            params[3*i+2] = 0.0
    
    return params


def optimize_packing():
    """Main optimization: try multiple initial configs and outer orientations."""
    best_side = 1e6
    best_result = None
    rng = random.Random(42)
    
    # Try different outer orientations
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    strategies = [0, 1, 2, 3, 4]
    
    for oa in outer_angles:
        for strat in strategies:
            if time.time() > DEADLINE - 5:
                break
            
            x0 = make_initial_config(strat, rng)
            
            try:
                result = minimize(
                    objective, x0, 
                    args=(oa,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6,
                             'maxfev': 1500, 'adaptive': True}
                )
                
                if result.fun < 1e5:
                    # Extract and validate
                    inner = params_to_inner(result.x)
                    oc = (0.0, 0.0)
                    side = compute_min_outer_side(inner, oc, oa)
                    
                    if check_validity(inner, oc, side, oa) and side < best_side:
                        best_side = side
                        best_result = (inner, oc, side, oa)
                        
            except Exception:
                pass
    
    # Second pass: refine best result with more iterations if time allows
    if best_result is not None and time.time() < DEADLINE - 10:
        inner, oc, side, oa = best_result
        x0 = np.array([inner[i][0] for i in range(N)] + 
                      [inner[i][1] for i in range(N)] + 
                      [inner[i][2] for i in range(N)])
        
        try:
            result = minimize(
                objective, x0,
                args=(oa,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8,
                         'maxfev': 3000, 'adaptive': True}
            )
            
            if result.fun < 1e5:
                inner2 = params_to_inner(result.x)
                side2 = compute_min_outer_side(inner2, oc, oa)
                if check_validity(inner2, oc, side2, oa) and side2 < best_side:
                    best_side = side2
                    best_result = (inner2, oc, side2, oa)
        except Exception:
            pass
    
    return best_side, best_result


def construct():
    t0 = time.time()
    
    # Try direct geometric optimization
    try:
        best_side, best_result = optimize_packing()
        if best_result is not None and best_side < 1e5:
            return best_result
    except Exception:
        pass
    
    # Fallback: simple linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

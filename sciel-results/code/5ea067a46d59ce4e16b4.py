"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a topological-graph / contact-constraint approach with symmetry exploitation
and local refinement via least-squares on contact constraints.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, least_squares

N = 12
DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def polygons_overlap_sat(A, B, tol=1e-9):
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
            if maxA < minB - tol or maxB < minA - tol:
                return False
    return True


def point_in_convex_polygon(px, py, poly, tol=1e-9):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < tol:
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
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # The inradius (distance from center to edge midpoint) is s*sqrt(3)/2
    # So for a point at distance d from center along a normal, we need d <= s*sqrt(3)/2
    # The 6 outward normals of the outer hexagon:
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
    # apothem = s * sqrt(3)/2, so s = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def distance_between_hexagons(p1, p2, a1, a2):
    """Compute minimum distance between two hexagons (0 if touching/overlapping).
    Returns the negative of penetration depth if overlapping."""
    poly1 = hex_vertices(p1[0], p1[1], 1.0, a1)
    poly2 = hex_vertices(p2[0], p2[1], 1.0, a2)
    # Use SAT to find minimum separation
    min_sep = float('inf')
    for poly in (poly1, poly2):
        for k in range(6):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % 6]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-10:
                continue
            nx, ny = nx / norm, ny / norm
            proj1 = [nx * px + ny * py for px, py in poly1]
            proj2 = [nx * px + ny * py for px, py in poly2]
            sep = min(proj2) - max(proj1)
            if sep < min_sep:
                min_sep = sep
            sep2 = min(proj1) - max(proj2)
            if sep2 < min_sep:
                min_sep = sep2
    return min_sep  # negative means overlap


def build_symmetric_config(symmetry_order, outer_angle=0.0):
    """Build a symmetric configuration.
    symmetry_order: 1=none, 2=6-fold (center+ring), 3=3-fold, 4=2-fold
    
    Returns list of (x, y, angle) for 12 hexagons.
    """
    if symmetry_order == 6:
        # Center + 6 in ring1 + 5 in ring2 (asymmetric)
        # Better: use 6-fold with center + 6 at r1 + 5 at r2
        positions = [(0.0, 0.0)]
        r1 = 2.0
        for k in range(6):
            theta = math.radians(k * 60.0)
            positions.append((r1 * math.cos(theta), r1 * math.sin(theta)))
        r2 = 4.0
        for k in range(5):
            theta = math.radians(k * 60.0)
            positions.append((r2 * math.cos(theta), r2 * math.sin(theta)))
        angles = [0.0] * 12
        return [(x, y, a) for (x, y), a in zip(positions, angles)]
    elif symmetry_order == 3:
        # 3-fold symmetry: 4 hexagons in each sector
        # Place 4 hexagons at angles 0, 60, 120 degrees from center, replicated
        positions = []
        # Base positions in one 120-degree sector
        base = [
            (1.5, 0.0),
            (3.0, 0.0),
            (1.5, 2.6),   # approx 1.5*sqrt(3)
            (3.0, 2.6),
        ]
        for rot in range(3):
            base_angle = math.radians(rot * 120.0)
            for (x, y) in base:
                xr = x * math.cos(base_angle) - y * math.sin(base_angle)
                yr = x * math.sin(base_angle) + y * math.cos(base_angle)
                positions.append((xr, yr))
        angles = [0.0] * 12
        return [(x, y, a) for (x, y), a in zip(positions, angles)]
    elif symmetry_order == 2:
        # 2-fold symmetry: 6 hexagons in each half
        positions = []
        base = [
            (1.0, 0.0),
            (2.5, 0.0),
            (4.0, 0.0),
            (1.0, 1.7),
            (2.5, 1.7),
            (4.0, 1.7),
        ]
        for rot in range(2):
            base_angle = math.radians(rot * 180.0)
            for (x, y) in base:
                xr = x * math.cos(base_angle) - y * math.sin(base_angle)
                yr = x * math.sin(base_angle) + y * math.cos(base_angle)
                positions.append((xr, yr))
        angles = [0.0] * 12
        return [(x, y, a) for (x, y), a in zip(positions, angles)]
    else:
        # No symmetry: use a compact arrangement
        # 3x4 grid with hexagonal offset
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * 2.0 + (row % 2) * 1.0
                y = row * 1.732  # sqrt(3)
                positions.append((x - 3.0, y - 1.732))
        angles = [0.0] * 12
        return [(x, y, a) for (x, y), a in zip(positions, angles)]


def objective_free(params, outer_angle=0.0):
    """Objective for free placement: outer_side + penalties."""
    if time.time() > DEADLINE:
        return 1e6
    
    inner = []
    for i in range(N):
        x = params[3 * i]
        y = params[3 * i + 1]
        ang = params[3 * i + 2]
        inner.append((x, y, ang))
    
    outer_center = (0.0, 0.0)  # Fix outer center at origin for symmetry
    
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Overlap penalty
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sep = distance_between_hexagons(inner[i], inner[j], inner[i][2], inner[j][2])
            if sep < 0:
                overlap_pen += sep * sep  # quadratic penalty
    
    return outer_side + 50.0 * overlap_pen


def optimize_symmetric(symmetry_order, outer_angle=0.0, max_iter=200):
    """Optimize a symmetric configuration."""
    initial = build_symmetric_config(symmetry_order, outer_angle)
    
    params = []
    for (x, y, ang) in initial:
        params.extend([x, y, ang])
    
    best_side = 1e6
    best_params = params.copy()
    
    try:
        result = minimize(objective_free, params, args=(outer_angle,),
                         method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5,
                                  'adaptive': True})
        if result.fun < best_side and result.fun < 100:
            best_side = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    # Extract result
    inner = []
    for i in range(N):
        inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
    
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    if check_validity(inner, outer_center, outer_side, outer_angle):
        return inner, outer_center, outer_side, outer_angle
    
    return None


def optimize_lattice_scale():
    """Try various lattice scales and rotations."""
    best_side = 1e6
    best_result = None
    
    for scale in np.linspace(0.95, 1.05, 21):
        for rot_in in [0.0, 30.0]:
            for rot_out in [0.0, 30.0]:
                if time.time() > DEADLINE - 15:
                    break
                
                # Build hexagonal cluster with given scale
                positions = [(0.0, 0.0)]
                r1 = 2.0 * scale
                for k in range(6):
                    theta = math.radians(k * 60.0)
                    positions.append((r1 * math.cos(theta), r1 * math.sin(theta)))
                r2 = 4.0 * scale
                for k in range(5):
                    theta = math.radians(k * 60.0)
                    positions.append((r2 * math.cos(theta), r2 * math.sin(theta)))
                
                inner = [(x, y, rot_in) for (x, y) in positions]
                outer_center = (0.0, 0.0)
                outer_side = compute_min_outer_side(inner, outer_center, rot_out)
                
                if check_validity(inner, outer_center, outer_side, rot_out):
                    if outer_side < best_side:
                        best_side = outer_side
                        best_result = (inner, outer_center, outer_side, rot_out)
    
    return best_result


def optimize_rect_grid():
    """Try rectangular grid arrangements."""
    best_side = 1e6
    best_result = None
    
    # 3x4 grid
    for dx in np.linspace(1.9, 2.1, 11):
        for dy in np.linspace(1.6, 1.8, 11):
            for rot in [0.0, 30.0]:
                if time.time() > DEADLINE - 10:
                    break
                
                positions = []
                for row in range(3):
                    for col in range(4):
                        x = col * dx + (row % 2) * dx * 0.5
                        y = row * dy
                        positions.append((x - 1.5*dx, y - dy))
                
                inner = [(x, y, rot) for (x, y) in positions]
                outer_center = (0.0, 0.0)
                outer_side = compute_min_outer_side(inner, outer_center, rot)
                
                if check_validity(inner, outer_center, outer_side, rot):
                    if outer_side < best_side:
                        best_side = outer_side
                        best_result = (inner, outer_center, outer_side, rot)
    
    return best_result


def refine_with_least_squares(inner, outer_center, outer_side, outer_angle):
    """Refine using least-squares on contact constraints."""
    if time.time() > DEADLINE - 20:
        return inner, outer_center, outer_side, outer_angle
    
    # Try to improve by adjusting positions slightly
    params = []
    for (x, y, ang) in inner:
        params.extend([x, y, ang])
    
    def residuals(p):
        if time.time() > DEADLINE:
            return [1e3] * 36
        res = []
        for i in range(N):
            x, y, ang = p[3*i], p[3*i+1], p[3*i+2]
            # Containment: push inward if outside
            poly = hex_vertices(x, y, 1.0, ang)
            outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
            for (px, py) in poly:
                if not point_in_convex_polygon(px, py, outer):
                    res.append(1.0)
                else:
                    res.append(0.0)
            # Non-overlap with others
            for j in range(i+1, N):
                xj, yj, angj = p[3*j], p[3*j+1], p[3*j+2]
                sep = distance_between_hexagons((x,y), (xj,yj), ang, angj)
                if sep < 0:
                    res.append(sep)
                else:
                    res.append(0.0)
        return res
    
    try:
        result = least_squares(residuals, params, method='lm', max_nfev=200)
        new_inner = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
        new_side = compute_min_outer_side(new_inner, outer_center, outer_angle)
        if check_validity(new_inner, outer_center, new_side, outer_angle) and new_side < outer_side:
            return new_inner, outer_center, new_side, outer_angle
    except Exception:
        pass
    
    return inner, outer_center, outer_side, outer_angle


def construct():
    t0 = time.time()
    best_result = None
    best_side = 1e6
    
    # Strategy 1: Symmetric optimization with different symmetries
    for sym in [6, 3, 2, 1]:
        if time.time() > DEADLINE - 40:
            break
        result = optimize_symmetric(sym, outer_angle=0.0, max_iter=150)
        if result is not None:
            inner, oc, side, oa = result
            if side < best_side:
                best_side = side
                best_result = result
    
    # Strategy 2: Lattice scale optimization
    if time.time() < DEADLINE - 30:
        result = optimize_lattice_scale()
        if result is not None:
            inner, oc, side, oa = result
            if side < best_side:
                best_side = side
                best_result = result
    
    # Strategy 3: Rectangular grid
    if time.time() < DEADLINE - 20:
        result = optimize_rect_grid()
        if result is not None:
            inner, oc, side, oa = result
            if side < best_side:
                best_side = side
                best_result = result
    
    # Refine the best result
    if best_result is not None and time.time() < DEADLINE - 10:
        inner, oc, side, oa = best_result
        inner, oc, side, oa = refine_with_least_squares(inner, oc, side, oa)
        if check_validity(inner, oc, side, oa):
            best_result = (inner, oc, side, oa)
            best_side = side
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple arrangement
    inner = []
    for i in range(12):
        inner.append((0.0, 0.0, 0.0))
    return inner, (0.0, 0.0), 50.0, 0.0

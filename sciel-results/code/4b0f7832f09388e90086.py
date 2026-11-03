"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses free-placement optimization with smooth penalties for non-overlap and containment.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 135.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


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
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True


def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
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
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def build_hex_cluster_positions():
    """Build initial hexagonal cluster positions for 12 hexagons.
    Center + 6 ring + 5 of ring 2 positions.
    """
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Ring 1: 6 at distance 2 (touching for unit hexagons)
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        positions.append((x, y))
    # Ring 2: 5 positions at distance 4 (or use lattice positions)
    # Actually, for a proper hexagonal lattice, ring 2 has positions at distance 4
    # in the 6 main directions, plus 6 more at distance 2*sqrt(3) in the 6 intermediate directions
    # Let's use the 5 main direction positions
    for k in range(5):
        theta = math.radians(k * 60.0)
        x = 4.0 * math.cos(theta)
        y = 4.0 * math.sin(theta)
        positions.append((x, y))
    return positions


def params_to_inner(params):
    """Convert flat parameter vector to inner hexagon list.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11] (36 params)
    """
    inner = []
    for i in range(N):
        x = params[3 * i]
        y = params[3 * i + 1]
        ang = params[3 * i + 2]
        inner.append((x, y, ang))
    return inner


def smooth_max(values, beta=10.0):
    """Smooth approximation of max using log-sum-exp."""
    values = np.array(values)
    m = np.max(values)
    if m < -1000:
        return 0.0
    return m + math.log(np.sum(np.exp(beta * (values - m)))) / beta


def pairwise_overlap_penalty(polys, beta=20.0):
    """Compute smooth penalty for pairwise overlaps using SAT-based separation."""
    n = len(polys)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # For each edge of both polygons, check separation
            min_sep = float('inf')
            for poly in (polys[i], polys[j]):
                for k in range(6):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k + 1) % 6]
                    dx, dy = x2 - x1, y2 - y1
                    nx, ny = -dy, dx
                    norm = math.sqrt(nx * nx + ny * ny)
                    if norm < 1e-10:
                        continue
                    nx, ny = nx / norm, ny / norm
                    # Project both polygons
                    proj_i = [nx * px + ny * py for px, py in polys[i]]
                    proj_j = [nx * px + ny * py for px, py in polys[j]]
                    sep = min(proj_j) - max(proj_i)
                    if sep < min_sep:
                        min_sep = sep
                    sep2 = min(proj_i) - max(proj_j)
                    if sep2 < min_sep:
                        min_sep = sep2
            # If min_sep < 0, they overlap; penalty is smooth max of -min_sep
            if min_sep < 0:
                penalty += smooth_max([-min_sep], beta)
    return penalty


def free_placement_objective(params, beta=10.0):
    """Objective for free placement optimization.
    params: [x0, y0, a0, ..., x11, y11, a11, outer_cx, outer_cy, outer_angle]
    Returns: outer_side + penalty for overlaps and containment violations.
    """
    if time.time() > DEADLINE:
        return 1e6
    
    inner = params_to_inner(params[:3 * N])
    outer_cx = params[3 * N]
    outer_cy = params[3 * N + 1]
    outer_angle = params[3 * N + 2]
    
    # Compute outer side needed
    outer_side = compute_min_outer_side(inner, (outer_cx, outer_cy), outer_angle)
    
    # Compute overlap penalty
    polys = []
    for (x, y, ang) in inner:
        polys.append(hex_vertices(x, y, 1.0, ang))
    
    overlap_pen = pairwise_overlap_penalty(polys, beta)
    
    # Containment penalty: check if all vertices are inside outer hexagon
    outer = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    contain_pen = 0.0
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                # Compute how far outside
                # Use distance to boundary as penalty
                contain_pen += 1.0
    
    return outer_side + 10.0 * overlap_pen + 5.0 * contain_pen


def optimize_free_placement():
    """Optimize free placement of 12 hexagons."""
    # Initial positions from hexagonal cluster
    positions = build_hex_cluster_positions()
    
    # Build initial params: positions + angles + outer center + outer angle
    params = []
    for (x, y) in positions:
        params.extend([x, y, 0.0])  # initial angle 0
    params.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    best_side = 1e6
    best_params = params.copy()
    
    # Multiple restarts with different initial conditions
    restarts = [
        (0.0, 0.0),    # no rotation
        (0.0, 30.0),   # outer rotated 30 deg
        (30.0, 0.0),   # inner rotated 30 deg
        (30.0, 30.0),  # both rotated
    ]
    
    for inner_rot, outer_rot in restarts:
        if time.time() > DEADLINE - 10:
            break
        
        # Create params with rotations
        p = []
        for (x, y) in positions:
            p.extend([x, y, inner_rot])
        p.extend([0.0, 0.0, outer_rot])
        
        try:
            result = minimize(free_placement_objective, p, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4,
                                      'adaptive': True})
            if result.fun < best_side and result.fun < 100:
                best_side = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract best result
    if best_side < 1e5:
        inner = params_to_inner(best_params[:3 * N])
        outer_center = (best_params[3 * N], best_params[3 * N + 1])
        outer_angle = best_params[3 * N + 2]
        outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
        
        # Validate
        if check_validity(inner, outer_center, outer_side, outer_angle):
            return inner, outer_center, outer_side, outer_angle
    
    return None


def build_lattice_packing(scale, rot_in_deg, rot_out_deg):
    """Build a hexagonal lattice packing of 12 unit hexagons."""
    a = 2.0 * scale
    
    positions = []
    positions.append((0.0, 0.0))
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
    
    inner = []
    for (x, y) in positions:
        inner.append((x, y, rot_in_deg))
    
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out_deg)
    
    return inner, outer_center, outer_side, rot_out_deg


def optimize_lattice():
    """Optimize lattice-based packing."""
    best_side = 1e6
    best_params = (1.0, 0.0, 0.0)
    
    scale_vals = np.linspace(0.9, 2.0, 15)
    rot_in_vals = [0.0, 30.0]
    rot_out_vals = [0.0, 30.0]
    
    for scale in scale_vals:
        for rot_in in rot_in_vals:
            for rot_rot in rot_out_vals:
                if time.time() > DEADLINE - 20:
                    break
                inner, oc, side, oa = build_lattice_packing(scale, rot_in, rot_rot)
                if check_validity(inner, oc, side, oa):
                    if side < best_side:
                        best_side = side
                        best_params = (scale, rot_in, rot_rot)
    
    if best_side < 1e5:
        scale, rot_in, rot_out = best_params
        inner, oc, side, oa = build_lattice_packing(scale, rot_in, rot_out)
        return inner, oc, side, oa
    
    return None


def construct():
    t0 = time.time()
    
    # Try free placement optimization first (more powerful)
    result = optimize_free_placement()
    if result is not None:
        return result
    
    # Fallback to lattice optimization
    result = optimize_lattice()
    if result is not None:
        return result
    
    # Final fallback: simple linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

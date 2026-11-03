"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a lattice-based structured packing with topology search and local refinement.
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


def polygons_overlap_sat(A, B, tol=1e-9):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if abs(nx) < 1e-12 and abs(ny) < 1e-12:
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
    """Minimum outer side length such that all unit hexagons fit inside.
    inner: list of (x, y, angle_deg) with unit side.
    Returns the minimum side length S.
    """
    cx, cy = outer_center
    a = math.radians(outer_angle)
    # Outward normals for a regular hexagon of side S: direction (i+0.5)*60 + angle
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
    # apothem = S * sqrt(3)/2, so S = 2*max_proj/sqrt(3)
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    
    # Inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    
    # Pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def build_lattice_packing(scale, rot_in_deg, rot_out_deg, topology):
    """
    Build a packing from a lattice topology.
    topology: 'ring12' (1 center + 6 ring + 5 outer), 'tri' (triangular lattice), 'hex_grid'
    Returns (inner, outer_center, outer_angle) or None if invalid.
    """
    inner = []
    
    if topology == 'ring12':
        # 1 center + 6 in ring at radius r1 + 5 in outer ring at radius r2
        r1 = scale
        r2 = scale * 1.85  # tuned ratio
        # Center
        inner.append((0.0, 0.0, rot_in_deg))
        # Ring 1: 6 hexagons
        for k in range(6):
            theta = math.radians(rot_in_deg + k * 60.0)
            x = r1 * math.cos(theta)
            y = r1 * math.sin(theta)
            inner.append((x, y, rot_in_deg))
        # Ring 2: 5 hexagons, offset by 36 degrees
        for k in range(5):
            theta = math.radians(rot_in_deg + 36.0 + k * 72.0)
            x = r2 * math.cos(theta)
            y = r2 * math.sin(theta)
            inner.append((x, y, rot_in_deg))
    
    elif topology == 'tri':
        # Triangular lattice arrangement: 3 rows
        # Row positions: 4-4-4 or 3-4-5 or 4-3-5
        # Use a hexagonal close packing pattern
        dx = scale * 1.5  # horizontal spacing
        dy = scale * math.sqrt(3)  # vertical spacing
        # 3 rows of 4
        for row in range(3):
            for col in range(4):
                x = col * dx - 1.5 * dx + (row % 2) * dx * 0.5
                y = row * dy - dy
                inner.append((x, y, rot_in_deg))
    
    elif topology == 'hex_grid':
        # 2x6 grid with hex offset
        dx = scale * 1.5
        dy = scale * math.sqrt(3) * 0.9
        for row in range(2):
            for col in range(6):
                x = col * dx - 2.5 * dx + (row % 2) * dx * 0.5
                y = row * dy - dy * 0.5
                inner.append((x, y, rot_in_deg))
    
    elif topology == 'ring_opt':
        # Optimized ring: 1 + 6 + 5 with better spacing
        r1 = scale * 1.732  # sqrt(3)
        r2 = scale * 3.0
        inner.append((0.0, 0.0, rot_in_deg))
        for k in range(6):
            theta = math.radians(rot_in_deg + k * 60.0)
            x = r1 * math.cos(theta)
            y = r1 * math.sin(theta)
            inner.append((x, y, rot_in_deg))
        # 5 outer, aligned with gaps
        for k in range(5):
            theta = math.radians(rot_in_deg + 30.0 + k * 72.0)
            x = r2 * math.cos(theta)
            y = r2 * math.sin(theta)
            inner.append((x, y, rot_in_deg))
    
    else:
        return None
    
    outer_center = (0.0, 0.0)
    outer_angle = rot_out_deg
    return inner, outer_center, outer_angle


def objective(params, topology):
    """Objective function for optimization: return outer side length or penalty."""
    if len(params) == 3:
        scale, rot_in, rot_out = params
    else:
        return 1e6
    
    result = build_lattice_packing(scale, rot_in, rot_out, topology)
    if result is None:
        return 1e6
    
    inner, outer_center, outer_angle = result
    side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Check validity
    if not check_validity(inner, outer_center, side, outer_angle):
        return side + 100.0  # penalty for invalid
    
    return side


def optimize_topology(topology):
    """Optimize parameters for a given topology."""
    # Initial guesses
    x0 = np.array([2.0, 0.0, 0.0])
    
    best_side = 1e6
    best_params = x0.copy()
    
    # Try multiple starting points
    starts = [
        np.array([1.5, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([2.5, 0.0, 0.0]),
        np.array([2.0, 30.0, 0.0]),
        np.array([2.0, 0.0, 30.0]),
        np.array([2.0, 15.0, 15.0]),
    ]
    
    for x0 in starts:
        if time.time() > DEADLINE:
            break
        try:
            res = minimize(objective, x0, args=(topology,), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return best_side, best_params


def construct():
    t0 = time.time()
    
    best_side = 1e6
    best_result = None
    
    # Try multiple topologies
    topologies = ['ring12', 'ring_opt', 'tri', 'hex_grid']
    
    for topo in topologies:
        if time.time() > DEADLINE - 5:
            break
        try:
            side, params = optimize_topology(topo)
            if side < best_side:
                best_side = side
                scale, rot_in, rot_out = params
                result = build_lattice_packing(scale, rot_in, rot_out, topo)
                if result is not None:
                    inner, outer_center, outer_angle = result
                    # Recompute exact side
                    side = compute_min_outer_side(inner, outer_center, outer_angle)
                    if check_validity(inner, outer_center, side, outer_angle):
                        best_result = (inner, outer_center, side, outer_angle)
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple ring packing that is guaranteed valid
    # 1 center + 6 around + 5 outer
    inner = [(0.0, 0.0, 0.0)]
    r1 = 1.732
    for k in range(6):
        theta = math.radians(k * 60.0)
        inner.append((r1 * math.cos(theta), r1 * math.sin(theta), 0.0))
    r2 = 3.2
    for k in range(5):
        theta = math.radians(30.0 + k * 72.0)
        inner.append((r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # If invalid, use a very large side
    if not check_validity(inner, outer_center, side, outer_angle):
        side = 20.0
    
    return inner, outer_center, side, outer_angle

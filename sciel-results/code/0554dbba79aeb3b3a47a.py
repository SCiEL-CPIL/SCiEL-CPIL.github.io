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


def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # normal to edge
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            # project A
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            # project B
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False  # separating axis found
    return True


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
    """Minimum outer side length such that all unit hexagons fit inside.
    For a regular hexagon of side S, apothem = S*sqrt(3)/2.
    A point p is inside iff for each of the 6 outward normals n_k, (p-c)·n_k <= apothem.
    So S >= (2/sqrt(3)) * max over all vertices and all k of (p-c)·n_k.
    """
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


def build_lattice_packing(scale, rot_in_deg, rot_out_deg, pattern='ring12'):
    """
    Build a hexagonal lattice packing of 12 unit hexagons.
    pattern: 'ring12' (center+6+5), 'hex6+6' (two rings of 6), 'rect3x4' (rectangular grid)
    """
    a = 2.0 * scale  # lattice constant
    
    positions = []
    
    if pattern == 'ring12':
        # Center + Ring 1 (6) + Ring 2 (5)
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
    elif pattern == 'hex6+6':
        # Two full rings: Ring 1 (6) + Ring 2 (6) - no center
        # This gives a hexagonal ring of 12
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(6):
            theta = math.radians(k * 60.0 + 30.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
    elif pattern == 'rect3x4':
        # 3 rows x 4 columns rectangular arrangement
        # Row spacing = a * sin(60) = a * sqrt(3)/2
        # Column spacing = a
        row_spacing = a * math.sqrt(3) / 2.0
        for row in range(3):
            for col in range(4):
                x = col * a
                y = row * row_spacing
                if row % 2 == 1:
                    x += a / 2.0  # offset alternating rows
                positions.append((x - 1.5 * a, y - row_spacing))
    else:
        # Default to ring12
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
    
    # Build inner list with rotation
    inner = []
    for (x, y) in positions:
        inner.append((x, y, rot_in_deg))
    
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out_deg)
    
    return inner, outer_center, outer_side, rot_out_deg


def objective(params, pattern='ring12'):
    """Objective function for optimization. Returns the outer side length,
    or a large penalty if the configuration is invalid."""
    if time.time() > DEADLINE:
        return 1e6
    
    scale, rot_in_deg, rot_out_deg = params
    
    # Constrain scale to reasonable range
    if scale < 0.7 or scale > 3.5:
        return 1e6
    
    inner, outer_center, outer_side, outer_angle = build_lattice_packing(scale, rot_in_deg, rot_out_deg, pattern)
    
    # Check validity
    if not check_validity(inner, outer_center, outer_side, outer_angle):
        return 1e6
    
    return outer_side


def optimize_topology():
    """Search over lattice parameters to find the best valid packing."""
    best_side = 1e6
    best_params = (1.0, 0.0, 0.0)
    best_pattern = 'ring12'
    
    patterns = ['ring12', 'hex6+6', 'rect3x4']
    
    # Coarse grid search over all patterns
    scale_vals = np.linspace(0.8, 2.5, 20)
    rot_in_vals = [0.0, 15.0, 30.0, 45.0]
    rot_out_vals = [0.0, 15.0, 30.0]
    
    for pattern in patterns:
        for scale in scale_vals:
            for rot_in in rot_in_vals:
                for rot_out in rot_out_vals:
                    if time.time() > DEADLINE:
                        return best_side, best_params, best_pattern
                    
                    side = objective([scale, rot_in, rot_out], pattern)
                    if side < best_side:
                        best_side = side
                        best_params = (scale, rot_in, rot_out)
                        best_pattern = pattern
    
    # Local refinement using scipy for the best pattern
    if best_side < 1e5:
        try:
            # Create a wrapper that fixes the pattern
            def obj_wrapped(params):
                return objective(params, best_pattern)
            
            result = minimize(obj_wrapped, best_params, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_side:
                best_side = result.fun
                best_params = tuple(result.x)
        except Exception:
            pass
    
    return best_side, best_params, best_pattern


def construct():
    t0 = time.time()
    
    # Try lattice packing optimization
    try:
        best_side, best_params, best_pattern = optimize_topology()
        if best_side < 1e5:
            scale, rot_in_deg, rot_out_deg = best_params
            inner, outer_center, outer_side, outer_angle = build_lattice_packing(
                scale, rot_in_deg, rot_out_deg, best_pattern)
            return inner, outer_center, outer_side, outer_angle
    except Exception:
        pass
    
    # Fallback: simple hexagonal cluster
    a = 2.0
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    
    inner = [(x, y, 0.0) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, 0.0)
    return inner, outer_center, outer_side, 0.0

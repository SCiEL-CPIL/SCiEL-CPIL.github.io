"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses hexagonal lattice topologies + scipy local refinement with smooth constraints.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 138.0  # safety margin before 150s kill
EPS = 1e-7

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

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
            if maxA < minB - EPS or maxB < minA - EPS:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle):
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

# --- Topology builders ---

def build_hex_lattice(rows, cols, outer_angle=0.0):
    """Build a hexagonal lattice of unit hexagons (same orientation).
    rows x cols grid with hexagonal spacing.
    Returns list of (x, y, angle_deg)."""
    inner = []
    # For same-orientation hexagons, nearest neighbor distance = 2 * apothem = sqrt(3)
    # Hex grid: offset every other row
    dx = 3.0  # horizontal spacing for flat-top hexagons (side 1): actually 3 for pointy? 
    # Let's use the standard: for flat-top hex (angle 0), vertices at 0,60,120,...
    # Nearest neighbor in same orientation: distance = sqrt(3) * 1 = sqrt(3) ~ 1.732
    # But actually for two same-oriented unit hexagons to just touch, center distance = 2*apothem = sqrt(3)
    # Hex grid spacing:
    # For flat-top hexagons:
    #   horizontal neighbors: distance 3 (vertex to vertex) -> no, that's too far
    #   Actually the 6 nearest neighbors of a flat-top hex at distance sqrt(3):
    #   at angles 30, 90, 150, 210, 270, 330 degrees
    # So a hex grid with same orientation:
    # Row spacing (vertical): sqrt(3)/2 * 2 = sqrt(3) for adjacent rows? 
    # Let me think more carefully.
    # A flat-top unit hexagon has vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # The 6 edge-midpoints are at distance sqrt(3)/2 from center, at angles 30,90,150,210,270,330.
    # Two same-oriented hexagons sharing an edge: centers are 2*(sqrt(3)/2) = sqrt(3) apart, in the direction of the edge normal.
    # So the 6 neighbor offsets are:
    #   (0, sqrt(3)), (sqrt(3)*cos(30), sqrt(3)*sin(30)) = (1.5, sqrt(3)/2), etc.
    # Actually: neighbor offsets for flat-top hex grid:
    #   (0, sqrt(3)), (3/2, sqrt(3)/2), (3/2, -sqrt(3)/2), (0, -sqrt(3)), (-3/2, -sqrt(3)/2), (-3/2, sqrt(3)/2)
    # For a rectangular grid approximation:
    #   Column spacing: 3 (two hexagons side by side horizontally share a vertex, not an edge)
    #   Actually horizontal neighbors at distance 3 (vertex touching) or we can use the 6-neighbor grid.
    
    # Let me use the standard hex grid coordinates:
    # For flat-top hexagons, the 6 neighbors are at:
    #   (0, h), (w, h/2), (w, -h/2), (0, -h), (-w, -h/2), (-w, h/2)
    # where h = sqrt(3), w = 3/2
    # So a "row" in the hex grid: 
    #   Even rows: x = col * 3, y = row * h
    #   Odd rows: x = col * 3 + 1.5, y = row * h
    
    h = math.sqrt(3)
    w = 1.5
    for r in range(rows):
        for c in range(cols):
            if r % 2 == 0:
                x = c * 3.0
            else:
                x = c * 3.0 + w
            y = r * h
            inner.append((x, y, 0.0))
    return inner

def build_rect_grid(rows, cols):
    """Rectangular grid of same-oriented hexagons (may have gaps)."""
    inner = []
    h = math.sqrt(3)
    for r in range(rows):
        for c in range(cols):
            x = c * 3.0
            y = r * h
            inner.append((x, y, 0.0))
    return inner

def build_ring12():
    """1 center + 6 inner ring + 5 outer ring (72 deg spacing)."""
    inner = [(0.0, 0.0, 0.0)]
    r1 = math.sqrt(3)  # touching
    for k in range(6):
        theta = math.radians(k * 60.0)
        inner.append((r1 * math.cos(theta), r1 * math.sin(theta), 0.0))
    r2 = 2 * math.sqrt(3)
    for k in range(5):
        theta = math.radians(36.0 + k * 72.0)
        inner.append((r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
    return inner

def build_hex6_plus6():
    """Two clusters of 6 hexagons (each a flower: 1+5) placed side by side."""
    inner = []
    # Cluster 1: center at (-1.5, 0)
    cx1, cy1 = -1.5, 0.0
    inner.append((cx1, cy1, 0.0))
    for k in range(5):
        theta = math.radians(36.0 + k * 72.0)
        inner.append((cx1 + math.sqrt(3) * math.cos(theta), cy1 + math.sqrt(3) * math.sin(theta), 0.0))
    # Cluster 2: center at (1.5, 0)
    cx2, cy2 = 1.5, 0.0
    inner.append((cx2, cy2, 0.0))
    for k in range(5):
        theta = math.radians(36.0 + k * 72.0)
        inner.append((cx2 + math.sqrt(3) * math.cos(theta), cy2 + math.sqrt(3) * math.sin(theta), 0.0))
    return inner

def build_linear_chain():
    """12 hexagons in a line (fallback)."""
    inner = []
    for i in range(12):
        inner.append(((i - 5.5) * 3.0, 0.0, 0.0))
    return inner

def build_hexagonal_cluster():
    """1 + 6 + 5 = 12 in hexagonal arrangement (same as ring12 but optimized)."""
    return build_ring12()

# --- Optimization ---

def smooth_max(values, beta=50.0):
    """Smooth maximum approximation."""
    m = max(values)
    if m < -100:
        return 0.0
    s = 0.0
    for v in values:
        s += math.exp(beta * (v - m))
    return m + math.log(s) / beta

def sat_gap(A, B):
    """Minimum gap between two convex polygons (negative if overlapping)."""
    min_gap = float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0:
                continue
            nx /= norm
            ny /= norm
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            gap = min(minB - maxA, minA - maxB)
            if gap < min_gap:
                min_gap = gap
    return min_gap

def objective(params, inner_template, outer_angle=0.0):
    """
    Objective: minimize outer side length + penalty for violations.
    params: flattened [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    inner_template: initial positions to use as reference for angles (or we optimize all).
    """
    if time.time() > DEADLINE:
        return 1e6
    
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        inner.append((x, y, ang))
    
    outer_c = (0.0, 0.0)
    side = compute_min_outer_side(inner, outer_c, outer_angle)
    
    # Check containment and non-overlap with smooth penalties
    outer = hex_vertices(0.0, 0.0, side, outer_angle)
    
    # Containment penalty: max vertex distance beyond boundary
    contain_pen = 0.0
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                # Compute penetration depth (approximate)
                contain_pen += 1.0
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0:
                overlap_pen += (-gap) ** 2
    
    penalty = 10.0 * contain_pen + 5.0 * overlap_pen
    return side + penalty

def optimize_from_start(inner_init, outer_angle=0.0, max_iter=200):
    """Local optimization from initial positions using scipy."""
    if time.time() > DEADLINE:
        return None
    
    # Flatten params
    x0 = []
    for (x, y, ang) in inner_init:
        x0.extend([x, y, ang])
    
    def obj(p):
        if time.time() > DEADLINE:
            return 1e6
        return objective(p, inner_init, outer_angle)
    
    result = minimize(obj, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if time.time() > DEADLINE:
        return None
    
    best_inner = []
    for i in range(N):
        best_inner.append((result.x[3*i], result.x[3*i+1], result.x[3*i+2]))
    
    outer_c = (0.0, 0.0)
    side = compute_min_outer_side(best_inner, outer_c, outer_angle)
    
    if check_validity(best_inner, outer_c, side, outer_angle):
        return best_inner, outer_c, side, outer_angle
    
    return None

def try_topology(builder_func, outer_angle=0.0):
    """Build a topology, compute side, check validity, then optimize."""
    if time.time() > DEADLINE:
        return None
    
    inner = builder_func()
    if len(inner) != N:
        return None
    
    outer_c = (0.0, 0.0)
    side = compute_min_outer_side(inner, outer_c, outer_angle)
    
    if check_validity(inner, outer_c, side, outer_angle):
        # Try to improve with local optimization
        result = optimize_from_start(inner, outer_angle, max_iter=100)
        if result and result[2] < side:
            return result
        return inner, outer_c, side, outer_angle
    
    # Try optimization even if initial is invalid
    result = optimize_from_start(inner, outer_angle, max_iter=150)
    return result

def construct():
    t0 = time.time()
    best = None
    best_side = float('inf')
    
    topologies = [
        ("hex_lattice_3x4", lambda: build_hex_lattice(3, 4)),
        ("hex_lattice_2x6", lambda: build_hex_lattice(2, 6)),
        ("hex_lattice_4x3", lambda: build_hex_lattice(4, 3)),
        ("rect_3x4", lambda: build_rect_grid(3, 4)),
        ("rect_2x6", lambda: build_rect_grid(2, 6)),
        ("rect_4x3", lambda: build_rect_grid(4, 3)),
        ("ring12", build_ring12),
        ("hex6_plus6", build_hex6_plus6),
        ("linear_chain", build_linear_chain),
        ("hex_cluster", build_hexagonal_cluster),
    ]
    
    for name, builder in topologies:
        if time.time() > DEADLINE:
            break
        try:
            result = try_topology(builder)
            if result:
                inner, oc, side, oa = result
                if side < best_side:
                    best_side = side
                    best = result
        except Exception:
            continue
    
    if best:
        return best
    
    # Fallback
    inner = build_linear_chain()
    outer_c = (0.0, 0.0)
    side = compute_min_outer_side(inner, outer_c, 0.0)
    return inner, outer_c, side, 0.0

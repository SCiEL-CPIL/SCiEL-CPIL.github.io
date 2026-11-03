"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses contact-graph based optimization: fix outer hexagon, optimize inner positions
via penalty-based SLSQP with explicit non-overlap and containment constraints.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 148.0  # safety margin before 150s kill

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
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True

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

# --- Objective function for fixed outer hexagon ---
# Parameters: 12*(x,y,angle) = 36 params

def decode_params(params):
    """Decode parameter vector into inner positions."""
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        inner.append((x, y, ang))
    return inner

def encode_config(inner):
    """Encode a configuration into parameter vector."""
    params = []
    for (x, y, ang) in inner:
        params.extend([x, y, ang])
    return np.array(params, dtype=float)

def objective_fixed_outer(params, outer_center, outer_side, outer_angle):
    """
    Objective: sum of constraint violations (non-overlap + containment).
    Minimize to 0 for a valid packing.
    """
    if time.time() > DEADLINE - 3.0:
        return 1e10
    
    inner = decode_params(params)
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    penalty = 0.0
    
    # Containment: each vertex of each inner hex must be inside outer
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                # Distance to outer boundary (approximate)
                # Find minimum distance from point to outer polygon
                min_dist = float('inf')
                for i in range(6):
                    x1, y1 = outer[i]
                    x2, y2 = outer[(i+1) % 6]
                    # Point-to-line-segment distance
                    dx, dy = x2-x1, y2-y1
                    if dx == 0 and dy == 0:
                        d = math.sqrt((px-x1)**2 + (py-y1)**2)
                    else:
                        t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx+dy*dy)))
                        qx, qy = x1+t*dx, y1+t*dy
                        d = math.sqrt((px-qx)**2 + (py-qy)**2)
                    min_dist = min(min_dist, d)
                penalty += min_dist ** 2
    
    # Non-overlap: pairwise SAT gap
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap_sat(polys[i], polys[j]):
                # Compute overlap depth (approximate)
                # Use the maximum penetration depth
                overlap = 0.0
                for poly in (polys[i], polys[j]):
                    n = len(poly)
                    for k in range(n):
                        x1, y1 = poly[k]
                        x2, y2 = poly[(k+1) % n]
                        dx, dy = x2-x1, y2-y1
                        nx, ny = -dy, dx
                        norm = math.sqrt(nx*nx + ny*ny)
                        if norm == 0:
                            continue
                        nx, ny = nx/norm, ny/norm
                        minA = min(nx*px+ny*py for px,py in polys[i])
                        maxA = max(nx*px+ny*py for px,py in polys[i])
                        minB = min(nx*px+ny*py for px,py in polys[j])
                        maxB = max(nx*px+ny*py for px,py in polys[j])
                        gap = min(maxA, maxB) - max(minA, minB)
                        if gap > overlap:
                            overlap = gap
                penalty += overlap ** 2
    
    return penalty

def objective_smooth_fixed_outer(params, outer_center, outer_side, outer_angle, beta=50.0):
    """Smooth version of the objective for better gradient behavior."""
    if time.time() > DEADLINE - 3.0:
        return 1e10
    
    inner = decode_params(params)
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # For smoothness, use a soft penalty
    penalty = 0.0
    
    # Containment
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            # Check if inside
            inside = point_in_convex_polygon(px, py, outer)
            if not inside:
                min_dist = float('inf')
                for i in range(6):
                    x1, y1 = outer[i]
                    x2, y2 = outer[(i+1) % 6]
                    dx, dy = x2-x1, y2-y1
                    if dx == 0 and dy == 0:
                        d = math.sqrt((px-x1)**2 + (py-y1)**2)
                    else:
                        t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx+dy*dy)))
                        qx, qy = x1+t*dx, y1+t*dy
                        d = math.sqrt((px-qx)**2 + (py-qy)**2)
                    min_dist = min(min_dist, d)
                penalty += min_dist ** 2
    
    # Non-overlap
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap_sat(polys[i], polys[j]):
                overlap = 0.0
                for poly in (polys[i], polys[j]):
                    n = len(poly)
                    for k in range(n):
                        x1, y1 = poly[k]
                        x2, y2 = poly[(k+1) % n]
                        dx, dy = x2-x1, y2-y1
                        nx, ny = -dy, dx
                        norm = math.sqrt(nx*nx + ny*ny)
                        if norm == 0:
                            continue
                        nx, ny = nx/norm, ny/norm
                        minA = min(nx*px+ny*py for px,py in polys[i])
                        maxA = max(nx*px+ny*py for px,py in polys[i])
                        minB = min(nx*px+ny*py for px,py in polys[j])
                        maxB = max(nx*px+ny*py for px,py in polys[j])
                        gap = min(maxA, maxB) - max(minA, minB)
                        if gap > overlap:
                            overlap = gap
                penalty += overlap ** 2
    
    return penalty

# --- Initial configuration builders ---

def build_hex_lattice_ring():
    """12 hexagons in a ring pattern."""
    inner = []
    for k in range(12):
        theta = k * math.pi / 6.0
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner

def build_two_rings():
    """Two concentric rings of 6."""
    inner = []
    for k in range(6):
        theta = k * math.pi / 3.0
        x = 1.8 * math.cos(theta)
        y = 1.8 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(6):
        theta = (k + 0.5) * math.pi / 3.0
        x = 3.2 * math.cos(theta)
        y = 3.2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner

def build_center_plus_ring():
    """Center hexagon + ring of 11 (need 12 total)."""
    inner = [(0.0, 0.0, 0.0)]
    for k in range(11):
        theta = k * 2 * math.pi / 11.0
        x = 2.2 * math.cos(theta)
        y = 2.2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner

def build_rect_grid():
    """3x4 rectangular grid."""
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * 2.0
            y = (j - 1.5) * 1.8
            inner.append((x, y, 0.0))
    return inner

def build_random():
    """Random initial configuration."""
    rng = np.random.RandomState(42)
    inner = []
    for i in range(N):
        x = rng.uniform(-2, 2)
        y = rng.uniform(-2, 2)
        ang = rng.uniform(0, 180)
        inner.append((x, y, ang))
    return inner

# --- Optimization ---

def optimize_for_fixed_outer(outer_center, outer_side, outer_angle, initial_configs):
    """Optimize inner positions for a fixed outer hexagon."""
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 180.0)) # angle
    
    best_x = None
    best_penalty = float('inf')
    
    # Try each initial configuration
    for inner in initial_configs:
        x0 = encode_config(inner)
        
        # First check if it's already valid
        penalty = objective_fixed_outer(x0, outer_center, outer_side, outer_angle)
        if penalty < best_penalty:
            best_penalty = penalty
            best_x = x0.copy()
        
        if penalty < 1e-6:
            break
    
    if best_x is None:
        return None, float('inf')
    
    # Local optimization from best starting point
    if time.time() < DEADLINE - 5.0:
        try:
            result = minimize(
                objective_smooth_fixed_outer, best_x,
                args=(outer_center, outer_side, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_penalty:
                best_penalty = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Polish with exact objective
    if time.time() < DEADLINE - 3.0 and best_penalty < 1e-3:
        try:
            result = minimize(
                objective_fixed_outer, best_x,
                args=(outer_center, outer_side, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_penalty:
                best_penalty = result.fun
                best_x = result.x
        except Exception:
            pass
    
    return best_x, best_penalty

def optimize():
    """Main optimization routine."""
    t_start = time.time()
    
    # Generate initial configurations
    initial_configs = [
        build_hex_lattice_ring(),
        build_two_rings(),
        build_center_plus_ring(),
        build_rect_grid(),
        build_random()
    ]
    
    # Binary search on outer side length
    # Lower bound: ~3.0 (minimum for 12 hexagons)
    # Upper bound: ~5.0 (definitely fits)
    lo = 3.0
    hi = 5.0
    
    # First, find a feasible upper bound
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Check if hi is feasible
    _, penalty_hi = optimize_for_fixed_outer(outer_center, hi, outer_angle, initial_configs)
    
    if penalty_hi > 1e-4:
        # hi is not feasible, increase
        hi = 5.5
        _, penalty_hi = optimize_for_fixed_outer(outer_center, hi, outer_angle, initial_configs)
        if penalty_hi > 1e-4:
            hi = 6.0
    
    # Binary search for minimum feasible side
    best_inner = None
    best_side = hi
    
    for _ in range(8):  # 8 iterations of binary search
        if time.time() > DEADLINE - 10.0:
            break
        
        mid = (lo + hi) / 2.0
        x, penalty = optimize_for_fixed_outer(outer_center, mid, outer_angle, initial_configs)
        
        if penalty < 1e-5:
            hi = mid
            best_side = mid
            best_inner = x
        else:
            lo = mid
    
    # Final refinement: try to improve the best solution found
    if best_inner is not None and time.time() < DEADLINE - 5.0:
        inner = decode_params(best_inner)
        
        # Compute the actual minimum side for this configuration
        actual_side = compute_min_outer_side(inner, outer_center, outer_angle)
        
        # Verify validity
        if check_validity(inner, outer_center, actual_side, outer_angle):
            best_side = actual_side
    
    # If we didn't find a valid solution, use the best we have
    if best_inner is None:
        # Fallback: use the largest outer that we tried
        inner = build_hex_lattice_ring()
        best_inner = encode_config(inner)
        best_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    inner = decode_params(best_inner)
    
    # Final validity check and adjustment
    side = compute_min_outer_side(inner, outer_center, outer_angle)
    if not check_validity(inner, outer_center, side, outer_angle):
        # Scale inner positions toward center
        for scale in [0.95, 0.9, 0.85, 0.8, 0.75]:
            inner_scaled = [(x*scale, y*scale, ang) for (x, y, ang) in inner]
            side_scaled = compute_min_outer_side(inner_scaled, outer_center, outer_angle)
            if check_validity(inner_scaled, outer_center, side_scaled, outer_angle):
                inner = inner_scaled
                side = side_scaled
                break
    
    return inner, outer_center, side, outer_angle

def construct():
    """Entry point: construct the packing."""
    t0 = time.time()
    try:
        inner, center, side, outer_angle = optimize()
        # Final safety check
        if check_validity(inner, center, side, outer_angle):
            return inner, center, side, outer_angle
    except Exception:
        pass
    
    # Fallback: simple ring arrangement
    inner = []
    for k in range(12):
        theta = k * math.pi / 6.0
        x = 2.5 * math.cos(theta)
        y = 2.5 * math.sin(theta)
        inner.append((x, y, 0.0))
    side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
    return inner, (0.0, 0.0), side, 0.0

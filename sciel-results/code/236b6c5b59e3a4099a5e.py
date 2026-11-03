"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses free-placement optimization with smooth penalty-based objective and multiple restarts.
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


def smooth_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = max(values)
    if m == 0:
        return 0.0
    exps = [math.exp(beta * (v - m)) for v in values]
    s = sum(exps)
    if s == 0:
        return m
    return m + math.log(s) / beta


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


def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta=30.0):
    """Smooth version of compute_min_outer_side for gradient-based optimization."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    all_projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                all_projs.append(proj)
    sm = smooth_max(all_projs, beta)
    return (2.0 / math.sqrt(3.0)) * sm


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


def pairwise_overlap_penalty(polys, soft_beta=20.0):
    """
    Compute a smooth penalty for pairwise overlaps.
    For each pair, compute the minimum distance between the two convex polygons.
    If they overlap, the penalty is positive.
    """
    penalty = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            # Compute minimum distance between two convex polygons
            # Using SAT-based approach: find the axis of minimum separation
            min_sep = float('inf')
            A = polys[i]
            B = polys[j]
            # Check all edges of both polygons
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
                    # Project both polygons
                    minA = min(nx * px + ny * py for px, py in A)
                    maxA = max(nx * px + ny * py for px, py in A)
                    minB = min(nx * px + ny * py for px, py in B)
                    maxB = max(nx * px + ny * py for px, py in B)
                    # Separation on this axis
                    if maxA < minB:
                        sep = minB - maxA
                    elif maxB < minA:
                        sep = minA - maxB
                    else:
                        sep = 0.0
                    if sep < min_sep:
                        min_sep = sep
            
            # If min_sep < 0, they overlap. Use smooth penalty.
            # We want penalty to be 0 when min_sep >= 0, and positive when min_sep < 0
            # Use: penalty = -min_sep * (1 - exp(-soft_beta * max(0, -min_sep)))
            # Simpler: penalty = max(0, -min_sep) but smoothed
            if min_sep < 0:
                # Smooth approximation of max(0, -min_sep)
                val = -min_sep
                penalty += val / (1.0 + math.exp(-soft_beta * val))
    
    return penalty


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


def params_to_inner(params):
    """Convert flat parameter vector to inner hexagon list."""
    inner = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        ang = params[i * 3 + 2]
        inner.append((x, y, ang))
    return inner


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


def free_placement_objective(params, outer_center=(0.0, 0.0), outer_angle=0.0, 
                               overlap_weight=10.0, boundary_weight=10.0):
    """
    Objective for free placement optimization.
    params: 36-dim vector [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    Returns: outer_side + overlap_penalty + boundary_penalty
    """
    if time.time() > DEADLINE:
        return 1e6
    
    inner = params_to_inner(params)
    
    # Compute minimum outer side needed
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Compute overlap penalty
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_pen = pairwise_overlap_penalty(polys)
    
    # Boundary penalty: how much each hexagon extends beyond the current outer_side
    # We want all hexagons to fit in a hexagon of side outer_side
    # If they don't, add penalty
    outer_poly = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    boundary_pen = 0.0
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer_poly):
                # Point is outside; compute distance to boundary
                # Simple penalty: just add a fixed amount
                boundary_pen += 1.0
    
    return outer_side + overlap_weight * overlap_pen + boundary_weight * boundary_pen


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


def optimize_free_placement(n_restarts=5, max_time=100.0):
    """
    Optimize free placement of 12 hexagons using scipy minimize.
    Returns (best_side, best_inner, best_outer_center, best_outer_angle)
    """
    t_start = time.time()
    best_side = 1e6
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - t_start > max_time:
            break
        
        # Generate initial configuration
        # Start with a hexagonal cluster and add random perturbations
        a = 2.0
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
        
        # Add random perturbations
        noise_scale = 0.3 + 0.5 * restart / max(n_restarts - 1, 1)
        params = []
        for (x, y) in positions:
            params.append(x + rng.uniform(-noise_scale, noise_scale))
            params.append(y + rng.uniform(-noise_scale, noise_scale))
            params.append(rng.uniform(-30, 30))
        
        params = np.array(params)
        
        # Try multiple outer angles
        for outer_angle in [0.0, 15.0, 30.0]:
            if time.time() - t_start > max_time:
                break
            
            # Optimize
            try:
                result = minimize(
                    lambda p: free_placement_objective(p, (0.0, 0.0), outer_angle),
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                if result.fun < 1e5:
                    inner = params_to_inner(result.x)
                    
                    # Compute actual outer side
                    outer_side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
                    
                    # Check validity
                    if check_validity(inner, (0.0, 0.0), outer_side, outer_angle):
                        if outer_side < best_side:
                            best_side = outer_side
                            best_inner = inner
                            best_outer_center = (0.0, 0.0)
                            best_outer_angle = outer_angle
            except Exception:
                pass
    
    return best_side, best_inner, best_outer_center, best_outer_angle


def construct():
    t0 = time.time()
    
    # First try lattice packing optimization (fast)
    best_lattice_side = 1e6
    best_lattice_result = None
    try:
        best_side, best_params, best_pattern = optimize_topology()
        if best_side < 1e5:
            best_lattice_side = best_side
            scale, rot_in_deg, rot_out_deg = best_params
            inner, outer_center, outer_side, outer_angle = build_lattice_packing(
                scale, rot_in_deg, rot_out_deg, best_pattern)
            best_lattice_result = (inner, outer_center, outer_side, outer_angle)
    except Exception:
        pass
    
    # Then try free placement optimization (slower but potentially better)
    best_free_side = 1e6
    best_free_result = None
    remaining_time = DEADLINE - time.time()
    if remaining_time > 30:  # Only if we have enough time
        try:
            free_max_time = min(remaining_time - 10, 100.0)
            best_side, best_inner, best_outer_center, best_outer_angle = optimize_free_placement(
                n_restarts=5, max_time=free_max_time)
            if best_side < 1e5 and best_inner is not None:
                best_free_side = best_side
                best_free_result = (best_inner, best_outer_center, best_side, best_outer_angle)
        except Exception:
            pass
    
    # Return the best result
    if best_free_side < best_lattice_side:
        if best_free_result is not None:
            return best_free_result
    elif best_lattice_result is not None:
        return best_lattice_result
    
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

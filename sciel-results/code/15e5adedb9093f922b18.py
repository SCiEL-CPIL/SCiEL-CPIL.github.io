"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses force-directed packing with efficient distance-based repulsion.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

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


def min_pairwise_distance(polys):
    """
    Compute minimum distance between all pairs of convex polygons.
    Returns (min_dist, pair_index) where min_dist is negative if overlapping.
    Uses SAT to find minimum separation.
    """
    n = len(polys)
    min_dist = float('inf')
    min_pair = (0, 1)
    
    for i in range(n):
        for j in range(i + 1, n):
            A = polys[i]
            B = polys[j]
            sep = float('inf')
            
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
                    if maxA < minB:
                        s = minB - maxA
                    elif maxB < minA:
                        s = minA - maxB
                    else:
                        s = 0.0
                    if s < sep:
                        sep = s
            
            if sep < min_dist:
                min_dist = sep
                min_pair = (i, j)
    
    return min_dist, min_pair


def repulsion_force(params, outer_center=(0.0, 0.0), outer_angle=0.0, 
                     repulsion_strength=50.0, boundary_strength=100.0):
    """
    Compute repulsion forces for force-directed packing.
    Returns (objective_value, forces) where forces is a 36-dim vector.
    """
    inner = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        ang = params[i * 3 + 2]
        inner.append((x, y, ang))
    
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    
    # Compute outer side
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Boundary penalty: vertices outside the outer hexagon
    outer_poly = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    boundary_pen = 0.0
    forces = np.zeros(36)
    
    for i, p in enumerate(polys):
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer_poly):
                boundary_pen += 1.0
                # Push inward: compute direction from outer center
                cx, cy = outer_center
                dx, dy = px - cx, py - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-9:
                    fx = -dx / dist
                    fy = -dy / dist
                    # Distribute force to the hexagon's position
                    forces[i * 3] += fx
                    forces[i * 3 + 1] += fy
    
    # Pairwise repulsion: push apart overlapping hexagons
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            A = polys[i]
            B = polys[j]
            # Check if overlapping using SAT
            overlapping = True
            for poly in (A, B):
                m = len(poly)
                for k in range(m):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k + 1) % m]
                    dx, dy = x2 - x1, y2 - y1
                    nx, ny = -dy, dx
                    if nx == 0 and ny == 0:
                        continue
                    minA = min(nx * px + ny * py for px, py in A)
                    maxA = max(nx * px + ny * py for px, py in A)
                    minB = min(nx * px + ny * py for px, py in B)
                    maxB = max(nx * px + ny * py for px, py in B)
                    if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                        overlapping = False
                        break
                if not overlapping:
                    break
            
            if overlapping:
                # Compute center-to-center direction
                xi, yi = params[i * 3], params[i * 3 + 1]
                xj, yj = params[j * 3], params[j * 3 + 1]
                dx, dy = xj - xi, yj - yi
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-9:
                    fx = dx / dist
                    fy = dy / dist
                    forces[i * 3] -= fx
                    forces[i * 3 + 1] -= fy
                    forces[j * 3] += fx
                    forces[j * 3 + 1] += fy
                    overlap_pen += 1.0
    
    objective = outer_side + repulsion_strength * overlap_pen + boundary_strength * boundary_pen
    return objective, forces


def optimize_force_directed(n_restarts=8, max_iter=200, max_time=100.0):
    """
    Force-directed optimization for packing 12 hexagons.
    """
    t_start = time.time()
    best_side = 1e6
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal cluster (center + 6 + 5)
    a = 2.0
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    initial_configs.append([(x, y, 0.0) for (x, y) in positions])
    
    # Config 2: Two rings of 6
    positions = []
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(6):
        theta = math.radians(k * 60.0 + 30.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    initial_configs.append([(x, y, 0.0) for (x, y) in positions])
    
    # Config 3: 3x4 grid
    row_spacing = a * math.sqrt(3) / 2.0
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * a
            y = row * row_spacing
            if row % 2 == 1:
                x += a / 2.0
            positions.append((x - 1.5 * a, y - row_spacing))
    initial_configs.append([(x, y, 0.0) for (x, y) in positions])
    
    # Config 4-8: Random perturbations of config 1
    for r in range(5):
        noise = 0.5 + 0.3 * r
        positions = []
        for (x, y, _) in initial_configs[0]:
            positions.append((x + rng.uniform(-noise, noise), 
                            y + rng.uniform(-noise, noise),
                            rng.uniform(-20, 20)))
        initial_configs.append(positions)
    
    for restart in range(n_restarts):
        if time.time() - t_start > max_time:
            break
        if restart >= len(initial_configs):
            break
        
        config = initial_configs[restart]
        
        # Try multiple outer angles
        for outer_angle in [0.0, 15.0, 30.0]:
            if time.time() - t_start > max_time:
                break
            
            params = np.array([v for item in config for v in item])
            
            # Force-directed iteration
            step_size = 0.1
            for it in range(max_iter):
                if time.time() - t_start > max_time:
                    break
                
                obj, forces = repulsion_force(params, (0.0, 0.0), outer_angle)
                
                # Update positions
                for i in range(N):
                    params[i * 3] += step_size * forces[i * 3]
                    params[i * 3 + 1] += step_size * forces[i * 3 + 1]
                    # Angles: small random jitter
                    params[i * 3 + 2] += rng.uniform(-0.5, 0.5)
                
                # Reduce step size
                step_size *= 0.95
            
            # Check final configuration
            inner = []
            for i in range(N):
                inner.append((params[i * 3], params[i * 3 + 1], params[i * 3 + 2]))
            
            outer_side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
            
            if check_validity(inner, (0.0, 0.0), outer_side, outer_angle):
                if outer_side < best_side:
                    best_side = outer_side
                    best_inner = inner
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
    
    return best_side, best_inner, best_outer_center, best_outer_angle


def optimize_nelder_mead(n_restarts=5, max_time=80.0):
    """
    Nelder-Mead optimization with efficient objective.
    """
    t_start = time.time()
    best_side = 1e6
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    rng = np.random.RandomState(123)
    
    # Initial configurations
    a = 2.0
    base_positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        base_positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        base_positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    
    for restart in range(n_restarts):
        if time.time() - t_start > max_time:
            break
        
        noise = 0.3 + 0.4 * restart / max(n_restarts - 1, 1)
        params = []
        for (x, y) in base_positions:
            params.append(x + rng.uniform(-noise, noise))
            params.append(y + rng.uniform(-noise, noise))
            params.append(rng.uniform(-15, 15))
        
        params = np.array(params)
        
        for outer_angle in [0.0, 30.0]:
            if time.time() - t_start > max_time:
                break
            
            def obj(p):
                if time.time() > DEADLINE:
                    return 1e6
                inner = []
                for i in range(N):
                    inner.append((p[i * 3], p[i * 3 + 1], p[i * 3 + 2]))
                side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
                # Quick validity check
                outer = hex_vertices(0.0, 0.0, side, outer_angle)
                for (x, y, ang) in inner:
                    pverts = hex_vertices(x, y, 1.0, ang)
                    for (px, py) in pverts:
                        if not point_in_convex_polygon(px, py, outer):
                            return side + 10.0
                # Pairwise overlap check (simplified)
                polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
                for i in range(N):
                    for j in range(i + 1, N):
                        if polygons_overlap_sat(polys[i], polys[j]):
                            return side + 10.0
                return side
            
            try:
                result = minimize(obj, params, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
                if result.fun < 1e5:
                    inner = []
                    for i in range(N):
                        inner.append((result.x[i * 3], result.x[i * 3 + 1], result.x[i * 3 + 2]))
                    outer_side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
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
    
    best_side = 1e6
    best_result = None
    
    # Method 1: Force-directed packing (fast, diverse)
    remaining = DEADLINE - time.time()
    if remaining > 40:
        try:
            side, inner, center, angle = optimize_force_directed(
                n_restarts=8, max_iter=150, max_time=min(remaining - 20, 80.0))
            if side < 1e5 and inner is not None:
                best_side = side
                best_result = (inner, center, side, angle)
        except Exception:
            pass
    
    # Method 2: Nelder-Mead refinement
    remaining = DEADLINE - time.time()
    if remaining > 30:
        try:
            side, inner, center, angle = optimize_nelder_mead(
                n_restarts=4, max_time=min(remaining - 10, 60.0))
            if side < best_side and inner is not None:
                best_side = side
                best_result = (inner, center, side, angle)
        except Exception:
            pass
    
    # Fallback: simple hexagonal cluster
    if best_result is None:
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
    
    return best_result

"""NLP-based packing of 11 unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import math

N = 11
_DEADLINE = 150.0  # seconds wall-clock limit (evaluator kills at 150s)
_TIME_MARGIN = 15.0  # safety margin before deadline


def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side length, rotation."""
    a = math.radians(angle_deg)
    verts = np.array([[math.cos(a + math.pi / 3 * k), math.sin(a + math.pi / 3 * k)] for k in range(6)]) * side
    return verts + np.array([cx, cy])


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    cos_a, sin_a = math.cos(-a), math.sin(-a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    inradius = side * math.sqrt(3) / 2
    for k in range(6):
        theta = math.pi / 6 + k * math.pi / 3
        if rx * math.cos(theta) + ry * math.sin(theta) > inradius + 1e-9:
            return False
    return True


def _support_function_distance(v1, v2):
    """
    Compute the minimum distance between two convex hexagons given their vertex arrays.
    Returns 0 if they overlap, positive if separated.
    Uses the separating axis theorem for hexagons.
    """
    # For two convex polygons, if there exists a separating axis, they don't overlap.
    # The candidate axes are the normals to all edges of both polygons.
    n1 = len(v1)
    n2 = len(v2)
    
    def project(verts, axis):
        dots = verts @ axis
        return dots.min(), dots.max()
    
    # Collect all edge normals
    axes = []
    for i in range(n1):
        edge = v1[(i + 1) % n1] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            axes.append(normal / norm)
    for i in range(n2):
        edge = v2[(i + 1) % n2] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            axes.append(normal / norm)
    
    min_gap = float('inf')
    for axis in axes:
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        # Gap is the distance between the two intervals
        gap = max(min2 - max1, min1 - max2)
        if gap > 0:
            min_gap = min(min_gap, gap)
        else:
            # They overlap on this axis
            return 0.0
    
    if min_gap < float('inf'):
        return min_gap
    return 0.0


def _min_distance_between_hexes(ix1, iy1, ia1, ix2, iy2, ia2):
    """Minimum distance between two unit hexagons (0 if overlapping)."""
    v1 = _hex_vertices(ix1, iy1, 1.0, ia1)
    v2 = _hex_vertices(ix2, iy2, 1.0, ia2)
    return _support_function_distance(v1, v2)


def _construct_objective(params):
    """
    Objective: minimize outer side length.
    params: [ocx, ocy, oside, oangle, ix1, iy1, ia1, ..., ixN, iyN, iaN]
    """
    ocx, ocy, oside, oangle = params[0], params[1], params[2], params[3]
    oside = max(oside, 1.0)
    return oside


def _construct_constraints(params):
    """
    Return list of constraint dicts for scipy minimize with method='SLSQP'.
    Each constraint: {'type': 'ineq', 'fun': lambda p: ...}
    We need:
    1. Containment: each inner hex vertex inside outer hex
    2. Non-overlap: each pair of inner hexes don't overlap
    """
    ocx, ocy, oside, oangle = params[0], params[1], params[2], params[3]
    oside = max(oside, 1.0)
    
    constraints = []
    
    # Containment constraints: for each inner hex, each vertex must be inside outer hex
    a = math.radians(oangle)
    cos_a, sin_a = math.cos(-a), math.sin(-a)
    inradius = oside * math.sqrt(3) / 2
    
    for i in range(N):
        ix = params[4 + i * 3]
        iy = params[5 + i * 3]
        iang = params[6 + i * 3]
        inner_verts = _hex_vertices(ix, iy, 1.0, iang)
        for k in range(6):
            vx, vy = inner_verts[k]
            # Transform to outer hex local frame
            dx, dy = vx - ocx, vy - ocy
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            # For each of 6 edges of outer hex
            for edge_k in range(6):
                theta = math.pi / 6 + edge_k * math.pi / 3
                # Constraint: rx*cos(theta) + ry*sin(theta) <= inradius
                # Inequality form: inradius - (rx*cos(theta) + ry*sin(theta)) >= 0
                ct = math.cos(theta)
                st = math.sin(theta)
                # This is a function of params that we can't easily make linear
                # So we'll use penalty-based approach instead
                pass
    
    # For SLSQP, we need differentiable constraints. Let's use a penalty approach
    # with the objective instead, but make it smoother.
    
    return constraints


def _smooth_objective(params):
    """
    Smooth objective: outer side + penalty for containment violations + penalty for overlaps.
    Uses smooth approximations.
    """
    ocx, ocy, oside, oangle = params[0], params[1], params[2], params[3]
    oside = max(oside, 1.0)
    
    penalty = 0.0
    
    # Containment penalty
    a = math.radians(oangle)
    cos_a, sin_a = math.cos(-a), math.sin(-a)
    inradius = oside * math.sqrt(3) / 2
    
    for i in range(N):
        ix = params[4 + i * 3]
        iy = params[5 + i * 3]
        iang = params[6 + i * 3]
        inner_verts = _hex_vertices(ix, iy, 1.0, iang)
        for k in range(6):
            vx, vy = inner_verts[k]
            dx, dy = vx - ocx, vy - ocy
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            for edge_k in range(6):
                theta = math.pi / 6 + edge_k * math.pi / 3
                dist = rx * math.cos(theta) + ry * math.sin(theta)
                violation = dist - inradius
                if violation > 0:
                    penalty += violation * violation
    
    # Non-overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            ix1, iy1, ia1 = params[4 + i * 3], params[5 + i * 3], params[6 + i * 3]
            ix2, iy2, ia2 = params[4 + j * 3], params[5 + j * 3], params[6 + j * 3]
            v1 = _hex_vertices(ix1, iy1, 1.0, ia1)
            v2 = _hex_vertices(ix2, iy2, 1.0, ia2)
            # Compute overlap penalty using vertex-in-polygon and edge intersection
            # For smoothness, use a soft penalty
            # Check if any vertex of v1 is inside v2 or vice versa
            def point_in_poly(px, py, verts):
                n = len(verts)
                inside = False
                j_idx = n - 1
                for i_idx in range(n):
                    xi, yi = verts[i_idx]
                    xj, yj = verts[j_idx]
                    if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                        inside = not inside
                    j_idx = i_idx
                return inside
            
            overlap = 0.0
            for vx, vy in v1:
                if point_in_poly(vx, vy, v2):
                    overlap = 1.0
                    break
            if overlap == 0:
                for vx, vy in v2:
                    if point_in_poly(vx, vy, v1):
                        overlap = 1.0
                        break
            if overlap == 0:
                # Check edge intersections
                def seg_intersect(p1, p2, p3, p4):
                    def ccw(a, b, c):
                        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
                    d1 = ccw(p3, p4, p1)
                    d2 = ccw(p3, p4, p2)
                    d3 = ccw(p1, p2, p3)
                    d4 = ccw(p1, p2, p4)
                    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                        return True
                    return False
                
                n1 = len(v1)
                n2 = len(v2)
                for i_idx in range(n1):
                    for j_idx in range(n2):
                        if seg_intersect(v1[i_idx], v1[(i_idx + 1) % n1], v2[j_idx], v2[(j_idx + 1) % n2]):
                            overlap = 1.0
                            break
                    if overlap > 0:
                        break
            
            if overlap > 0:
                penalty += overlap * 10.0
    
    return oside + penalty


def _initial_guess():
    """Create a safe initial guess with hexagonal packing."""
    # Use a hexagonal arrangement with safe spacing
    positions = [(0.0, 0.0)]
    d1 = 3.5  # Safe distance for ring 1
    for k in range(6):
        angle = k * math.pi / 3
        positions.append((d1 * math.cos(angle), d1 * math.sin(angle)))
    d2 = 6.5  # Safe distance for ring 2
    for k in range(4):
        angle = (k * math.pi / 3) + math.pi / 6
        positions.append((d2 * math.cos(angle), d2 * math.sin(angle)))
    
    init = [0.0, 0.0, 15.0, 0.0]
    for px, py in positions:
        init.extend([px, py, 0.0])
    return np.array(init)


def _verify_solution(params):
    """Verify that the solution is valid (no overlaps, all contained)."""
    ocx, ocy, oside, oangle = params[0], params[1], params[2], params[3]
    oside = max(oside, 1.0)
    
    # Check containment
    for i in range(N):
        ix = params[4 + i * 3]
        iy = params[5 + i * 3]
        iang = params[6 + i * 3]
        inner_verts = _hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in inner_verts:
            if not _point_in_hex(vx, vy, ocx, ocy, oside, oangle):
                return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            ix1, iy1, ia1 = params[4 + i * 3], params[5 + i * 3], params[6 + i * 3]
            ix2, iy2, ia2 = params[4 + j * 3], params[5 + j * 3], params[6 + j * 3]
            v1 = _hex_vertices(ix1, iy1, 1.0, ia1)
            v2 = _hex_vertices(ix2, iy2, 1.0, ia2)
            
            def point_in_poly(px, py, verts):
                n = len(verts)
                inside = False
                j_idx = n - 1
                for i_idx in range(n):
                    xi, yi = verts[i_idx]
                    xj, yj = verts[j_idx]
                    if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                        inside = not inside
                    j_idx = i_idx
                return inside
            
            for vx, vy in v1:
                if point_in_poly(vx, vy, v2):
                    return False
            for vx, vy in v2:
                if point_in_poly(vx, vy, v1):
                    return False
            
            def seg_intersect(p1, p2, p3, p4):
                def ccw(a, b, c):
                    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
                d1 = ccw(p3, p4, p1)
                d2 = ccw(p3, p4, p2)
                d3 = ccw(p1, p2, p3)
                d4 = ccw(p1, p2, p4)
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
                return False
            
            n1 = len(v1)
            n2 = len(v2)
            for i_idx in range(n1):
                for j_idx in range(n2):
                    if seg_intersect(v1[i_idx], v1[(i_idx + 1) % n1], v2[j_idx], v2[(j_idx + 1) % n2]):
                        return False
    
    return True


def construct():
    t_start = time.time()
    
    # Initial guess
    init = _initial_guess()
    
    # Bounds
    bounds = [(-20, 20), (-20, 20), (1.0, 25.0), (0, 60)]
    for i in range(N):
        bounds.append((-20, 20))
        bounds.append((-20, 20))
        bounds.append((0, 60))
    
    best_params = init.copy()
    best_score = _smooth_objective(best_params)
    
    # Try multiple optimization runs with different starting points
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        elapsed = time.time() - t_start
        if elapsed > _DEADLINE - _TIME_MARGIN:
            break
        
        if restart == 0:
            x0 = init.copy()
        else:
            # Perturb the best solution found so far
            x0 = best_params.copy()
            noise_scale = max(0.5, 3.0 / (restart + 1))
            x0[4:] += rng.uniform(-noise_scale, noise_scale, size=len(x0) - 4)
            # Ensure side is reasonable
            x0[2] = max(1.0, min(25.0, x0[2] + rng.uniform(-2, 2)))
            x0[3] = (x0[3] + rng.uniform(-10, 10)) % 60
        
        # Run SLSQP
        try:
            result = minimize(
                _smooth_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Check time
        elapsed = time.time() - t_start
        if elapsed > _DEADLINE - _TIME_MARGIN:
            break
    
    # Final verification and cleanup
    # Ensure the solution is valid; if not, try to fix it
    if not _verify_solution(best_params):
        # Try a few more local optimizations with tighter penalties
        for attempt in range(3):
            elapsed = time.time() - t_start
            if elapsed > _DEADLINE - _TIME_MARGIN:
                break
            try:
                result = minimize(
                    _smooth_objective,
                    best_params,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
    
    # Extract results
    ocx, ocy, oside, oangle = best_params[0], best_params[1], best_params[2], best_params[3]
    inner = []
    for i in range(N):
        ix = best_params[4 + i * 3]
        iy = best_params[5 + i * 3]
        iang = best_params[6 + i * 3]
        inner.append([float(ix), float(iy), float(iang)])
    
    return inner, (float(ocx), float(ocy)), float(oside), float(oangle)

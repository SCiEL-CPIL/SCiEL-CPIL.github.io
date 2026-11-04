"""Pack 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np
from scipy.optimize import minimize, least_squares

N = 12
DEADLINE = 135.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return np.array(verts)

def _signed_dist_to_hex_boundary(px, py, ocx, ocy, os, oangle_deg):
    """
    Signed distance from point (px,py) to the boundary of a regular hexagon.
    Positive if inside, negative if outside.
    For a regular hexagon with 'flat-top' orientation (angle=0), the boundary
    consists of 6 lines. We compute the minimum signed distance to all 6 edge lines.
    """
    a = math.radians(oangle_deg)
    # The 6 edge normals point outward from center. For a regular hexagon,
    # the edges are at angles a + (k+0.5)*pi/3 from the x-axis... 
    # Actually, let's use the support function approach.
    # A regular hexagon with side s has inradius r = s * sqrt(3)/2.
    # The boundary is defined by 6 half-planes.
    # For edge k (k=0..5), the outward normal direction is at angle a + (k+0.5)*pi/3? 
    # Let's think: vertices are at angles a + k*pi/3.
    # Edges connect vertex k and k+1. The midpoint of edge k is at angle a + (k+0.5)*pi/3.
    # The outward normal to edge k points in the direction of the midpoint from center.
    # Distance from center to each edge (inradius) = s * sqrt(3)/2.
    
    r_in = os * math.sqrt(3) / 2.0
    min_dist = float('inf')
    for k in range(6):
        # Outward normal angle for edge k
        n_ang = a + (k + 0.5) * math.pi / 3.0
        nx = math.cos(n_ang)
        ny = math.sin(n_ang)
        # Signed distance from point to the line: (p - c) . n - r_in
        # If this is negative, point is inside (on the center side of the edge)
        d = (px - ocx) * nx + (py - ocy) * ny - r_in
        if d < min_dist:
            min_dist = d
    return min_dist

def _penetration_depth_hex_hex(v1, v2):
    """
    Approximate penetration depth between two convex hexagons.
    Uses the separating axis theorem: for each edge normal of both hexagons,
    project both hexagons onto that axis and check for overlap.
    Returns the maximum penetration (positive if overlapping, 0 if separated).
    """
    # Edges of hex1
    n1 = len(v1)
    n2 = len(v2)
    max_pen = 0.0
    
    # Axes from edges of hex1
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i+1) % n1]
        edge = p2 - p1
        # Normal
        norm = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(norm)
        if norm_len < 1e-12:
            continue
        norm = norm / norm_len
        
        # Project both hexagons onto this axis
        proj1 = np.dot(v1, norm)
        proj2 = np.dot(v2, norm)
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check overlap
        if max1 < min2 or max2 < min1:
            return 0.0  # Separating axis found
        
        # Penetration depth along this axis
        pen = min(max1, max2) - max(min1, min2)
        if pen > max_pen:
            max_pen = pen
    
    # Axes from edges of hex2
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i+1) % n2]
        edge = p2 - p1
        norm = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(norm)
        if norm_len < 1e-12:
            continue
        norm = norm / norm_len
        
        proj1 = np.dot(v1, norm)
        proj2 = np.dot(v2, norm)
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2 or max2 < min1:
            return 0.0
        
        pen = min(max1, max2) - max(min1, min2)
        if pen > max_pen:
            max_pen = pen
    
    return max_pen

def _pack_cost(params, outer_center, outer_side, outer_angle):
    """
    Compute the cost: maximum of all constraint violations.
    params: array of length 3*N: [x1,y1,a1, x2,y2,a2, ...]
    Returns a scalar cost (positive means infeasible).
    """
    x = params[:N]
    y = params[N:2*N]
    angles = params[2*N:]
    
    # Compute vertices for each inner hexagon
    verts = []
    for i in range(N):
        v = _hex_vertices(x[i], y[i], 1.0, angles[i])
        verts.append(v)
    
    max_violation = 0.0
    
    # Check inner-inner collisions
    for i in range(N):
        for j in range(i+1, N):
            pen = _penetration_depth_hex_hex(verts[i], verts[j])
            if pen > 0:
                # We want pen <= 0 (no overlap). Violation is pen.
                if pen > max_violation:
                    max_violation = pen
    
    # Check inner-outer containment
    # Each vertex of each inner hex must be inside the outer hex
    for i in range(N):
        for (vx, vy) in verts[i]:
            d = _signed_dist_to_hex_boundary(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle)
            if d < 0:
                # Violation: vertex is outside
                viol = -d
                if viol > max_violation:
                    max_violation = viol
    
    return max_violation

def _feasible(params, outer_center, outer_side, outer_angle):
    return _pack_cost(params, outer_center, outer_side, outer_angle) <= 1e-9

def _try_optimize(params0, outer_center, outer_side, outer_angle, time_limit=30.0):
    """
    Try to find a feasible packing with the given outer parameters.
    Returns (success, best_params).
    """
    t0 = time.time()
    best_params = params0.copy()
    best_cost = _pack_cost(best_params, outer_center, outer_side, outer_angle)
    
    if best_cost <= 1e-9:
        return True, best_params
    
    # Use SLSQP or trust-constr to minimize the cost
    def objective(p):
        return _pack_cost(p, outer_center, outer_side, outer_angle)
    
    # Gradient-free first (Nelder-Mead)
    try:
        res = minimize(objective, params0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if time.time() - t0 > time_limit:
            pass
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x.copy()
            if best_cost <= 1e-9:
                return True, best_params
    except Exception:
        pass
    
    # Then try L-BFGS-B with finite differences
    if time.time() - t0 < time_limit * 0.7:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12})
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_params = res2.x.copy()
                if best_cost <= 1e-9:
                    return True, best_params
        except Exception:
            pass
    
    return (best_cost <= 1e-9), best_params

def _initial_params_hex_lattice():
    """
    Generate initial positions on a hexagonal lattice.
    For 12 hexagons, use a compact hexagonal arrangement.
    """
    # Hexagonal lattice: neighbors at distance 2 (for unit hexagons, center-to-center = 2 * inradius = sqrt(3) for edge-to-edge)
    # Actually for regular hexagons of side 1, the distance between centers of adjacent hexagons in a hex packing is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    # Wait: inradius = sqrt(3)/2 ≈ 0.866. Distance between centers of two edge-sharing hexagons = 2 * inradius = sqrt(3).
    
    # Let's place them in a hexagonal pattern:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons (offset)
    # That's 12.
    
    dx = math.sqrt(3)  # horizontal spacing for adjacent hexagons
    dy = 1.5  # vertical spacing (3/2 * side)
    
    positions = []
    # 4 rows of 3
    for row in range(4):
        for col in range(3):
            x = col * dx
            if row % 2 == 1:
                x += dx / 2
            y = row * dy
            positions.append((x, y, 0.0))
    
    # Center them
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    
    params = []
    for (px, py, ang) in positions:
        params.extend([px - cx, py - cy, ang])
    
    return np.array(params)

def _initial_params_linear():
    """Linear arrangement as fallback."""
    params = []
    for k in range(N):
        params.extend([4.0 * (k - N//2), 0.0, 0.0])
    return np.array(params)

def construct():
    t_start = time.time()
    
    # Initial guess for outer hexagon
    # The seed had side=40 which is way too big. Let's start with a reasonable estimate.
    # For 12 unit hexagons, the bounding hexagon side is roughly... 
    # In a hex packing, 12 hexagons form a hexagonal cluster of "radius" 2 (in hex units).
    # The outer hexagon side would be roughly 2 * (something).
    # Let's start with outer_side = 4.0 and refine.
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Try different outer side lengths from small to large
    # The minimum possible is at least the diameter of one hex = 2
    # Let's try a range
    
    params0 = _initial_params_hex_lattice()
    params_linear = _initial_params_linear()
    
    best_result = None
    best_side = 40.0  # fallback
    
    # Binary search on outer side length
    # First, find an upper bound that works
    lo, hi = 2.0, 10.0
    
    # Check if hi works
    feasible_hi, _ = _try_optimize(params0, outer_center, hi, outer_angle, time_limit=10.0)
    if not feasible_hi:
        # Try linear
        feasible_hi, _ = _try_optimize(params_linear, outer_center, hi, outer_angle, time_limit=10.0)
    
    if not feasible_hi:
        # Just return the seed-like answer
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Binary search
    for _ in range(15):  # ~15 iterations of bisection
        if time.time() - t_start > DEADLINE - 20:
            break
        mid = (lo + hi) / 2
        feasible_mid, _ = _try_optimize(params0, outer_center, mid, outer_angle, time_limit=5.0)
        if not feasible_mid:
            feasible_mid, _ = _try_optimize(params_linear, outer_center, mid, outer_angle, time_limit=5.0)
        
        if feasible_mid:
            hi = mid
        else:
            lo = mid
    
    best_side = hi
    
    # Final optimization with the best side
    if time.time() - t_start < DEADLINE - 10:
        success, best_params = _try_optimize(params0, outer_center, best_side, outer_angle, time_limit=20.0)
        if not success:
            success, best_params = _try_optimize(params_linear, outer_center, best_side, outer_angle, time_limit=20.0)
        
        if success:
            inner = []
            for i in range(N):
                inner.append([best_params[i], best_params[N+i], best_params[2*N+i]])
            return inner, outer_center, best_side, outer_angle
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

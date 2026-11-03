"""
Placement of 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.

Strategy:
- Use a smooth surrogate (log-sum-exp / soft-max) for the non-smooth containment constraint.
- Multi-stage optimization:
  1. Simulated annealing to find a good global configuration.
  2. L-BFGS-B with smooth surrogate to polish.
  3. Final high-precision polish with tighter tolerances.
- Deadline guard: track elapsed time and return best result found with safety margin.
"""

import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # Leave 10s margin before 120s kill

def check_time():
    return time.time() - START_TIME < TIME_LIMIT

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """
    Return vertices of a regular hexagon with center (cx, cy), side length 'side',
    rotated by angle_deg degrees.
    Vertices are at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation).
    """
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """
    Check if point (px, py) is inside or on the boundary of a regular hexagon
    centered at (cx, cy) with side length 'side' and rotation angle_deg.
    Uses the fact that a regular hexagon is the intersection of 6 half-planes.
    """
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Rotate point by -angle_rad to align hexagon with flat-top
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side length s centered at origin (flat-top),
    # the constraints are:
    # |lx| <= s
    # |ly| <= s * sqrt(3) / 2
    # |lx/2 + ly*sqrt(3)/2| <= s  ... actually let's use the 6 half-plane approach
    
    # The 6 normals for a flat-top hexagon (vertices at 0, 60, 120, 180, 240, 300 deg)
    # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330 deg
    # Actually, for a hexagon with vertices at angles 0, 60, ..., the edge normals
    # are at angles 30, 90, 150, 210, 270, 330 deg.
    # The distance from center to each edge (apothem) is s * sqrt(3) / 2
    
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        normal_angle = math.radians(30 + i * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        # The point must satisfy nx*lx + ny*ly <= apothem
        if nx * lx + ny * ly > apothem + 1e-10:
            return False
    return True

def hexagon_contains_hexagon(outer_c, outer_side, outer_angle, inner_c, inner_side, inner_angle):
    """
    Check if inner hexagon is fully contained in outer hexagon.
    All 6 vertices of inner hexagon must be inside or on the outer hexagon.
    """
    inner_verts = hex_vertices(inner_c[0], inner_c[1], inner_side, inner_angle)
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, outer_c[0], outer_c[1], outer_side, outer_angle):
            return False
    return True

def hexagons_disjoint(h1_c, h1_angle, h2_c, h2_angle, side=1.0, tol=1e-9):
    """
    Check if two unit hexagons are disjoint (no interior overlap).
    For convex polygons, they are disjoint if no vertex of one is inside the other
    and no edges intersect. For simplicity and robustness, we check:
    1. No vertex of h1 is strictly inside h2
    2. No vertex of h2 is strictly inside h1
    3. No edge of h1 intersects any edge of h2
    
    Actually, for the purpose of optimization, we use a penalty-based approach
    where we compute the minimum distance between the two hexagons and penalize
    negative distances (overlap).
    
    For now, let's use a simpler check: if any vertex of one is inside the other,
    they overlap. Edge intersection is harder. Let's use a distance-based penalty.
    
    We'll compute the minimum distance between the two convex hexagons.
    If min distance < -tol, they overlap.
    """
    # Compute all 6 vertices of each hexagon
    v1 = hex_vertices(h1_c[0], h1_c[1], side, h1_angle)
    v2 = hex_vertices(h2_c[0], h2_c[1], side, h2_angle)
    
    # Check if any vertex of v1 is inside v2 or vice versa
    for vx, vy in v1:
        if point_in_hexagon(vx, vy, h2_c[0], h2_c[1], side, h2_angle):
            return False
    for vx, vy in v2:
        if point_in_hexagon(vx, vy, h1_c[0], h1_c[1], side, h1_angle):
            return False
    
    # Check edge intersections
    # Edge i of hexagon connects vertex i to vertex (i+1)%6
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4, tol):
                return False
    
    return True

def segments_intersect(p1, p2, p3, p4, tol=1e-10):
    """Check if line segment p1-p2 intersects line segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
       ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
        return True
    
    # Check for collinear cases
    if abs(d1) < tol and on_segment(p3, p1, p4):
        return True
    if abs(d2) < tol and on_segment(p3, p2, p4):
        return True
    if abs(d3) < tol and on_segment(p1, p3, p2):
        return True
    if abs(d4) < tol and on_segment(p1, p4, p2):
        return True
    
    return False

def on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if min(p[0], r[0]) - 1e-10 <= q[0] <= max(p[0], r[0]) + 1e-10 and \
       min(p[1], r[1]) - 1e-10 <= q[1] <= max(p[1], r[1]) + 1e-10:
        return True
    return False

def min_distance_between_hexagons(h1_c, h1_angle, h2_c, h2_angle, side=1.0):
    """
    Compute the minimum distance between two convex hexagons.
    Positive if disjoint, negative if overlapping.
    Uses the separating axis theorem approach.
    """
    v1 = hex_vertices(h1_c[0], h1_c[1], side, h1_angle)
    v2 = hex_vertices(h2_c[0], h2_c[1], side, h2_angle)
    
    # Get all edge normals (separating axes)
    axes = []
    for i in range(6):
        # Edge of v1
        e = v1[(i + 1) % 6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            axes.append(n / norm)
        # Edge of v2
        e = v2[(i + 1) % 6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            axes.append(n / norm)
    
    min_dist = float('inf')
    
    for axis in axes:
        # Project all vertices of v1 and v2 onto the axis
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Distance along this axis
        if max1 < min2:
            dist = min2 - max1
        elif max2 < min1:
            dist = min1 - max2
        else:
            dist = 0.0
        
        min_dist = min(min_dist, dist)
    
    return min_dist

def compute_objective(params, outer_c, outer_side, outer_angle, beta=50.0):
    """
    Compute the smooth surrogate objective.
    params: array of length 3*N = [x1, y1, a1, x2, y2, a2, ...]
    Returns: (objective_value, is_valid)
    
    The objective is:
    1. Containment penalty: for each inner hexagon, the maximum violation of
       being inside the outer hexagon (using log-sum-exp for smoothness)
    2. Disjointness penalty: for each pair, the maximum violation of being
       disjoint (negative min distance means overlap)
    """
    n = len(params) // 3
    positions = params.reshape(n, 3)
    
    # Containment check: all vertices of each inner hex must be inside outer
    # For each inner hex, compute the max distance that any vertex is outside
    # Use log-sum-exp for smooth max
    
    containment_violations = []
    for i in range(n):
        cx, cy, angle = positions[i]
        verts = hex_vertices(cx, cy, 1.0, angle)
        # For each vertex, compute how far it is outside the outer hexagon
        # We use the half-plane representation: for each of the 6 half-planes,
        # compute the violation
        max_violation = 0.0
        log_sum = 0.0
        for vx, vy in verts:
            dx = vx - outer_c[0]
            dy = vy - outer_c[1]
            angle_rad = math.radians(outer_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            apothem = outer_side * math.sqrt(3) / 2.0
            for k in range(6):
                normal_angle = math.radians(30 + k * 60)
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                violation = nx * lx + ny * ly - apothem
                if violation > max_violation:
                    max_violation = violation
                # Accumulate for log-sum-exp
                if violation > 0:
                    log_sum += math.exp(beta * violation)
        
        # Smooth max using log-sum-exp
        if log_sum > 0:
            smooth_max = math.log(log_sum) / beta
        else:
            smooth_max = 0.0
        
        containment_violations.append(max(smooth_max, max_violation))
    
    # Disjointness check: for each pair, compute min distance
    # Negative min distance means overlap
    disjoint_violations = []
    for i in range(n):
        for j in range(i + 1, n):
            c1 = (positions[i, 0], positions[i, 1])
            a1 = positions[i, 2]
            c2 = (positions[j, 0], positions[j, 1])
            a2 = positions[j, 2]
            min_d = min_distance_between_hexagons(c1, a1, c2, a2, 1.0)
            # If min_d < 0, they overlap by -min_d
            violation = max(0, -min_d)
            disjoint_violations.append(violation)
    
    # Combine: we want all violations to be zero
    # Use log-sum-exp over all violations for a smooth objective
    all_violations = containment_violations + disjoint_violations
    if len(all_violations) == 0:
        return 0.0, True
    
    max_v = max(all_violations)
    if max_v < 1e-10:
        return 0.0, True
    
    # Log-sum-exp for smooth max
    log_sum = 0.0
    for v in all_violations:
        if v > 0:
            log_sum += math.exp(beta * v)
    if log_sum > 0:
        smooth_obj = math.log(log_sum) / beta
    else:
        smooth_obj = 0.0
    
    is_valid = max_v < 1e-6
    return smooth_obj, is_valid

def initial_config():
    """
    Generate an initial configuration for the 12 hexagons.
    Use a grid-like arrangement to start with a reasonable guess.
    """
    # Start with a hexagonal grid arrangement
    # For 12 hexagons, we can use a 3x4 or 4x3 arrangement
    # Let's use a centered arrangement
    configs = []
    
    # Hex grid positions for 12 hexagons
    # Rows: -1, 0, 1 with offsets
    positions = []
    # Row 0: 4 hexagons
    for i in range(-2, 2):
        positions.append((i * 2.0, 0.0))
    # Row 1: 4 hexagons, offset by 1 in x, sqrt(3) in y
    for i in range(-2, 2):
        positions.append((i * 2.0 + 1.0, math.sqrt(3)))
    # Row 2: 4 hexagons
    for i in range(-2, 2):
        positions.append((i * 2.0, 2 * math.sqrt(3)))
    
    # Actually, let's use a simpler grid
    # 3 rows of 4
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * 2.0 - 3.0
            y = row * math.sqrt(3) - math.sqrt(3)
            positions.append((x, y))
    
    params = []
    for x, y in positions:
        params.extend([x, y, 0.0])
    
    return np.array(params)

def simulated_annealing(params, outer_c, outer_side, outer_angle, beta=50.0, max_iter=5000, 
                         t0=1.0, t1=0.01, cooling=0.999):
    """
    Simulated annealing to find a good configuration.
    """
    current = params.copy()
    current_obj, _ = compute_objective(current, outer_c, outer_side, outer_angle, beta)
    best = current.copy()
    best_obj = current_obj
    
    temp = t0
    for i in range(max_iter):
        if not check_time():
            break
        
        # Generate a random perturbation
        new = current.copy()
        idx = np.random.randint(0, len(new))
        delta = np.random.randn() * 0.5
        new[idx] += delta
        
        # Keep angles in [0, 360)
        if idx % 3 == 2:
            new[idx] = new[idx] % 360.0
        
        new_obj, _ = compute_objective(new, outer_c, outer_side, outer_angle, beta)
        
        # Accept if better or with probability exp(-(new_obj - current_obj) / temp)
        if new_obj < current_obj:
            current = new
            current_obj = new_obj
            if new_obj < best_obj:
                best = new.copy()
                best_obj = new_obj
        else:
            delta_obj = new_obj - current_obj
            if delta_obj < 0:
                current = new
                current_obj = new_obj
            else:
                prob = math.exp(-delta_obj / temp) if temp > 1e-10 else 0
                if np.random.rand() < prob:
                    current = new
                    current_obj = new_obj
        
        temp *= cooling
        if temp < t1:
            temp = t1
    
    return best, best_obj

def polish_with_lbfgs(params, outer_c, outer_side, outer_angle, beta=50.0, maxiter=1000):
    """
    Polish the configuration using L-BFGS-B.
    """
    def objective(p):
        obj, _ = compute_objective(p, outer_c, outer_side, outer_angle, beta)
        return obj
    
    result = minimize(objective, params, method='L-BFGS-B', 
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    
    return result.x, result.fun

def final_polish(params, outer_c, outer_side, outer_angle, beta=100.0, maxiter=2000):
    """
    Final high-precision polish with higher beta and tighter tolerances.
    """
    def objective(p):
        obj, _ = compute_objective(p, outer_c, outer_side, outer_angle, beta)
        return obj
    
    result = minimize(objective, params, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    
    return result.x, result.fun

def find_min_outer_side(params, outer_c, outer_angle):
    """
    Given inner hexagon positions, find the minimum outer side length
    that contains all inner hexagons.
    """
    # Binary search on outer side length
    lo = 1.0
    hi = 20.0
    
    for _ in range(50):
        mid = (lo + hi) / 2
        valid = True
        for i in range(N):
            cx, cy, angle = params[3*i:3*i+3]
            if not hexagon_contains_hexagon(outer_c, mid, outer_angle, 
                                            (cx, cy), 1.0, angle):
                valid = False
                break
        if valid:
            hi = mid
        else:
            lo = mid
    
    return hi

def construct():
    """
    Main entry point. Returns:
    inner: list of [x, y, angle_degrees] for N hexagons
    center: (x, y) of outer hexagon
    side: outer hexagon side length
    angle: outer hexagon rotation angle
    """
    global START_TIME
    START_TIME = time.time()
    
    # Start with a reasonable outer side length estimate
    # For 12 unit hexagons, a rough estimate is around 8-10
    outer_side = 10.0
    outer_c = (0.0, 0.0)
    outer_angle = 0.0
    
    # Initial configuration
    params = initial_config()
    
    # Stage 1: Simulated annealing
    if check_time():
        params, obj = simulated_annealing(params, outer_c, outer_side, outer_angle, 
                                           beta=50.0, max_iter=3000,
                                           t0=1.0, t1=0.01, cooling=0.9995)
    
    # Stage 2: L-BFGS-B polish
    if check_time():
        params, obj = polish_with_lbfgs(params, outer_c, outer_side, outer_angle,
                                         beta=50.0, maxiter=500)
    
    # Stage 3: Final high-precision polish
    if check_time():
        params, obj = final_polish(params, outer_c, outer_side, outer_angle,
                                    beta=100.0, maxiter=1000)
    
    # Find the minimum outer side length for this configuration
    if check_time():
        outer_side = find_min_outer_side(params, outer_c, outer_angle)
    
    # Construct the inner list
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        angle = params[3*i + 2] % 360.0
        inner.append([x, y, angle])
    
    return inner, (outer_c[0], outer_c[1]), outer_side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

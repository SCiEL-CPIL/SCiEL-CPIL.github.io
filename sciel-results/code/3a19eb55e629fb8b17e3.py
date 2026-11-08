"""Packing unit regular hexagons inside a regular hexagon using trust-region optimization."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(center, side, angle_deg):
    """Get the 6 vertices of a regular hexagon."""
    cx, cy = center
    theta = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = theta + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append([x, y])
    return np.array(verts)

def _hex_support(center, side, angle_deg, direction):
    """
    Compute the support function of a regular hexagon in a given direction.
    The support function h(u) = max_{p in hex} p·u
    For a regular hexagon with center c, side s, rotation angle theta,
    the support function is s * max_{i=0..5} cos(theta + i*pi/3 - phi) where phi = atan2(u_y, u_x)
    Actually for a regular hexagon (which is a zonotope), the support function in direction u
    is s * max_{i=0..5} |cos(angle_between_u_and_edge_normal_i)|
    
    More precisely: A regular hexagon with side length s can be represented as the Minkowski sum
    of 3 line segments. The support function is the sum of support functions of these segments.
    
    The 3 edge normals of a regular hexagon (with one edge horizontal) are at angles 0, pi/3, 2pi/3.
    Wait, let me think more carefully.
    
    A regular hexagon with side length s centered at origin, with vertices at angles 0, 60, 120, 180, 240, 300 degrees
    (for angle=0). The edge normals point at angles 30, 90, 150, 210, 270, 330 degrees.
    
    Actually, for a regular hexagon, the support function in direction u (unit vector) is:
    h(u) = s * max_{k=0,1,2} |cos(alpha_k)|
    where alpha_k is the angle between u and the k-th pair of parallel edge normals.
    
    For a hexagon with rotation theta, the edge normal directions are at theta + pi/6 + k*pi/3 for k=0..5,
    but due to symmetry, we only need 3 directions: theta + pi/6, theta + pi/6 + pi/3, theta + pi/6 + 2*pi/3.
    
    The support function is: s * (|cos(beta_0)| + |cos(beta_1)| + |cos(beta_2)|) / 1
    where beta_k = angle_between(u, normal_k)
    
    Wait, that's not right either. Let me reconsider.
    
    A regular hexagon with side length s is the Minkowski sum of 3 line segments of length s,
    oriented at angles 0, pi/3, 2pi/3 (for the standard orientation).
    
    The support function of a line segment of length L in direction u is (L/2) * |cos(angle_between_segment_and_u)|.
    
    So for a hexagon that is the Minkowski sum of 3 segments of length s at angles theta, theta+pi/3, theta+2pi/3:
    h(u) = (s/2) * (|cos(phi - theta)| + |cos(phi - theta - pi/3)| + |cos(phi - theta - 2pi/3)|)
    where phi = atan2(u_y, u_x)
    
    This is correct.
    """
    phi = math.atan2(direction[1], direction[0])
    theta = math.radians(angle_deg)
    total = 0.0
    for k in range(3):
        angle_diff = phi - theta - k * math.pi / 3
        total += abs(math.cos(angle_diff))
    return side * 0.5 * total

def _min_distance_hex_hex(c1, s1, a1, c2, s2, a2):
    """
    Compute the minimum distance between two regular hexagons.
    If they overlap, return a negative value (penalty).
    We use the fact that for two convex polygons, the minimum distance can be found
    by checking all pairs of edges/vertices, but for hexagons we can use support functions.
    
    Actually, a simpler approach: the minimum distance between two convex sets A and B is
    min_{p in A, q in B} ||p - q||. If they intersect, this is 0 (or negative if we define it as
    the maximum amount of separation needed).
    
    For optimization purposes, we want a smooth approximation. Let's use the following:
    The two hexagons are separated iff for all directions u, 
    h_A(u) + h_B(-u) <= -distance... 
    
    Actually, let's just compute the actual minimum distance by sampling.
    For two convex polygons, the minimum distance is either:
    1. Between a vertex of one and an edge of the other, or
    2. Between two parallel edges.
    
    For simplicity and differentiability, let's use a different approach:
    We'll compute the minimum distance by checking all vertex-edge pairs.
    """
    verts1 = _hex_vertices(c1, s1, a1)
    verts2 = _hex_vertices(c2, s2, a2)
    
    min_dist = float('inf')
    
    # Check vertex-to-edge distances
    for v in verts1:
        for i in range(6):
            e1 = verts2[i]
            e2 = verts2[(i + 1) % 6]
            # Distance from point v to segment e1-e2
            d = _point_segment_distance(v, e1, e2)
            min_dist = min(min_dist, d)
    
    for v in verts2:
        for i in range(6):
            e1 = verts1[i]
            e2 = verts1[(i + 1) % 6]
            d = _point_segment_distance(v, e1, e2)
            min_dist = min(min_dist, d)
    
    return min_dist

def _point_segment_distance(p, a, b):
    """Compute distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    if np.dot(ab, ab) == 0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def _hex_in_hex_distance(inner_center, inner_side, inner_angle, outer_center, outer_side, outer_angle):
    """
    Compute how much the inner hexagon is inside the outer hexagon.
    Return a positive value if the inner hexagon is fully inside,
    negative if it protrudes.
    
    The inner hexagon is inside the outer hexagon iff for all 6 edge normals of the outer hexagon,
    the support function of the inner hexagon in that normal direction (relative to outer center)
    is <= the support function of the outer hexagon in that direction.
    
    More precisely: for each of the 6 edge normals n_k of the outer hexagon,
    we need: (inner_center - outer_center)·n_k + h_inner(n_k) <= h_outer(n_k)
    
    The "distance" (margin) for edge k is: h_outer(n_k) - h_inner(n_k) - (inner_center - outer_center)·n_k
    The inner hexagon is inside iff all margins are >= 0.
    We return the minimum margin (negative means violation).
    """
    outer_normals = []
    for k in range(6):
        angle = math.radians(outer_angle) + math.pi / 6 + k * math.pi / 3
        n = np.array([math.cos(angle), math.sin(angle)])
        outer_normals.append(n)
    
    min_margin = float('inf')
    for n in outer_normals:
        h_outer = _hex_support((0, 0), outer_side, 0)  # Support of outer hex at origin in direction n
        # Wait, the support function depends on the hexagon's own orientation.
        # h_outer(n) for the outer hexagon with center outer_center, side outer_side, rotation outer_angle
        # = (outer_center)·n + _hex_support((0,0), outer_side, outer_angle, n)
        # But we're measuring relative to outer_center, so we use:
        # margin = _hex_support((0,0), outer_side, outer_angle, n) - _hex_support((0,0), inner_side, inner_angle, n) - (inner_center - outer_center)·n
        
        h_out = _hex_support((0, 0), outer_side, outer_angle, n)
        h_in = _hex_support((0, 0), inner_side, inner_angle, n)
        rel_center = np.array(inner_center) - np.array(outer_center)
        margin = h_out - h_in - np.dot(rel_center, n)
        min_margin = min(min_margin, margin)
    
    return min_margin

def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Returns (inner, center, side, angle) where:
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (x, y) of outer hexagon center
      side: side length of outer hexagon
      angle: rotation angle of outer hexagon in degrees
    """
    n = N
    unit_side = 1.0
    
    # Initial guess: place hexagons in a rough grid pattern
    # For N hexagons, arrange them in a hexagonal packing pattern
    # Let's start with a simple arrangement and let the optimizer find the optimum
    
    # Initial positions: spread them out in a hexagonal lattice pattern
    positions = []
    angles = [0.0] * n
    
    # Arrange in a hexagonal pattern
    # For N=11, we can do:
    # Center + 6 around + 4 more
    # Or just use a grid-like arrangement
    
    # Let's use a simple initial guess: place them in a row with some spacing
    # The minimum distance between centers of two non-overlapping unit hexagons is at least 2*sqrt(3)/2 = sqrt(3) ≈ 1.732
    # (distance between parallel edges)
    
    spacing = 1.9  # Slightly more than the minimum
    
    # Arrange in a hexagonal pattern
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    # Total: 11
    
    row_configs = [(3, 0), (4, 1), (4, 2)]
    idx = 0
    for count, row_idx in row_configs:
        y_pos = row_idx * spacing * math.sqrt(3) / 2
        x_offset = (row_idx % 2) * spacing / 2
        for j in range(count):
            if idx >= n:
                break
            x_pos = (j - (count - 1) / 2.0) * spacing + x_offset
            positions.append([x_pos, y_pos])
            idx += 1
    
    # Fill remaining with random-ish positions
    while len(positions) < n:
        positions.append([float(len(positions)) * spacing, 0.0])
    
    positions = np.array(positions[:n])
    
    # Define the optimization problem
    # Variables: [x_0, y_0, a_0, x_1, y_1, a_1, ..., x_{n-1}, y_{n-1}, a_{n-1}, outer_side]
    # We fix the outer center at (0,0) and outer angle at 0 for simplicity
    # (due to symmetry, we can always rotate/translate the whole configuration)
    
    def objective(vars):
        """Minimize outer_side, with penalties for constraint violations."""
        outer_side = vars[-1]
        if outer_side < 2.0:
            return 1000.0 + outer_side
        
        inner_params = vars[:-1].reshape(n, 3)
        
        # Penalty for inner hexagons not being inside outer
        penalty = 0.0
        for i in range(n):
            cx, cy, ang = inner_params[i]
            margin = _hex_in_hex_distance((cx, cy), unit_side, ang, (0.0, 0.0), outer_side, 0.0)
            if margin < 0:
                penalty += margin * margin * 100.0
        
        # Penalty for inner hexagons overlapping each other
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = inner_params[i], inner_params[j]
                dist = _min_distance_hex_hex(
                    (ci[0], ci[1]), unit_side, ci[2],
                    (cj[0], cj[1]), unit_side, cj[2]
                )
                if dist < 0.01:
                    penalty += (0.01 - dist) * (0.01 - dist) * 100.0
        
        return outer_side + penalty
    
    # Initial variables
    x0 = np.zeros(n * 3 + 1)
    for i in range(n):
        x0[i * 3] = positions[i, 0]
        x0[i * 3 + 1] = positions[i, 1]
        x0[i * 3 + 2] = 0.0
    x0[-1] = 5.0  # Initial outer side guess
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((-20.0, 20.0))  # x
        bounds.append((-20.0, 20.0))  # y
        bounds.append((-180.0, 180.0))  # angle
    bounds.append((2.0, 30.0))  # outer side
    
    # Try to optimize
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.success or result.fun < objective(x0):
            best_vars = result.x
        else:
            best_vars = x0
    except Exception:
        best_vars = x0
    
    # Try a second pass with better initial guess if needed
    # Also try different initial configurations
    
    best_score = objective(best_vars)
    best_vars_final = best_vars
    
    # Try a few different initial configurations
    for trial in range(3):
        np.random.seed(42 + trial)
        pos_trial = np.random.uniform(-3, 3, size=(n, 2))
        ang_trial = np.random.uniform(-30, 30, size=n)
        
        x0_trial = np.zeros(n * 3 + 1)
        for i in range(n):
            x0_trial[i * 3] = pos_trial[i, 0]
            x0_trial[i * 3 + 1] = pos_trial[i, 1]
            x0_trial[i * 3 + 2] = ang_trial[i]
        x0_trial[-1] = 4.0 + trial * 0.5
        
        try:
            result = minimize(objective, x0_trial, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_vars_final = result.x
        except Exception:
            pass
    
    # Extract solution
    outer_side = max(2.0, best_vars_final[-1])
    inner = []
    for i in range(n):
        cx = best_vars_final[i * 3]
        cy = best_vars_final[i * 3 + 1]
        ang = best_vars_final[i * 3 + 2]
        inner.append([float(cx), float(cy), float(ang)])
    
    return inner, (0.0, 0.0), float(outer_side), 0.0
# EVOLVE-BLOCK-END

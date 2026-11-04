"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(side, angle_deg, center):
    """Return vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.pi * np.arange(6) / 3.0
    verts = side * np.column_stack([np.cos(angles), np.sin(angles)])
    return verts + np.array(center)

def _support_function_hexagon(point, side, angle_deg):
    """
    Support function of a regular hexagon centered at origin with given side and angle,
    evaluated in direction 'point' (normalized).
    For a regular hexagon with side length s, the support in direction theta is:
    s * (cos(theta - a) * cos(pi/6) / (cos(theta - a - pi/3) * cos(pi/6) + cos(theta - a) * sin(pi/3) + ...))
    
    Actually, let's use the formula: for a regular hexagon with circumradius R = side,
    the support function h(theta) = R * cos(pi/6) / cos((theta - a) mod (pi/3) - pi/6)
    where a is the rotation angle.
    """
    # Regular hexagon has circumradius = side length
    # The support function in direction phi (relative to hexagon orientation) is:
    # h(phi) = R * cos(pi/6) / cos(phi - pi/6) where phi is in [-pi/6, pi/6]
    # More generally, for direction theta and hexagon rotated by angle a:
    # Let d = (theta - a) mod (2*pi/3) - pi/3  ... this gets complicated.
    
    # Simpler approach: use the fact that a regular hexagon with side s has
    # inradius = s * sqrt(3)/2 and circumradius = s
    # The support function in direction theta for a hexagon with one vertex at angle a:
    # The hexagon is the intersection of 6 half-planes.
    # Each side is at distance inradius = s*sqrt(3)/2 from center.
    # The normal to side i is at angle a + i*pi/3
    # The support in direction theta is max over i of (inradius * cos(theta - (a + i*pi/3)))
    # But actually, the support function is:
    # h(theta) = max_i { (s*sqrt(3)/2) * cos(theta - a - i*pi/3) } for i=0..5
    # Wait, that's not right either. Let me think more carefully.
    
    # A regular hexagon with side length s, centered at origin, with one vertex at angle 0:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The sides are perpendicular to directions 30, 90, 150, 210, 270, 330 degrees
    # Each side is at distance inradius = s * sqrt(3)/2 from center
    # So the hexagon = { x : dot(x, n_i) <= s*sqrt(3)/2 for all i }
    # where n_i are unit vectors at angles 30 + i*60 degrees
    # The support function h(theta) = max_{x in hex} dot(x, theta) = max_i { s*sqrt(3)/2 * cos(theta - (30 + i*60)) }
    # Actually no, the support function of a polytope defined by half-planes dot(x, n_i) <= d_i
    # is h(theta) = max_i { d_i * cos(theta - angle(n_i)) } ... no that's not right.
    
    # Let me just use: h(theta) = max_{vertices v} dot(v, theta)
    # But that requires computing vertices. For efficiency in optimization, let's precompute.
    
    # For a regular hexagon with side s, the support function in direction theta (unit vector)
    # when the hexagon is rotated by angle a:
    # The hexagon has 6 sides with outward normals at angles a + pi/6 + k*pi/3 for k=0..5
    # Each side is at distance s*sqrt(3)/2 from center
    # The support function is: h(theta) = s*sqrt(3)/2 * max_k { cos(theta - a - pi/6 - k*pi/3) }
    # But cos is not max, we need to be careful. Actually:
    # h(theta) = s*sqrt(3)/2 * cos( (theta - a - pi/6) mod (pi/3) - pi/6 )
    # Let me verify: if theta = a + pi/6 (normal to a side), then h = s*sqrt(3)/2 * cos(0) = s*sqrt(3)/2. Correct.
    # If theta = a (toward a vertex), then theta - a - pi/6 = -pi/6, mod pi/3 = pi/6 - pi/6 = 0? 
    # (-pi/6) mod (pi/3): -pi/6 + pi/3 = pi/6, so mod gives pi/6. Then pi/6 - pi/6 = 0, cos(0)=1.
    # h = s*sqrt(3)/2. But the vertex is at distance s from center, so h should be s.
    # So this formula is wrong.
    
    # Let me re-derive. The vertices of a regular hexagon with side s, center at origin, 
    # one vertex at angle 0:
    # v_k = s * (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    # The support in direction theta is max_k dot(v_k, theta) = s * max_k cos(theta - k*pi/3)
    # For a rotated hexagon by angle a: v_k = s * (cos(a + k*pi/3), sin(a + k*pi/3))
    # h(theta) = s * max_k cos(theta - a - k*pi/3)
    
    # This is correct! Let's use this.
    theta = np.arctan2(point[1], point[0])
    rel = theta - np.radians(angle_deg)
    # max_k cos(rel - k*pi/3) for k=0..5
    # This equals cos( (rel mod (2*pi/3)) - pi/3 ) ... let's just compute all 6
    ks = np.arange(6)
    vals = np.cos(rel - ks * np.pi / 3.0)
    return side * np.max(vals)

def _min_distance_between_hexagons(center1, angle1, center2, angle2, side=1.0):
    """
    Compute the minimum distance between two hexagons (0 if overlapping).
    Using support functions: the distance between two convex sets A and B is:
    min_{x in A, y in B} ||x - y||
    This is hard to compute directly. Instead, we can use:
    Two convex sets don't intersect iff there exists a direction u such that
    h_A(u) + h_B(-u) < 0 ... no, that's for separation.
    
    Actually, for two convex polygons, the minimum distance can be found by checking
    vertex-vertex, vertex-edge, edge-edge distances. But that's complex.
    
    Alternative: use the fact that for two convex sets, the minimum distance is:
    min_{u unit} { h_A(u) - h_B(u) } ... no.
    
    Let me use a different approach: the Minkowski difference.
    Two convex sets A and B intersect iff 0 is in A - B (Minkowski difference).
    The Minkowski difference of two hexagons is a convex polygon (up to 12 sides).
    We can check if the origin is inside this polygon.
    
    But for optimization, we need a smooth-ish measure of "how much they overlap".
    Let's use: the minimum over all directions u of { h_{A-B}(u) } where A-B is Minkowski diff.
    If min < 0, they don't intersect (well, this isn't quite right).
    
    Actually, let's just compute the actual minimum distance between the two hexagons
    by checking all vertex-vertex, vertex-edge pairs. For 6 vertices each, that's manageable.
    """
    verts1 = _hex_vertices(side, angle1, center1)
    verts2 = _hex_vertices(side, angle2, center2)
    
    min_dist = float('inf')
    
    # Check vertex-vertex
    for v1 in verts1:
        for v2 in verts2:
            d = np.linalg.norm(v1 - v2)
            min_dist = min(min_dist, d)
    
    # Check vertex-edge (vertex of 1 to edges of 2, and vice versa)
    for i in range(6):
        v1 = verts1[i]
        e2_start = verts2[i]
        e2_end = verts2[(i + 1) % 6]
        # Distance from point v1 to segment e2_start-e2_end
        seg = e2_end - e2_start
        t = np.dot(v1 - e2_start, seg) / np.dot(seg, seg)
        t = np.clip(t, 0, 1)
        closest = e2_start + t * seg
        d = np.linalg.norm(v1 - closest)
        min_dist = min(min_dist, d)
    
    for i in range(6):
        v2 = verts2[i]
        e1_start = verts1[i]
        e1_end = verts1[(i + 1) % 6]
        seg = e1_end - e1_start
        t = np.dot(v2 - e1_start, seg) / np.dot(seg, seg)
        t = np.clip(t, 0, 1)
        closest = e1_start + t * seg
        d = np.linalg.norm(v2 - closest)
        min_dist = min(min_dist, d)
    
    return min_dist

def _hexagon_in_outer(center, angle_deg, outer_center, outer_side, outer_angle):
    """
    Check if a unit hexagon at (center, angle_deg) is inside the outer hexagon.
    Returns True if inside, False otherwise.
    Uses support functions: the inner hex is inside the outer iff for all directions u,
    h_inner(u) + ||center - outer_center|| * ||u|| <= h_outer(u) ... 
    Actually, a set A is inside set B (both convex) iff for all directions u:
    h_A(u) <= h_B(u) + h_{B-A}(0) ... this is getting complicated.
    
    Simpler: the inner hex is inside the outer iff all vertices of the inner hex
    are inside the outer hex. For convex polygons, this is sufficient.
    """
    inner_verts = _hex_vertices(1.0, angle_deg, center)
    outer_verts = _hex_vertices(outer_side, outer_angle, outer_center)
    
    # Check if all inner vertices are inside outer hex
    # A point is inside a convex polygon iff it's on the same side of each edge
    for v in inner_verts:
        inside = True
        for i in range(6):
            e1 = outer_verts[i]
            e2 = outer_verts[(i + 1) % 6]
            # Normal pointing inward
            edge = e2 - e1
            normal = np.array([-edge[1], edge[0]])
            # Check if v is on the same side as the center
            center_vec = np.array(outer_center) - e1
            v_vec = v - e1
            if np.dot(normal, center_vec) * np.dot(normal, v_vec) < 0:
                inside = False
                break
        if not inside:
            return False
    return True

def _penalty_for_violations(inner_positions, outer_center, outer_side, outer_angle):
    """
    Compute a penalty for constraint violations.
    inner_positions: list of (x, y, angle_deg)
    Returns a scalar penalty (0 if all constraints satisfied).
    """
    penalty = 0.0
    
    # Check outer containment
    for i, (x, y, ang) in enumerate(inner_positions):
        if not _hexagon_in_outer((x, y), ang, outer_center, outer_side, outer_angle):
            penalty += 100.0
    
    # Check pairwise non-overlap
    for i in range(len(inner_positions)):
        for j in range(i + 1, len(inner_positions)):
            x1, y1, a1 = inner_positions[i]
            x2, y2, a2 = inner_positions[j]
            dist = _min_distance_between_hexagons((x1, y1), a1, (x2, y2), a2)
            if dist < 0.01:  # Small tolerance
                penalty += 100.0 * (0.01 - dist)
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initial guess: place hexagons in a ring or grid pattern
    # Let's try a hexagonal packing pattern
    # For 12 hexagons, we can do 2 rows of 6, or a ring of 12, etc.
    
    # Let's start with a reasonable initial configuration
    # Place them in a 3x4 grid-like pattern with hexagonal offsets
    initial_positions = []
    # 2 rows of 6
    for row in range(2):
        for col in range(6):
            x = col * 1.8 - 4.5  # Spread out
            y = row * 1.732 - 0.866
            angle = 0.0
            initial_positions.append([x, y, angle])
    
    # Convert to numpy array for optimization
    x0 = np.array([v for pos in initial_positions for v in pos])
    
    def objective(params):
        """Objective: minimize outer side length needed to contain all hexagons."""
        # Reconstruct positions
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            positions.append((x, y, ang))
        
        # Estimate the outer side length needed
        # The outer hex must contain all inner hexes
        # Find the bounding hexagon: for each direction, find the max support
        # The outer hex side length is related to the max support in 6 directions
        
        # For a hexagon with side s and orientation a, the support in direction theta is:
        # s * max_k cos(theta - a - k*pi/3)
        # We need: for all inner hexes and all directions, the inner is inside the outer
        # This is equivalent to: for each direction theta, 
        # max_i { h_inner_i(theta) + dot(center_i, theta) } <= h_outer(theta)
        # where h_inner_i(theta) is the support of the i-th inner hex in direction theta
        # (relative to its center)
        
        # The minimum outer side s such that this holds for all theta is:
        # s >= max_theta { max_i { h_inner_i(theta) + dot(center_i, theta) } / h_unit_hex(theta) }
        # where h_unit_hex(theta) is the support of a unit hexagon in direction theta
        
        # Let's sample theta at many angles
        thetas = np.linspace(0, 2*np.pi, 361)
        max_ratio = 0.0
        
        for theta in thetas:
            u = np.array([np.cos(theta), np.sin(theta)])
            # For each inner hex, compute its support in direction u
            max_support = -float('inf')
            for i, (x, y, ang) in enumerate(positions):
                # Support of inner hex i in direction u
                # = h_unit(u relative to hex orientation) + dot(center, u)
                rel_theta = theta - np.radians(ang)
                ks = np.arange(6)
                h_unit = np.max(np.cos(rel_theta - ks * np.pi / 3.0))
                support = 1.0 * h_unit + x * u[0] + y * u[1]
                max_support = max(max_support, support)
            
            # Support of unit hexagon in direction u (for the outer, with side 1)
            rel_theta_outer = theta - 0.0  # outer angle = 0 for now
            ks = np.arange(6)
            h_outer_unit = np.max(np.cos(rel_theta_outer - ks * np.pi / 3.0))
            
            if h_outer_unit > 1e-6:
                ratio = max_support / h_outer_unit
                max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    # Use Nelder-Mead optimization
    best_result = None
    best_score = float('inf')
    
    # Multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial solution
            x_start = x0 + np.random.randn(3*N) * 0.5
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to initial
        best_result = x0
        best_score = objective(x0)
    
    # Reconstruct final positions
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        ang = best_result[3*i+2]
        inner.append([x, y, ang])
    
    # Compute the outer side length
    outer_side = best_score
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END

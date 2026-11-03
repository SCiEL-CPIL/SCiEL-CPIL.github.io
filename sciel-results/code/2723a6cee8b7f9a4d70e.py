"""Packing unit hexagons in a regular hexagon using lattice-based search."""
import numpy as np
import time
import math
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = math.radians(-angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * math.cos(a) + dy * math.sin(a)
    ry = -dx * math.sin(a) + dy * math.cos(a)
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Inradius = side * sqrt(3)/2
    inr = side * math.sqrt(3) / 2
    # Check against 6 half-planes
    for k in range(6):
        theta = k * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        if rx * nx + ry * ny > inr:
            return False
    return True

def _hex_in_hex(inner_c, inner_a, outer_c, outer_s, outer_a):
    """Check if unit hexagon (center inner_c, angle inner_a) is inside outer hexagon."""
    iv = _hex_vertices(inner_c[0], inner_c[1], 1.0, inner_a)
    for vx, vy in iv:
        if not _point_in_hex(vx, vy, outer_c[0], outer_c[1], outer_s, outer_a):
            return False
    return True

def _hex_hex_overlap(c1, a1, c2, a2, eps=1e-7):
    """Check if two unit hexagons overlap (distance between centers < 2 - eps means potential overlap)."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist >= 2.0 - eps:
        return False  # definitely no overlap
    # Need to check if the hexagons actually overlap
    # For simplicity with lattice packing, if dist < 2 they might overlap
    # But for exact check, we'd need polygon intersection
    # For now, use conservative: if dist < 2 - small, assume overlap
    # Actually for regular hexagons, the minimum distance for no overlap is 2 (inradius*2)
    # Wait: inradius of unit hexagon = sqrt(3)/2, so two hexagons touch when center distance = 2*inradius = sqrt(3)
    # No wait. Let me reconsider.
    # A regular hexagon with side 1 has:
    #   - circumradius = 1
    #   - inradius = sqrt(3)/2
    # Two hexagons are disjoint if the distance between centers is at least 2*inradius = sqrt(3)
    # IF they are oriented the same way and aligned properly.
    # Actually, the minimum distance for two congruent hexagons to be disjoint depends on orientation.
    # For two hexagons with the same orientation, the minimum center distance for disjointness is 2 * inradius = sqrt(3)
    # But if rotated, it could be different.
    # For a safe check: if dist < sqrt(3) - eps, they definitely overlap (for same orientation)
    # Let's use a more careful check
    if dist < math.sqrt(3) - eps:
        return True  # likely overlap for similar orientations
    # For different orientations, the gap could be smaller
    # Conservative: assume overlap if dist < 1.7 (slightly less than sqrt(3)≈1.732)
    if dist < 1.7:
        # Do a more detailed check: check if any vertex of one is inside the other
        v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
        v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
        for vx, vy in v1:
            if _point_in_hex(vx, vy, c2[0], c2[1], 1.0, a2):
                return True
        for vx, vy in v2:
            if _point_in_hex(vx, vy, c1[0], c1[1], 1.0, a1):
                return True
        return False
    return False

def _packing_valid(positions, angles, outer_c, outer_s, outer_a, eps=1e-6):
    """Check if the packing is valid: all inner hexagons inside outer, no overlaps."""
    # Check all inner hexagons are inside outer
    for i in range(N):
        if not _hex_in_hex(positions[i], angles[i], outer_c, outer_s, outer_a):
            return False
    # Check no overlaps between inner hexagons
    for i in range(N):
        for j in range(i+1, N):
            if _hex_hex_overlap(positions[i], angles[i], positions[j], angles[j], eps):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s margin
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_c = (0.0, 0.0)
    best_outer_a = 0.0
    
    # Strategy: use hexagonal lattice packing
    # For N hexagons, try different lattice configurations
    # The hexagonal lattice has basis vectors at 60 degrees
    
    # First, try a compact hexagonal arrangement
    # For N=11, we can use a 4-3-4 or similar arrangement
    
    # Approach: Place hexagons on a hexagonal lattice with spacing d
    # Then find the minimum enclosing hexagon
    
    # Hexagonal lattice points for N points
    # Generate candidate lattice configurations
    
    # Method 1: Hexagonal cluster (center + rings)
    # For N=11: center (1) + ring 1 (6) + partial ring 2 (4) = 11
    # Or: use a rectangular-ish grid
    
    # Let's try multiple arrangements
    
    arrangements = []
    
    # Arrangement 1: Hexagonal cluster pattern
    # Center at origin, ring 1 at distance d in 6 directions, ring 2 partial
    # For N=11: 1 + 6 + 4
    # Ring 1: 6 hexagons at distance d, angles 0,60,120,180,240,300
    # Ring 2: 4 hexagons at distance 2d, angles 0,120,180,300 (or similar)
    
    # Actually, let me just do a grid search over d and arrangement type
    
    # For a hexagonal lattice with spacing d:
    # Basis vectors: (d, 0) and (d/2, d*sqrt(3)/2)
    # Lattice points: i*(d,0) + j*(d/2, d*sqrt(3)/2)
    
    # Generate N lattice points in various compact shapes
    # Then compute the minimum enclosing hexagon
    
    def _min_enclosing_hexagon(points, angle=0.0):
        """Find the minimum side length of a regular hexagon (with given orientation) containing all points."""
        # A regular hexagon with center (cx,cy), side s, angle a contains point (px,py)
        # if the point is within the inradius in all 6 normal directions
        # Inradius = s * sqrt(3)/2
        # The constraint is: for each of 6 directions (at angles a + k*pi/3),
        #   (px-cx)*cos(theta) + (py-cy)*sin(theta) <= s*sqrt(3)/2
        # We want to minimize s subject to these constraints
        # This is a linear program in s and (cx,cy)
        # For fixed angle, we can solve via:
        #   s >= max over all points and all 6 directions of [(px-cx)*cos(theta_k) + (py-cy)*sin(theta_k)] / (sqrt(3)/2)
        #   s >= 0
        # To minimize s, we choose (cx,cy) to minimize the max of these expressions
        # This is a convex optimization problem
        
        # Simple approach: the optimal center is the centroid of the points (for symmetric cases)
        # Or we can use a more careful method
        
        # For a regular hexagon, the minimum enclosing hexagon of a set of points
        # can be found by noting that the hexagon is the intersection of 6 half-planes
        # The optimal placement minimizes the maximum "support function" value
        
        # Let's use the fact that for a given orientation, the problem reduces to
        # finding the minimum s such that there exists (cx,cy) with
        #   max_i max_k [ (pi_x - cx)*cos(theta_k) + (pi_y - cy)*sin(theta_k) ] <= s*sqrt(3)/2
        #   min_i min_k [ (pi_x - cx)*cos(theta_k) + (pi_y - cy)*sin(theta_k) ] >= -s*sqrt(3)/2
        
        # This means:
        #   s >= (max_k [max_i (pi_x*cos(theta_k)+pi_y*sin(theta_k)) - min_i (pi_x*cos(theta_k)+pi_y*sin(theta_k))]) / (2 * sqrt(3)/2)
        #   s >= max_k [max_i (pi_x*cos(theta_k)+pi_y*sin(theta_k))] / (sqrt(3)/2) - cx*cos(theta_k) - cy*sin(theta_k) ... 
        # 
        # Actually, the width of the point set in direction theta is:
        #   w(theta) = max_i (pi_x*cos(theta)+pi_y*sin(theta)) - min_i (pi_x*cos(theta)+pi_y*sin(theta))
        # The hexagon needs inradius >= max_k w(theta_k)/2 where theta_k are the 6 normals
        # Wait, that's not quite right either.
        
        # Let me think more carefully. A regular hexagon with inradius r has the property that
        # for each of its 6 face normals theta_k, the support function in that direction is r.
        # So the hexagon contains all points iff for all k and all i:
        #   (pi - c) · n_k <= r  and  (pi - c) · (-n_k) <= r
        # i.e.,  |(pi - c) · n_k| <= r  for all k, i
        # where n_k are the 6 unit normals.
        
        # So we need: max_i |(pi - c) · n_k| <= r  for all k
        # i.e., r >= max_k max_i |(pi - c) · n_k|
        
        # To minimize r, we choose c to minimize max_k max_i |(pi - c) · n_k|
        # This is a convex problem. For 6 directions, it's like finding the minimum enclosing
        # hexagon.
        
        # Simple heuristic: try the centroid as center, and also try a few other centers.
        
        pts = np.array(points)
        n_pts = len(pts)
        
        # Try centroid
        c_centroid = np.mean(pts, axis=0)
        
        # Compute r for centroid
        thetas = [angle + k * math.pi / 3 for k in range(6)]
        r_centroid = 0
        for theta in thetas:
            n = np.array([math.cos(theta), math.sin(theta)])
            projections = pts @ n
            center_proj = c_centroid @ n
            r_centroid = max(r_centroid, np.max(np.abs(projections - center_proj)))
        
        # Also try: the center that minimizes the max distance to vertices
        # Or just use centroid for now
        
        # For a regular hexagon, the inradius r = s * sqrt(3)/2, so s = 2r/sqrt(3)
        s = 2 * r_centroid / math.sqrt(3)
        
        return s, c_centroid
    
    # Generate lattice-based arrangements
    # For a hexagonal lattice with spacing d, generate N points
    
    # Different "shapes" of N points on a hexagonal lattice
    # We'll generate points as (i, j) in lattice coordinates, then convert to Cartesian
    
    def _lattice_to_cartesian(i, j, d):
        """Convert hex lattice coords (i,j) to Cartesian with spacing d."""
        x = d * (i + j/2)
        y = d * j * math.sqrt(3)/2
        return (x, y)
    
    # Generate compact sets of N lattice points
    # Start from origin, add nearest neighbors
    def _generate_compact_set(N_pts, max_ring=5):
        """Generate compact sets of N points on hex lattice."""
        # Start with origin, add points in order of distance
        candidates = [(0, 0)]
        # Generate all points within max_ring
        for i in range(-max_ring, max_ring+1):
            for j in range(-max_ring, max_ring+1):
                # Hex distance
                dist = max(abs(i), abs(j), abs(i+j))
                if 0 < dist <= max_ring:
                    candidates.append((i, j))
        
        # Sort by hex distance
        candidates.sort(key=lambda p: max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])))
        
        # Take first N points (including origin)
        selected = [(0,0)] + candidates[:N_pts-1]
        return selected
    
    # For N=11, generate a compact set
    lattice_points = _generate_compact_set(N, max_ring=3)
    
    # Now, for various spacings d, compute the enclosing hexagon
    # The optimal d is when the hexagons are just touching: d = sqrt(3) (inradius*2)
    # But we can try a range of d values
    
    # Actually, for a given arrangement of lattice points, the minimum enclosing hexagon
    # side length is proportional to d. So we want the smallest d such that all hexagons
    # fit inside and don't overlap.
    
    # For non-overlap: minimum center distance >= sqrt(3) (for same orientation)
    # In our lattice, adjacent points are at distance d, so we need d >= sqrt(3)
    
    # For the enclosing hexagon: side length S = (2/sqrt(3)) * r where r is the max "radius"
    # in the hexagon metric
    
    # Let's just do a direct search:
    # 1. Fix the lattice arrangement (compact set of N points)
    # 2. Try d = sqrt(3) (minimum for no overlap)
    # 3. Compute the minimum enclosing hexagon
    # 4. Verify the packing is valid
    
    d_min = math.sqrt(3)  # minimum spacing for no overlap (same orientation)
    
    # Try the compact lattice arrangement
    for d in [d_min, d_min * 1.01, d_min * 1.05, d_min * 1.1]:
        if time.time() > deadline:
            break
        
        # Convert lattice points to Cartesian
        cart_points = [_lattice_to_cartesian(i, j, d) for i, j in lattice_points]
        
        # Center the arrangement at origin
        cx = sum(p[0] for p in cart_points) / N
        cy = sum(p[1] for p in cart_points) / N
        cart_points = [(p[0] - cx, p[1] - cy) for p in cart_points]
        
        # Compute minimum enclosing hexagon
        S, c_enc = _min_enclosing_hexagon(cart_points, angle=0.0)
        
        # The outer hexagon center should be at c_enc (relative to the arrangement center)
        # But we already centered the arrangement, so outer center is at c_enc
        outer_c = (c_enc[0], c_enc[1])
        outer_s = S
        outer_a = 0.0
        positions = cart_points
        angles = [0.0] * N
        
        # Verify the packing
        if _packing_valid(positions, angles, outer_c, outer_s, outer_a):
            if S < best_side:
                best_side = S
                best_positions = positions[:]
                best_angles = angles[:]
                best_outer_c = outer_c
                best_outer_a = outer_a
    
    # Also try different lattice arrangements
    # Generate alternative compact sets
    alternative_sets = []
    
    # Set 2: more linear arrangement
    for i in range(-2, 3):
        for j in range(-2, 3):
            if max(abs(i), abs(j), abs(i+j)) <= 2:
                alternative_sets.append((i, j))
    # This gives 19 points, take 11
    alternative_sets.sort(key=lambda p: max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])))
    alt_set = [(0,0)] + alternative_sets[:N-1]
    
    if time.time() < deadline:
        for d in [d_min, d_min * 1.01, d_min * 1.05]:
            cart_points = [_lattice_to_cartesian(i, j, d) for i, j in alt_set]
            cx = sum(p[0] for p in cart_points) / N
            cy = sum(p[1] for p in cart_points) / N
            cart_points = [(p[0] - cx, p[1] - cy) for p in cart_points]
            
            S, c_enc = _min_enclosing_hexagon(cart_points, angle=0.0)
            outer_c = (c_enc[0], c_enc[1])
            outer_s = S
            outer_a = 0.0
            positions = cart_points
            angles = [0.0] * N
            
            if _packing_valid(positions, angles, outer_c, outer_s, outer_a):
                if S < best_side:
                    best_side = S
                    best_positions = positions[:]
                    best_angles = angles[:]
                    best_outer_c = outer_c
                    best_outer_a = outer_a
    
    # Try with 30-degree rotation of inner hexagons
    if time.time() < deadline:
        for d in [d_min, d_min * 1.01]:
            cart_points = [_lattice_to_cartesian(i, j, d) for i, j in lattice_points]
            cx = sum(p[0] for p in cart_points) / N
            cy = sum(p[1] for p in cart_points) / N
            cart_points = [(p[0] - cx, p[1] - cy) for p in cart_points]
            
            S, c_enc = _min_enclosing_hexagon(cart_points, angle=30.0)
            outer_c = (c_enc[0], c_enc[1])
            outer_s = S
            outer_a = 30.0
            positions = cart_points
            angles = [30.0] * N
            
            if _packing_valid(positions, angles, outer_c, outer_s, outer_a):
                if S < best_side:
                    best_side = S
                    best_positions = positions[:]
                    best_angles = angles[:]
                    best_outer_c = outer_c
                    best_outer_a = outer_a
    
    # If no valid packing found, fall back to a safe large hexagon
    if best_positions is None:
        # Place hexagons in a line with spacing 2 (safe)
        best_positions = [(2.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_outer_c = (0.0, 0.0)
        best_outer_a = 0.0
        # Compute required outer side
        max_x = max(p[0] for p in best_positions) + 1
        max_y = max(abs(p[1]) for p in best_positions) + 1
        # For a flat-top hexagon, the extent in x is s, in y is s*sqrt(3)/2
        # Actually for point-in-hex check with angle 0:
        # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300
        # Inradius = s*sqrt(3)/2
        # The constraint is |x*cos(theta_k) + y*sin(theta_k)| <= s*sqrt(3)/2 for all k
        # For the farthest point, we need s >= max_k |x*cos(theta_k) + y*sin(theta_k)| / (sqrt(3)/2)
        S = 0
        for px, py in best_positions:
            for k in range(6):
                theta = k * math.pi / 3
                val = abs(px * math.cos(theta) + py * math.sin(theta))
                S = max(S, val / (math.sqrt(3)/2))
        best_side = S + 0.1  # small margin
    
    return best_positions, best_outer_c, best_side, best_outer_a

# EVOLVE-BLOCK-END

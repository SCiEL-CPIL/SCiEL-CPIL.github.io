import numpy as np
import time
from itertools import combinations

N = 12
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _point_in_convex_polygon(p, poly, eps=1e-9):
    """Check if point p is inside convex polygon poly (CCW)."""
    # poly is list of (x,y)
    # Cross product of edge vector and vector from start to p
    # If all cross products have the same sign (>= -eps for CCW), it's inside
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        if cross < -eps:
            return False
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    elif a < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(r + t * d2)
    elif e < 1e-12:
        s = -np.dot(d1, r) / a
        s = max(0.0, min(1.0, s))
        return np.linalg.norm(s * d1 - r)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    
    if denom < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        s = max(0.0, min(1.0, (b * t - c) / a))
    else:
        s = (b * f - c * e) / denom
        if s < 0.0 or s > 1.0:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
        else:
            t = (b * s + f) / e
            t = max(0.0, min(1.0, t))
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    
    return np.linalg.norm(s * d1 - r - t * d2)

def _polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap (including boundary)."""
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            if _seg_seg_dist(p1, p2, q1, q2) < eps:
                return True
    return False

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
            
            # Check containment
            is_inside = True
            for p in inner_verts[i]:
                if not _point_in_convex_polygon(p, outer_verts, eps=eps):
                    is_inside = False
                    break
            
            if not is_inside:
                return False, "outside"
                
    return True, "valid"

def _compute_side_for_topology(points, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with center at centroid
    and rotation angle angle_deg that contains all 'points'.
    """
    centroid = np.mean(points, axis=0)
    theta = np.deg2rad(angle_deg)
    
    # Normals of the hexagon sides
    # The apothem is S * sqrt(3) / 2
    # The distance from center to each side is S * sqrt(3) / 2
    # For a point P to be inside, its projection onto the normal of each side
    # must be <= S * sqrt(3) / 2
    # Normals are at angles theta + k * 60 degrees for k=0..5
    # Actually, for a regular hexagon, the normals to the sides are at angles
    # theta + k * 60, where theta is the angle of the first vertex?
    # Let's check _hex_vertices.
    # Vertices are at theta + k * 60.
    # The normal to the edge between vertex 0 and 1 is perpendicular to the edge.
    # Edge 0-1 vector: v1 - v0.
    # It is easier to use the support function.
    # The hexagon is the intersection of 6 half-planes.
    # The half-planes are defined by normals n_k = (cos(alpha_k), sin(alpha_k))
    # where alpha_k = theta + k * 60 + 30? Or just theta + k * 60?
    # Let's derive.
    # Vertex 0: (cos(theta), sin(theta)) * S
    # Vertex 1: (cos(theta+60), sin(theta+60)) * S
    # Midpoint of edge 0-1: S/2 * (cos(theta)+cos(theta+60), sin(theta)+sin(theta+60))
    # = S/2 * (2 cos(theta+30) cos(-30), 2 sin(theta+30) cos(-30))
    # = S * cos(30) * (cos(theta+30), sin(theta+30))
    # = S * sqrt(3)/2 * (cos(theta+30), sin(theta+30))
    # So the normal direction for edge 0-1 is theta + 30.
    # Generally, the normals are at angles theta + 30 + k * 60.
    
    max_proj = 0.0
    for p in points:
        v = p - centroid
        for k in range(6):
            alpha = theta + np.deg2rad(30 + k * 60)
            normal = np.array([np.cos(alpha), np.sin(alpha)])
            proj = np.dot(v, normal)
            max_proj = max(max_proj, abs(proj))
            
    # max_proj is the max distance to the supporting lines.
    # This must be <= apothem = S * sqrt(3) / 2
    # S >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    # 1. Generate candidate lattice points
    # Triangular lattice
    # We need a set of points that can form a valid packing.
    # Distance between centers must be >= 2 (for side 1 hexagons, min dist is sqrt(3) ~ 1.732, so 2 is safe).
    # Actually, if they are on the lattice, distance 1 is 1.0, which is < 1.732, so they overlap.
    # Distance 2 is 2.0, which is > 1.732, so they are disjoint.
    # So we need an independent set of size 12 in the triangular lattice graph where edges connect distance 1 neighbors.
    
    # Let's generate a grid of lattice points.
    # Lattice vectors: u = (1, 0), v = (0.5, sqrt(3)/2)
    # Point = i*u + j*v
    
    candidates = []
    # Radius 5 should be enough for 12 points.
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2.0) * j
            if x*x + y*y <= 5.0**2:
                candidates.append((x, y))
                
    candidates = np.array(candidates)
    
    # 2. Find valid topologies (sets of 12 points)
    # We can use a greedy or backtracking approach to find independent sets.
    # Given the small size, we can try to find a few distinct "shapes".
    # A simple shape is a "hexagonal" cluster.
    # Let's just find one valid set first, then maybe optimize.
    
    # Greedy selection:
    # Sort by distance from origin.
    # Pick a point if it is >= 1.8 away from all picked points.
    # This ensures they are on the lattice and disjoint (since lattice dist 1 is 1.0, dist 2 is 2.0).
    # Wait, if we pick points at lattice distance 2, their Euclidean distance is 2.0.
    # 2.0 > 1.732, so they are disjoint.
    # So we just need to pick 12 points from the lattice such that no two are at Euclidean distance 1.0.
    
    sorted_indices = np.argsort(np.sum(candidates**2, axis=1))
    
    picked = []
    for idx in sorted_indices:
        if len(picked) >= 12:
            break
        p = candidates[idx]
        valid = True
        for q in picked:
            if np.linalg.norm(p - q) < 1.8: # 1.8 is safe margin for 1.0
                valid = False
                break
        if valid:
            picked.append(p)
            
    if len(picked) < 12:
        # Fallback: try a different seed or random restarts?
        # For now, let's just return the seed if we fail, but we should find one.
        # The greedy above should find one.
        # Let's verify.
        # Center (0,0).
        # Neighbors at dist 1: (1,0), (0.5, 0.866), etc. 6 neighbors.
        # We pick (0,0).
        # Next closest: dist 2.
        # There are 6 points at dist 2 (e.g., (2,0), (1, 1.732), etc).
        # We can pick all 6?
        # Distance between (2,0) and (1, 1.732) is sqrt(1^2 + 1.732^2) = sqrt(1+3) = 2.
        # So yes, they are disjoint.
        # So we can pick Center + 6 at dist 2 + 5 at dist 3?
        # Dist 3 points: (3,0), (2.5, 2.598), etc.
        # This should work.
        
        # If greedy fails, we try a more robust search later.
        pass

    if len(picked) == 12:
        points = np.array(picked)
        
        # 3. Optimize the rotation angle
        # The bounding hexagon side length depends on the rotation angle.
        # We want to minimize S(angle).
        # S(angle) is a continuous function, likely periodic with period 60 degrees.
        # We can use golden-section search on [0, 60].
        
        best_side = float('inf')
        best_angle = 0.0
        
        # Golden section search
        phi = (np.sqrt(5) - 1) / 2
        a = 0.0
        b = 60.0
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        
        fc = _compute_side_for_topology(points, c)
        fd = _compute_side_for_topology(points, d)
        
        # We also need to check if the configuration is actually valid (disjoint and inside).
        # The side length calculation assumes the hexagon is centered at the centroid.
        # Is the centroid always a valid center?
        # For a set of points, the minimal enclosing hexagon might not be centered at the centroid?
        # Actually, for a regular polygon, the center of the minimal enclosing polygon
        # is not necessarily the centroid of the points.
        # However, if we fix the center to the centroid, we might miss the optimal.
        # But moving the center adds 2 more dimensions.
        # Given the symmetry of the lattice packing, the centroid is a good guess.
        # Let's assume the center is the centroid for now.
        
        # To be safe, we should also optimize the center?
        # That's a 3D optimization (cx, cy, angle).
        # Let's stick to 1D angle search with centroid fixed, as it's fast.
        # If the result is invalid, we can try adjusting the center slightly.
        
        for _ in range(50): # 50 iterations of golden section
            if time.time() - start_time > TIME_LIMIT:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                fc = _compute_side_for_topology(points, c)
            else:
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                fd = _compute_side_for_topology(points, d)
                
        best_angle = (a + b) / 2.0
        best_side = _compute_side_for_topology(points, best_angle)
        
        # Verify validity
        centroid = np.mean(points, axis=0)
        inner_hexes = [[p[0], p[1], 0.0] for p in points]
        
        # The inner hexagons are on the lattice, so their orientation is 0 (or 30, doesn't matter for lattice).
        # Wait, the lattice points are centers. The hexagons themselves have orientation.
        # If we place them on the triangular lattice, their "natural" orientation is such that
        # their sides are parallel to the lattice vectors?
        # No, a regular hexagon of side 1 has width sqrt(3).
        # If we place them at lattice distance 2, they are disjoint.
        # Their orientation can be anything, as long as they don't overlap.
        # Since they are far apart (dist 2 > 1.732), any orientation works.
        # So we can set angle = 0 for all inner hexagons.
        
        is_valid, reason = _check_validity(inner_hexes, best_side, centroid, best_angle)
        
        if is_valid:
            return (inner_hexes, [centroid[0], centroid[1]], best_side, best_angle)
        else:
            # If invalid, try to adjust the center?
            # Or just return a slightly larger side to ensure validity?
            # The side calculation is based on projection, so it should be sufficient.
            # The only risk is if the centroid is not the optimal center.
            # Let's try to expand the side slightly if it fails.
            side = best_side * 1.01
            is_valid, reason = _check_validity(inner_hexes, side, centroid, best_angle)
            if is_valid:
                return (inner_hexes, [centroid[0], centroid[1]], side, best_angle)
            
            # If still invalid, maybe the greedy picked a bad set?
            # Let's try a different set.
            # For now, return seed.
            pass

    # Fallback: Return the seed configuration if search fails
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

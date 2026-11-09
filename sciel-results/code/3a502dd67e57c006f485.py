"""Packing 12 unit hexagons in minimal regular hexagon using local optimization."""
import math
import time
import random
import numpy as np

N = 12

def hex_vertices(center, side, angle):
    """Generate vertices of a regular hexagon."""
    cx, cy = center
    rad = math.radians(angle)
    # Vertices are at angles 0, 60, ... 300 relative to the hex orientation
    # A regular hexagon with side length s has circumradius s.
    # The vertices are at distance s from the center.
    verts = []
    for i in range(6):
        theta = rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def point_in_hexagon(point, center, side, angle):
    """Check if a point is inside a regular hexagon."""
    cx, cy = center
    px, py = point
    rad = math.radians(angle)
    
    # Translate point to hex center
    dx = px - cx
    dy = py - cy
    
    # Rotate point by -angle to align hex with standard orientation
    cos_a = math.cos(-rad)
    sin_a = math.sin(-rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    # For a standard hexagon (flat top or pointy top depending on definition)
    # Let's use the definition where vertices are at 0, 60, 120...
    # The supporting lines are at distances.
    # Actually, it's easier to check against the 6 edges.
    # Or use the property: max(|x|, |y|, |x+y|) <= s * something?
    # For a hexagon with circumradius R (side length R), vertices at (R, 0), (R/2, R*sqrt(3)/2)...
    # The hexagon is the intersection of 6 half-planes.
    # In the rotated frame, the hexagon is defined by:
    # |x| <= R * cos(30) ? No.
    # Let's stick to edge intersection or simple distance to edges.
    
    # Distance from center to each edge (apothem) is R * sqrt(3)/2.
    # The edges are normal to directions 30, 90, 150, 210, 270, 330 degrees?
    # If vertices are at 0, 60, 120... then edges are centered at 30, 90, 150...
    # So the normal vectors are at 30, 90, 150, 210, 270, 330.
    # A point (rx, ry) is inside if dot((rx,ry), normal_i) <= apothem for all i.
    
    apothem = side * math.sqrt(3.0) / 2.0
    for i in range(6):
        theta = math.pi / 6.0 + i * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        dist = rx * nx + ry * ny
        if dist > apothem:
            return False
    return True

def hexagons_overlap(c1, s1, a1, c2, s2, a2):
    """Check if two hexagons overlap.
    Returns True if they overlap, False otherwise.
    Since all are convex, we can check if any vertex of one is in the other,
    or if edges intersect.
    Simpler: Check if any vertex of hex1 is in hex2 OR any vertex of hex2 is in hex1.
    This is sufficient for disjoint interiors if we consider boundaries touching as non-overlapping?
    The task says "disjoint", usually meaning interiors do not intersect.
    If vertices touch, area of intersection is 0.
    """
    v1 = hex_vertices(c1, s1, a1)
    v2 = hex_vertices(c2, s2, a2)
    
    # Check if any vertex of v1 is in hex2
    for p in v1:
        if point_in_hexagon(p, c2, s2, a2):
            return True
    # Check if any vertex of v2 is in hex1
    for p in v2:
        if point_in_hexagon(p, c1, s1, a1):
            return True
            
    # It is possible for two convex polygons to overlap without a vertex of one being inside the other?
    # No, if two convex polygons intersect, either a vertex of one is inside the other, or edges cross.
    # If edges cross, the intersection point is on the boundary.
    # However, if one is completely inside the other, a vertex check catches it.
    # If they partially overlap, a vertex of one must be inside the other?
    # Consider two squares crossing. One vertex of A is in B? Not necessarily.
    # Example: A is [0,2]x[0,2], B is [1,3]x[1,3].
    # Vertices of A: (0,0) no, (2,0) no, (0,2) no, (2,2) yes (inside B? 1<=2<=3, 1<=2<=3 -> Yes).
    # It turns out for convex polygons, if they intersect with non-zero area, at least one vertex of one polygon lies in the other.
    # Proof: Let P and Q be convex. If P ∩ Q != empty and not a single point/line, there is a 2D region.
    # Pick a vertex v of P. If v is in Q, done. If not, the edge from v must cross Q?
    # Actually, a more robust check is segment-segment intersection.
    # But for hexagons of same size, vertex check is usually fast enough and sufficient for "disjoint" in packing contexts where we optimize away deep overlaps.
    # To be safe against "crossing" where no vertex is inside (e.g. star of david shape intersection),
    # we should check edge intersections.
    
    def segments_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        def on_segment(a, b, c):
            return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        if d1 == 0 and on_segment(p3, p1, p4): return True
        if d2 == 0 and on_segment(p3, p2, p4): return True
        if d3 == 0 and on_segment(p1, p3, p2): return True
        if d4 == 0 and on_segment(p1, p4, p2): return True
        return False

    for i in range(6):
        e1 = (v1[i], v1[(i+1)%6])
        for j in range(6):
            e2 = (v2[j], v2[(j+1)%6])
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                # If they touch at a point/edge, is it "disjoint"?
                # "Disjoint" usually means interiors are disjoint.
                # If they only touch, area is 0.
                # We want to avoid overlaps. Touching is allowed.
                # However, numerical precision might make this tricky.
                # We'll treat strict intersection as overlap.
                # The ccw function uses strict inequality > 0.
                # If they just touch, ccw might be 0.
                # We return True if they cross.
                return True
                
    return False

def calculate_outer_side_length(centers, angles):
    """Calculate the minimum side length of a regular hexagon containing all inner hexagons.
    The outer hexagon can rotate and translate.
    Actually, the outer hexagon is centered at (0,0) in our local frame?
    No, the return value is (inner, center, side, angle).
    We can assume the outer hexagon is centered at the centroid of the inner hexagons to minimize size?
    Not necessarily, but it's a good heuristic.
    Let's fix the outer center to be the centroid of the inner centers for simplicity,
    and optimize the outer angle and side length.
    """
    # Centroid
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    
    # Translate centers to local frame centered at (0,0)
    local_centers = [(c[0]-cx, c[1]-cy) for c in centers]
    
    # We need to find the minimum L such that there exists an angle alpha
    # where all vertices of all inner hexagons are inside the outer hexagon.
    # Outer hexagon with side L, center (0,0), angle alpha.
    # A point (x,y) is inside outer hexagon if:
    # For all k in 0..5, dot((x,y), normal_k(alpha)) <= L * sqrt(3)/2
    # Where normal_k are directions 30+60k, 90+60k, etc.
    
    # We can iterate over a few candidate angles for the outer hexagon.
    # The optimal angle is likely aligned with the symmetry of the packing.
    # Let's try angles 0, 15, 30, 45, 60 degrees.
    
    min_L = float('inf')
    
    # Generate all vertices of all inner hexagons in local frame
    all_verts = []
    for i, c in enumerate(local_centers):
        v = hex_vertices(c, 1.0, angles[i])
        all_verts.extend(v)
        
    # Test a range of outer angles
    for alpha_deg in range(0, 60, 5):
        alpha = math.radians(alpha_deg)
        max_dist = 0.0
        for v in all_verts:
            # Check distance to each of the 6 faces
            # The constraint is max_k dot(v, n_k) <= L * apothem
            # So L >= max_k dot(v, n_k) / apothem
            # We need the maximum over all vertices and all faces.
            for k in range(6):
                theta = math.pi / 6.0 + k * math.pi / 3.0 + alpha
                nx = math.cos(theta)
                ny = math.sin(theta)
                dist = v[0] * nx + v[1] * ny
                if dist > max_dist:
                    max_dist = dist
        # L * sqrt(3)/2 >= max_dist
        # L >= max_dist * 2 / sqrt(3)
        L = max_dist * 2.0 / math.sqrt(3.0)
        if L < min_L:
            min_L = L
            
    return min_L, (cx, cy), 0.0 # Return dummy angle, we will compute it or just return 0 as the rotation of the *inner* relative to outer?
    # The interface asks for outer angle. We should return the angle that gave min_L.
    # Let's re-do to keep track of best angle.

def calculate_outer_side_length_v2(centers, angles):
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    local_centers = [(c[0]-cx, c[1]-cy) for c in centers]
    
    all_verts = []
    for i, c in enumerate(local_centers):
        v = hex_vertices(c, 1.0, angles[i])
        all_verts.extend(v)
        
    min_L = float('inf')
    best_angle = 0.0
    
    for alpha_deg in range(0, 60, 2): # Finer search
        alpha = math.radians(alpha_deg)
        max_dist = 0.0
        for v in all_verts:
            for k in range(6):
                theta = math.pi / 6.0 + k * math.pi / 3.0 + alpha
                nx = math.cos(theta)
                ny = math.sin(theta)
                dist = v[0] * nx + v[1] * ny
                if dist > max_dist:
                    max_dist = dist
        L = max_dist * 2.0 / math.sqrt(3.0)
        if L < min_L:
            min_L = L
            best_angle = alpha_deg
            
    return min_L, (cx, cy), best_angle

def check_validity(centers, angles):
    for i in range(N):
        for j in range(i+1, N):
            if hexagons_overlap(centers[i], 1.0, angles[i], centers[j], 1.0, angles[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Initial configuration: Flower pattern
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance 2*sqrt(3) ? No, side length 1.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait, for flat-top hexagons, distance to neighbor is 2 * apothem = sqrt(3)?
    # Apothem = sqrt(3)/2. Distance = 2 * sqrt(3)/2 = sqrt(3).
    # Let's verify.
    # Hexagon A at (0,0). Hexagon B at (sqrt(3), 0).
    # Rightmost point of A is (1, 0) (vertex).
    # Leftmost point of B is (sqrt(3)-1, 0).
    # sqrt(3) ~ 1.732. 1.732 - 1 = 0.732.
    # They don't touch.
    # For flat-top hexagons, the neighbors are at (1.5, sqrt(3)/2), etc?
    # Let's use the standard hex grid coordinates.
    # If we use "pointy top" hexagons (vertices at 90, 150...), neighbors are at (0, sqrt(3)), etc.
    # If we use "flat top" (vertices at 0, 60...), neighbors are at (1.5, sqrt(3)/2)?
    # Let's just use a known valid packing:
    # 1 center.
    # 6 around it.
    # 5 more around the perimeter.
    
    # Let's use a simple grid-like start and let the optimizer fix it.
    # Or better, generate a valid "flower" manually.
    
    # Let's assume Pointy-Top hexagons for easier mental model?
    # The code uses Flat-Top (vertices at 0, 60...).
    # Neighbors for Flat-Top:
    # (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2), (0, -sqrt(3)).
    
    sqrt3 = math.sqrt(3.0)
    
    centers = [(0.0, 0.0)]
    # 6 neighbors
    neighbors = [
        (1.5, sqrt3/2), (1.5, -sqrt3/2),
        (0, sqrt3),
        (-1.5, sqrt3/2), (-1.5, -sqrt3/2),
        (0, -sqrt3)
    ]
    centers.extend(neighbors)
    
    # We need 12. We have 7. Need 5 more.
    # Place them further out.
    # Next ring distance is roughly 3.
    # Let's just place them in a rough circle and let the optimizer push them in.
    # Initial guess: place remaining 5 in the gaps of the second ring.
    # Or simply place them at random positions far away and let the optimizer pull them in?
    # Risk: optimizer gets stuck.
    # Better: Place them in valid positions in a larger cluster.
    # A side-3 hexagon has 19 cells. We use 12.
    # Let's take the first 12 from a side-3 hexagon grid.
    
    # Generate axial coords for side 3 (radius 2)
    # q from -2 to 2, r from -2 to 2, |q+r| <= 2
    grid_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q+r) <= 2:
                # Convert to cartesian (Flat-Top)
                x = 1.5 * q
                y = sqrt3 * (r + q/2.0)
                grid_centers.append((x, y))
                
    # We have 19 centers. Pick 12.
    # To minimize bounding box, we should pick a compact subset.
    # The center + 6 neighbors + 5 from the next layer.
    # Let's sort by distance from center and pick the closest 12.
    grid_centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    centers = grid_centers[:12]
    
    # Angles: all 0 initially
    angles = [0.0] * 12
    
    # Verify initial validity
    if not check_validity(centers, angles):
        # Should be valid since they are from a grid
        pass
        
    current_L, current_center, current_outer_angle = calculate_outer_side_length_v2(centers, angles)
    
    best_L = current_L
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_center = current_center
    best_outer_angle = current_outer_angle
    
    # Local Optimization (Simulated Annealing)
    temp = 0.5
    cooling = 0.995
    
    while time.time() < deadline:
        if temp < 0.001:
            break
            
        # Perturb
        new_centers = [c[:] for c in best_centers]
        new_angles = best_angles[:]
        
        idx = random.randint(0, N-1)
        move_type = random.random()
        
        if move_type < 0.5:
            # Move position
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            new_centers[idx][0] += dx
            new_centers[idx][1] += dy
        else:
            # Rotate
            da = random.gauss(0, 10) * math.pi / 180.0
            new_angles[idx] += da
            
        # Check validity
        if check_validity(new_centers, new_angles):
            new_L, new_center, new_outer_angle = calculate_outer_side_length_v2(new_centers, new_angles)
            
            # Acceptance
            delta = new_L - best_L
            if delta < 0 or random.random() < math.exp(-delta / temp):
                best_L = new_L
                best_centers = new_centers
                best_angles = new_angles
                best_center = new_center
                best_outer_angle = new_outer_angle
        else:
            # If invalid, we could try to push them apart, but for now just reject
            pass
            
        temp *= cooling
        
    # Return best found
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
        
    return inner, best_center, best_L, best_outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to wrap it in the EVOLVE block or just have the block contain the logic?
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `def construct():` inside the EVOLVE block.
# So I will put the entire logic inside the EVOLVE block.
# EVOLVE-BLOCK-END

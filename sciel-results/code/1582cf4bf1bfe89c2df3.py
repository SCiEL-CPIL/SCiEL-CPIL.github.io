import numpy as np
import time
import math

N = 11

def get_hexagon_vertices(cx, cy, angle_deg, scale=1.0):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) 
    with given angle (degrees) and scale (side length).
    """
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + scale * math.cos(theta)
        y = cy + scale * math.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def point_in_polygon(point, polygon):
    """Check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def check_overlap(poly1, poly2):
    """
    Returns True if two convex polygons overlap.
    Uses Separating Axis Theorem (SAT).
    """
    axes = []
    for i in range(len(poly1)):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % len(poly1)]
        # Normal to edge p1-p2
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        
    for i in range(len(poly2)):
        p1 = poly2[i]
        p2 = poly2[(i + 1) % len(poly2)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        
    for axis in axes:
        # Project both polygons onto the axis
        proj1 = np.dot(poly1, axis)
        proj2 = np.dot(poly2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False # Disjoint
            
    # If no separating axis found, check if one is inside the other
    # (SAT for disjoint convex polygons usually suffices if we consider 
    # all edges, but if one is strictly inside the other, no edge intersects.
    # However, for "disjoint" packing, we want NO overlap.
    # If they touch, they are not overlapping in the interior sense usually 
    # meant by "disjoint" in packing (non-overlapping areas).
    # The prompt says "disjoint". Usually means interiors do not intersect.
    # SAT returns False (disjoint) if there is a gap. 
    # If they just touch, max1 == min2 or similar. 
    # Let's assume touching is allowed (measure zero area overlap).
    # To be safe, we can check if any vertex is inside the other.
    if point_in_polygon(poly1[0], poly2):
        return True
    if point_in_polygon(poly2[0], poly1):
        return True
        
    return False

def calc_enclosing_size(positions, angles, outer_angle_deg):
    """
    Calculate the minimum side length of a regular hexagon 
    with given outer_angle that encloses all inner hexagons.
    """
    # Normals for the outer hexagon
    outer_rad = math.radians(outer_angle_deg)
    normals = []
    for i in range(6):
        theta = outer_rad + i * math.pi / 3
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
    max_width = 0
    for n in normals:
        min_proj = np.inf
        max_proj = -np.inf
        for i in range(len(positions)):
            # Support function of inner hexagon i in direction n
            # h(n) = max_k (v_k . n) where v_k are vertices relative to center
            # Since h is homogeneous, we can just use the geometry of a unit hex
            # Rotated by angles[i].
            # Vertices of unit hex at angle a:
            # v_k = (cos(a + k*pi/3), sin(a + k*pi/3))
            # Dot with n = (cos(theta_n), sin(theta_n))
            # v_k . n = cos(a + k*pi/3 - theta_n)
            # Max of cos(x) over k is cos(min |a + k*pi/3 - theta_n| mod 2pi relative to 0?)
            # Actually, max_k cos(theta) where theta is the angle diff.
            # The max projection of a unit hexagon in direction n is 
            # the distance from center to the furthest vertex in that direction.
            # For a regular hexagon, the support function h(n) is:
            # If n aligns with a vertex direction, h=1.
            # If n aligns with a face normal, h=sqrt(3)/2.
            # Generally, h(n) = max_k cos(angle_diff_k).
            
            a_rad = math.radians(angles[i])
            max_h = 0
            for k in range(6):
                vertex_angle = a_rad + k * math.pi / 3
                # Angle difference between vertex direction and normal direction
                diff = vertex_angle - math.atan2(n[1], n[0])
                # Cos is even and periodic
                val = math.cos(diff)
                if val > max_h:
                    max_h = val
            
            # The actual projection of the center is p_i . n
            p = positions[i]
            proj = np.dot(p, n) + max_h
            
            if proj < min_proj:
                min_proj = proj
            if proj > max_proj:
                max_proj = proj
                
        width = max_proj - min_proj
        if width > max_width:
            max_width = width
            
    # The width of the enclosing hexagon in direction n is S * sqrt(3)
    # So S = width / sqrt(3)
    return max_width / math.sqrt(3)

def construct():
    deadline = time.time() + 100  # Leave 20s buffer
    
    # Initial Configuration: Hexagonal Grid
    # Center at (0,0). 
    # Hex grid spacing: 
    # Row 0: y=0, x from -2 to 2 (5 items)
    # Row 1: y=sqrt(3), x from -1.5 to 1.5 (3 items? No, offset)
    # Let's just place 11 items in a compact cluster.
    
    def init_grid():
        pos = []
        ang = []
        # Center
        pos.append([0.0, 0.0])
        ang.append(0.0)
        # Ring 1 (6 items)
        for i in range(6):
            theta = i * math.pi / 3
            x = 2.0 * math.cos(theta) # Distance 2 to avoid overlap? 
            # Actually, min dist is 2 for aligned hexes.
            # Let's use a small offset to allow rotation/packing
            x = 1.9 * math.cos(theta)
            y = 1.9 * math.sin(theta)
            pos.append([x, y])
            ang.append(math.degrees(theta))
        # Ring 2 (4 items to make 11)
        # Place them at the "corners" of the ring 1 hexagon
        for i in [0, 2, 4, 6]: # 4 spots
            theta = i * math.pi / 3
            # Push further out
            dist = 1.9 + 1.9
            # Wait, 1+6+4=11.
            # The 4 extra can be placed adjacent to the ring 1 hexagons.
            # Let's just scatter them slightly further out initially.
            x = 3.8 * math.cos(theta)
            y = 3.8 * math.sin(theta)
            pos.append([x, y])
            ang.append(math.degrees(theta))
        return np.array(pos), np.array(ang)

    best_pos, best_ang = init_grid()
    
    # Calculate initial cost
    def get_cost(pos, ang):
        # Check overlap
        poly_list = []
        for i in range(N):
            p = get_hexagon_vertices(pos[i][0], pos[i][1], ang[i])
            poly_list.append(p)
        
        overlap_penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(poly_list[i], poly_list[j]):
                    overlap_penalty += 100.0 # Large penalty
        
        if overlap_penalty > 0:
            return overlap_penalty + 1000.0 # Return huge cost
            
        # Calculate enclosing size
        # Try 0 and 30 degree rotation for outer hex
        s1 = calc_enclosing_size(pos, ang, 0.0)
        s2 = calc_enclosing_size(pos, ang, 30.0)
        return min(s1, s2)
        
    current_cost = get_cost(best_pos, best_ang)
    best_cost = current_cost
    
    # Simulated Annealing / Local Search
    T = 0.5
    T_min = 0.01
    cooling_rate = 0.95
    
    while time.time() < deadline and T > T_min:
        # Pick a random hexagon
        i = np.random.randint(N)
        
        # Random perturbation
        dx = np.random.normal(0, 0.2)
        dy = np.random.normal(0, 0.2)
        da = np.random.normal(0, 5.0) # 5 degrees
        
        new_pos = best_pos.copy()
        new_ang = best_ang.copy()
        new_pos[i][0] += dx
        new_pos[i][1] += dy
        new_ang[i] += da
        
        new_cost = get_cost(new_pos, new_ang)
        
        if new_cost < best_cost:
            best_pos = new_pos
            best_ang = new_ang
            best_cost = new_cost
            current_cost = new_cost
        else:
            # Accept worse move with probability
            delta = new_cost - current_cost
            if np.random.rand() < math.exp(-delta / T):
                best_pos = new_pos
                best_ang = new_ang
                current_cost = new_cost
                
        T *= cooling_rate
        
    # Final Polish: Try to minimize overlap and size more strictly
    # If still overlapping, try to push apart
    # But given the init, it should be disjoint.
    
    # Determine best outer rotation and center
    s1 = calc_enclosing_size(best_pos, best_ang, 0.0)
    s2 = calc_enclosing_size(best_pos, best_ang, 30.0)
    
    if s1 <= s2:
        final_side = s1
        final_outer_angle = 0.0
    else:
        final_side = s2
        final_outer_angle = 30.0
        
    # Center the configuration
    # We need the center of the enclosing hexagon.
    # For a hexagon of side S enclosing the points, the center is 
    # roughly the centroid of the points, adjusted for the "tight" directions.
    # For simplicity and validity, we can return the centroid of the inner hex centers 
    # as the outer center, and the calculated side length.
    # The evaluator will check if the inner hexagons fit.
    # If we return the centroid, and the side length is calculated as the min enclosing 
    # hexagon side *assuming* the hexagon is centered at the origin of the coordinate system 
    # of the points, we need to shift the points or the center.
    # The function returns (inner, center, side, angle).
    # The evaluator places the outer hexagon at 'center' with 'side' and 'angle'.
    # So we must provide the 'center' such that the outer hexagon encloses the 'inner' points.
    
    # Let's find the optimal center for the fixed orientation.
    # For a regular hexagon of side S and angle theta, the constraints are:
    # For each normal n_k: 
    # (p_i - C) . n_k + h_i(n_k) <= S/sqrt(3)  (Inradius)
    # (C - p_i) . n_k + h_i(n_k) <= S/sqrt(3)
    # This is a set of linear inequalities for C.
    # We can solve for C by taking the midpoint of the min and max projections for 
    # the direction that gives the width S*sqrt(3).
    
    # Simpler: Just use the centroid of the vertices of the convex hull of the centers?
    # Or iterate: 
    # 1. Start C = mean(pos)
    # 2. Calculate min/max projections in 6 directions.
    # 3. Adjust C to center the width in the most constraining direction.
    # 4. Repeat a few times.
    
    C = np.mean(best_pos, axis=0)
    for _ in range(10):
        # Calculate projections relative to C
        widths = []
        for k in range(6):
            theta = math.radians(final_outer_angle) + k * math.pi / 3
            n = np.array([math.cos(theta), math.sin(theta)])
            min_p = np.inf
            max_p = -np.inf
            for i in range(N):
                a_rad = math.radians(best_ang[i])
                max_h = 0
                for j in range(6):
                    va = a_rad + j * math.pi / 3
                    val = math.cos(va - math.atan2(n[1], n[0]))
                    if val > max_h: max_h = val
                proj = np.dot(best_pos[i] - C, n) + max_h
                if proj < min_p: min_p = proj
                if proj > max_p: max_p = proj
            widths.append((max_p - min_p, min_p, max_p, n))
            
        # Find direction with max width
        max_w, min_p, max_p, n = max(widths, key=lambda x: x[0])
        
        # The current width is max_p - min_p.
        # We want the center of this interval to be 0.
        # Current midpoint is (max_p + min_p) / 2.
        # We need to shift C by (midpoint) * n ?
        # If we shift C by delta, projections change by -delta . n.
        # New mid = old_mid - delta . n.
        # We want New mid = 0 => delta . n = old_mid.
        # Smallest delta is delta = old_mid * n.
        old_mid = (max_p + min_p) / 2.0
        C += old_mid * n
        
    # Final check
    # Recalculate side length with this center
    # Actually, calc_enclosing_size assumed center at 0.
    # If we shift points by -C, the side length is the same.
    # So final_side is correct.
    
    # Format output
    inner_list = []
    for i in range(N):
        inner_list.append([float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])])
        
    return inner_list, (float(C[0]), float(C[1])), float(final_side), float(final_outer_angle)

# EVOLVE-BLOCK-END

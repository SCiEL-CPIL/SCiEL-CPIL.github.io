"""Packing 12 unit hexagons inside a regular hexagon."""
import numpy as np
import time
import random

N = 12
DEADLINE = time.time() + 140.0  # Safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given angle."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + np.radians(60 * i)
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = np.radians(-angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side length s, the vertices are at distance s from center
    # In local coords, the hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The hexagon can be described by |lx| <= s, |ly| <= s*sqrt(3)/2, and |lx| + |ly|/sqrt(3) <= s
    s = side
    if abs(lx) > s + 1e-10:
        return False
    if abs(ly) > s * np.sqrt(3) / 2 + 1e-10:
        return False
    if abs(lx) + abs(ly) / np.sqrt(3) > s + 1e-10:
        return False
    return True

def hexagons_disjoint(v1, v2, v3, v4, eps=1e-9):
    """Check if two hexagons (defined by their vertex arrays) are disjoint (no overlap)."""
    # Check if any vertex of one is inside the other
    for i in range(6):
        if point_in_hexagon_from_verts(v1[i][0], v1[i][1], v2, eps):
            return False
        if point_in_hexagon_from_verts(v2[i][0], v2[i][1], v1, eps):
            return False
    # Check edge intersections
    if edges_intersect(v1, v2, eps):
        return False
    return True

def point_in_hexagon_from_verts(px, py, verts, eps=1e-9):
    """Check if point is inside hexagon defined by vertex array."""
    # Use ray casting or winding number
    inside = False
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        if ((y1 > py) != (y2 > py)):
            x_intersect = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_intersect:
                inside = not inside
    return inside

def segments_intersect(p1, p2, p3, p4, eps=1e-9):
    """Check if line segments p1-p2 and p3-p4 intersect."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return False

def edges_intersect(hex1, hex2, eps=1e-9):
    """Check if any edge of hex1 intersects any edge of hex2."""
    for i in range(6):
        p1 = hex1[i]
        p2 = hex1[(i + 1) % 6]
        for j in range(6):
            p3 = hex2[j]
            p4 = hex2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4, eps):
                return True
    return False

def all_inside_outer(inner_verts_list, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside the outer hexagon."""
    ocx, ocy = outer_center
    for verts in inner_verts_list:
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    return True

def is_valid(inner, outer_center, outer_side, outer_angle, eps=1e-8):
    """Check if the configuration is valid: no overlaps, all inside outer."""
    inner_verts_list = []
    for x, y, ang in inner:
        verts = hex_vertices(x, y, ang)
        inner_verts_list.append(verts)
    
    # Check containment
    if not all_inside_outer(inner_verts_list, outer_center, outer_side, outer_angle):
        return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts_list[i], inner_verts_list[j], eps):
                return False
    
    return True

def compute_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    ocx, ocy = outer_center
    max_dist = 0.0
    for x, y, ang in inner:
        verts = hex_vertices(x, y, ang)
        for vx, vy in verts:
            # Distance from point to outer hexagon boundary
            # Transform to outer hexagon's local coordinates
            dx = vx - ocx
            dy = vy - ocy
            angle = np.radians(-outer_angle)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # For a regular hexagon with side s, the support function in direction (lx, ly)
            # The hexagon contains point if |lx| <= s, |ly| <= s*sqrt(3)/2, |lx|+|ly|/sqrt(3) <= s
            # The minimum s such that point is inside: s >= max(|lx|, |ly|*sqrt(3)/2, |lx|+|ly|/sqrt(3))
            # Actually, the constraint is that the point must satisfy all three inequalities.
            # s >= |lx|, s >= |ly|*sqrt(3)/2, s >= |lx| + |ly|/sqrt(3)
            # Wait, let me reconsider. The hexagon with side s has:
            # Vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
            # The boundaries are:
            # Right: x = s
            # Top-right: x/2 + y/sqrt(3) = s/2  => x + 2y/sqrt(3) = s
            # Top-left: -x/2 + y/sqrt(3) = s/2  => -x + 2y/sqrt(3) = s
            # Left: x = -s
            # Bottom-left: -x/2 - y/sqrt(3) = s/2  => -x - 2y/sqrt(3) = s
            # Bottom-right: x/2 - y/sqrt(3) = s/2  => x - 2y/sqrt(3) = s
            
            # So the constraints are:
            # x <= s
            # x + 2y/sqrt(3) <= s
            # -x + 2y/sqrt(3) <= s
            # -x <= s  => x >= -s
            # -x - 2y/sqrt(3) <= s
            # x - 2y/sqrt(3) <= s
            
            # So s >= max(lx, lx + 2ly/sqrt(3), -lx + 2ly/sqrt(3), -lx, -lx - 2ly/sqrt(3), lx - 2ly/sqrt(3))
            # s >= max(|lx|, |lx + 2ly/sqrt(3)|, |-lx + 2ly/sqrt(3)|)
            # Actually simpler: s >= max(|lx|, |lx + 2ly/sqrt(3)|, |lx - 2ly/sqrt(3)|)
            # Hmm, let me just compute the maximum of all six expressions.
            
            sqrt3 = np.sqrt(3)
            expr1 = lx
            expr2 = lx + 2 * ly / sqrt3
            expr3 = -lx + 2 * ly / sqrt3
            expr4 = -lx
            expr5 = -lx - 2 * ly / sqrt3
            expr6 = lx - 2 * ly / sqrt3
            
            s_needed = max(expr1, expr2, expr3, expr4, expr5, expr6)
            max_dist = max(max_dist, s_needed)
    
    return max_dist

def objective(inner, outer_center, outer_angle):
    """Compute the outer side length for the given configuration."""
    return compute_outer_side(inner, outer_center, outer_angle)

def initial_config():
    """Create an initial configuration with hexagons in a circular arrangement."""
    inner = []
    # Place 6 hexagons in a ring, and 6 more
    # Actually, let's try a hexagonal lattice pattern
    # Or just place them in a circle
    r = 2.5  # radius of circle
    for i in range(N):
        angle = 2 * np.pi * i / N
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        inner.append([x, y, 0.0])
    return inner

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # Initial configuration
    inner = initial_config()
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial objective
    current_obj = objective(inner, outer_center, outer_angle)
    best_inner = [row[:] for row in inner]
    best_center = outer_center[:]
    best_angle = outer_angle
    best_obj = current_obj
    
    # Check if valid
    if is_valid(inner, outer_center, outer_angle):
        best_inner = [row[:] for row in inner]
        best_center = outer_center[:]
        best_angle = outer_angle
        best_obj = current_obj
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Generate a move
        move_type = random.random()
        new_inner = [row[:] for row in inner]
        new_center = outer_center[:]
        new_angle = outer_angle
        
        if move_type < 0.4:
            # Move a random hexagon
            idx = random.randint(0, N - 1)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            new_inner[idx][0] += dx
            new_inner[idx][1] += dy
        elif move_type < 0.6:
            # Rotate a random hexagon
            idx = random.randint(0, N - 1)
            dangle = random.gauss(0, 15)
            new_inner[idx][2] += dangle
        elif move_type < 0.7:
            # Move outer center
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            new_center[0] += dx
            new_center[1] += dy
        elif move_type < 0.8:
            # Rotate outer hexagon
            dangle = random.gauss(0, 5)
            new_angle += dangle
        else:
            # Move multiple hexagons
            idx1 = random.randint(0, N - 1)
            idx2 = random.randint(0, N - 1)
            if idx1 != idx2:
                dx1 = random.gauss(0, 0.3)
                dy1 = random.gauss(0, 0.3)
                new_inner[idx1][0] += dx1
                new_inner[idx1][1] += dy1
                dx2 = random.gauss(0, 0.3)
                dy2 = random.gauss(0, 0.3)
                new_inner[idx2][0] += dx2
                new_inner[idx2][1] += dy2
        
        # Compute new objective
        new_obj = objective(new_inner, new_center, new_angle)
        
        # Check validity
        if is_valid(new_inner, new_center, new_angle):
            delta = new_obj - current_obj
            if delta < 0 or random.random() < np.exp(-delta / max(T, T_min)):
                inner = new_inner
                outer_center = new_center
                outer_angle = new_angle
                current_obj = new_obj
                
                if new_obj < best_obj:
                    best_inner = [row[:] for row in inner]
                    best_center = outer_center[:]
                    best_angle = outer_angle
                    best_obj = new_obj
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return the best valid configuration found
    return best_inner, (best_center[0], best_center[1]), best_obj, best_angle

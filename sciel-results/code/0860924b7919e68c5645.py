import numpy as np
import time
import random

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Get 6 vertices of a regular hexagon (pointy-top by default with angle=0)."""
    angle = np.radians(angle_deg)
    vertices = []
    for k in range(6):
        a = angle + k * np.pi / 3
        vertices.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    # The 6 outward normals are at angles: angle_deg + 90 + k*60 for k=0..5
    # (for a hexagon with vertices at angle_deg + k*60)
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        normal_angle = np.radians(angle_deg + 90 + k * 60)
        dx = px - cx
        dy = py - cy
        proj = dx * np.cos(normal_angle) + dy * np.sin(normal_angle)
        if proj > apothem + 1e-10:
            return False
    return True

def hexagon_inside_hexagon(inner_cx, inner_cy, inner_side, inner_angle,
                            outer_cx, outer_cy, outer_side, outer_angle):
    """Check if inner hexagon is entirely inside outer hexagon."""
    inner_verts = hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def hexagons_overlap(cx1, cy1, a1, cx2, cy2, a2, side1=1.0, side2=1.0):
    """Check if two hexagons overlap."""
    v1 = hex_vertices(cx1, cy1, side1, a1)
    v2 = hex_vertices(cx2, cy2, side2, a2)
    
    # Check if any vertex of one is inside the other
    for vx, vy in v1:
        if point_in_hexagon(vx, vy, cx2, cy2, side2, a2):
            return True
    for vx, vy in v2:
        if point_in_hexagon(vx, vy, cx1, cy1, side1, a1):
            return True
    
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def compute_min_outer_side(inner, outer_cx, outer_cy, outer_angle):
    """Compute minimum outer side length for given center and angle to contain all inner hexagons."""
    apothem_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for (x, y, ang) in inner:
        verts = hex_vertices(x, y, 1.0, ang)
        for vx, vy in verts:
            for k in range(6):
                normal_angle = np.radians(outer_angle + 90 + k * 60)
                dx = vx - outer_cx
                dy = vy - outer_cy
                proj = dx * np.cos(normal_angle) + dy * np.sin(normal_angle)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / apothem_factor

def initial_hexagonal_lattice():
    """Create initial hexagonal lattice arrangement of 12 unit hexagons."""
    # For flat-top hexagons in a hex grid:
    # Horizontal spacing = sqrt(3) * side
    # Vertical spacing = 1.5 * side  
    # Odd rows offset by sqrt(3)/2 * side
    positions = []
    # 4 rows of 3
    for row in range(4):
        for col in range(3):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3) / 2
            y = row * 1.5
            positions.append([x, y, 0.0])
    
    # Center
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial configuration
    inner = initial_hexagonal_lattice()
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    outer_side = compute_min_outer_side(inner, outer_cx, outer_cy, outer_angle)
    best_side = outer_side
    best_inner = [list(p) for p in inner]
    best_outer_cx, best_outer_cy, best_outer_angle = outer_cx, outer_cy, outer_angle
    
    random.seed(42)
    step_size = 0.3
    
    # Try multiple initial angles for outer hexagon
    for trial_angle in [0, 30, 15, 45, 10, 20]:
        if time.time() > deadline - 10:
            break
        outer_angle = trial_angle
        # Re-optimize positions for this angle
        inner_copy = [list(p) for p in best_inner]
        ocx, ocy = best_outer_cx, best_outer_cy
        oside = compute_min_outer_side(inner_copy, ocx, ocy, outer_angle)
        if oside < best_side:
            best_side = oside
            best_outer_angle = outer_angle
    
    # Local search
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        
        new_inner = [list(p) for p in inner]
        new_ocx = outer_cx + random.gauss(0, step_size * 0.2)
        new_ocy = outer_cy + random.gauss(0, step_size * 0.2)
        new_oangle = (outer_angle + random.gauss(0, step_size * 0.1)) % 60
        
        n_perturb = random.randint(1, 2)
        for _ in range(n_perturb):
            idx = random.randint(0, N - 1)
            new_inner[idx][0] += random.gauss(0, step_size)
            new_inner[idx][1] += random.gauss(0, step_size)
            if random.random() < 0.2:
                new_inner[idx][2] = (new_inner[idx][2] + random.gauss(0, 10)) % 360
        
        new_oside = compute_min_outer_side(new_inner, new_ocx, new_ocy, new_oangle)
        
        # Check overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if hexagons_overlap(new_inner[i][0], new_inner[i][1], new_inner[i][2],
                                   new_inner[j][0], new_inner[j][1], new_inner[j][2]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and new_oside < best_side:
            best_side = new_oside
            best_inner = new_inner
            best_outer_cx, best_outer_cy, best_outer_angle = new_ocx, new_ocy, new_oangle
            inner = new_inner
            outer_cx, outer_cy, outer_angle = new_ocx, new_ocy, new_oangle
            step_size = max(0.005, step_size * 0.998)
        elif valid and random.random() < 0.15:
            inner = new_inner
            outer_cx, outer_cy, outer_angle = new_ocx, new_ocy, new_oangle
        
        if iterations % 1000 == 0:
            step_size = max(0.005, step_size * 0.9)
    
    # Final polish
    for _ in range(500):
        if time.time() > deadline - 2:
            break
        dx = random.gauss(0, 0.05)
        dy = random.gauss(0, 0.05)
        da = random.gauss(0, 0.02)
        new_side = compute_min_outer_side(best_inner, best_outer_cx + dx, best_outer_cy + dy, 
                                           (best_outer_angle + da) % 60)
        if new_side < best_side:
            best_side = new_side
            best_outer_cx += dx
            best_outer_cy += dy
            best_outer_angle = (best_outer_angle + da) % 60
    
    return best_inner, (best_outer_cx, best_outer_cy), best_side, best_outer_angle

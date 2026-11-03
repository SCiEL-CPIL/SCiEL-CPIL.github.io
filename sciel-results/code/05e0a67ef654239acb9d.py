"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        ang = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
    return verts

def _seg_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper or improper)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # Check collinear cases (touching)
    if d1 == 0 and _on_segment(p3, p4, p1): return True
    if d2 == 0 and _on_segment(p3, p4, p2): return True
    if d3 == 0 and _on_segment(p1, p2, p3): return True
    if d4 == 0 and _on_segment(p1, p2, p4): return True
    return False

def _on_segment(a, b, p):
    """Check if point p is on segment a-b (assuming collinear)."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

def _hexes_disjoint(v1, v2, eps=1e-9):
    """Check if two hexagons (lists of 6 vertices) are disjoint (no interior overlap, touching allowed)."""
    for i in range(6):
        j1 = (i + 1) % 6
        for j in range(6):
            j2 = (j + 1) % 6
            if _seg_intersect(v1[i], v1[j1], v2[j], v2[j2]):
                # Check if it's a proper intersection (crossing) vs just touching
                # For simplicity, treat any intersection as invalid unless we can verify touching
                # Actually, for packing, touching is allowed. Let's use a more robust check.
                pass
    # More robust: check if any vertex of one is strictly inside the other
    if _point_in_polygon_strict(v1[0], v2, eps): return False
    if _point_in_polygon_strict(v2[0], v1, eps): return False
    # Check edge crossings
    for i in range(6):
        j1 = (i + 1) % 6
        for j in range(6):
            j2 = (j + 1) % 6
            if _seg_intersect_strict(v1[i], v1[j1], v2[j], v2[j2]):
                return False
    return True

def _point_in_polygon_strict(p, poly, eps=1e-9):
    """Check if point p is strictly inside polygon poly."""
    # Ray casting
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint - eps:
                inside = not inside
    return inside

def _seg_intersect_strict(p1, p2, p3, p4):
    """Check if segments strictly intersect (crossing, not just touching)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _valid_config(inner, outer_center, outer_side, outer_angle):
    """Check if inner hexagons are disjoint and all inside outer hexagon."""
    # Check inner hexagons are inside outer
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_list = []
    for i in range(N):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        # Check all vertices inside outer
        for v in verts:
            if not _point_in_polygon_strict(v, outer_verts, -1e-9):
                return False
    # Check inner hexagons are pairwise disjoint
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexes_disjoint(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _side_length_for_inner(inner, outer_center, outer_angle):
    """Find minimum outer side length such that all inner hexagons fit inside outer hexagon."""
    # For a fixed outer center and angle, the minimum side length is determined by
    # the maximum distance from outer center to any vertex of any inner hexagon,
    # projected onto the 6 normal directions of the outer hexagon.
    # For a regular hexagon with side s and center c, a point p is inside iff
    # for each of 6 directions, the projection of (p-c) onto that direction <= s * sqrt(3)/2
    # Actually, the inradius is s * sqrt(3)/2.
    # The condition is: max over all vertices v of inner hexagons of (distance from outer center to v)
    # is not quite right because the shape is hexagonal.
    # Better: for each of the 6 half-planes defining the outer hexagon, find the max violation.
    # Outer hexagon with center (cx,cy), side s, angle a has 6 edges.
    # The distance from center to each edge is s * sqrt(3)/2.
    # For each edge normal direction n_k, the constraint is: dot(p - c, n_k) <= s * sqrt(3)/2 for all p in inner.
    # So s >= max over all inner vertices p and all k of dot(p - c, n_k) / (sqrt(3)/2)
    cx, cy = outer_center
    a = math.radians(outer_angle)
    max_val = 0.0
    for i in range(N):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        for v in verts:
            for k in range(6):
                ang_k = a + k * math.pi / 3.0
                nx, ny = math.cos(ang_k), math.sin(ang_k)
                val = (v[0] - cx) * nx + (v[1] - cy) * ny
                if val > max_val:
                    max_val = val
    return max_val / (math.sqrt(3) / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration: hexagonal packing
    # Place 12 hexagons in a roughly hexagonal cluster
    # Center + 6 around + 5 more
    random.seed(42)
    
    # Start with a simple valid-ish configuration
    # Hexagonal lattice spacing: 2*sqrt(3) for edge-to-edge touching
    d = 2 * math.sqrt(3)  # distance between centers of touching hexagons
    
    # Create a hexagonal cluster of 12 hexagons
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons (to make 12)
    # Actually, let's use a different arrangement
    # Center at origin, 6 around at distance d, 5 more at distance d*sqrt(3)
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    # 6 around
    for i in range(6):
        ang = i * math.pi / 3.0
        inner.append([d * math.cos(ang), d * math.sin(ang), 0.0])
    # 5 more at second ring
    for i in range(5):
        ang = i * 2 * math.pi / 5.0 + math.pi / 6
        r = d * math.sqrt(3)
        inner.append([r * math.cos(ang), r * math.sin(ang), 0.0])
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    current_side = _side_length_for_inner(inner, outer_center, outer_angle)
    
    # Simulated Annealing
    best_side = current_side
    best_inner = [row[:] for row in inner]
    best_center = outer_center[:]
    best_angle = outer_angle
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    step_size = 0.5
    
    while T > T_min:
        if time.time() > deadline - 10.0:
            break
        
        # Propose new configuration
        new_inner = [row[:] for row in inner]
        new_center = outer_center[:]
        new_angle = outer_angle
        
        # Randomly perturb one inner hexagon
        idx = random.randint(0, N - 1)
        new_inner[idx][0] += random.gauss(0, step_size)
        new_inner[idx][1] += random.gauss(0, step_size)
        new_inner[idx][2] += random.gauss(0, step_size * 30)  # degrees
        
        # Sometimes perturb outer center
        if random.random() < 0.1:
            new_center[0] += random.gauss(0, step_size)
            new_center[1] += random.gauss(0, step_size)
        
        # Sometimes perturb outer angle
        if random.random() < 0.05:
            new_angle += random.gauss(0, step_size * 10)
        
        new_side = _side_length_for_inner(new_inner, new_center, new_angle)
        
        # Check validity (disjointness)
        valid = True
        if new_side < 20:  # Only check if side is reasonable
            valid = _valid_config(new_inner, new_center, new_side, new_angle)
        
        if valid:
            delta = new_side - current_side
            if delta < 0 or random.random() < math.exp(-delta / T):
                inner = new_inner
                outer_center = new_center
                outer_angle = new_angle
                current_side = new_side
                if current_side < best_side:
                    best_side = current_side
                    best_inner = [row[:] for row in inner]
                    best_center = outer_center[:]
                    best_angle = outer_angle
        else:
            # Reject
            pass
        
        T *= cooling
        
        # Adapt step size
        if T < 0.1:
            step_size = 0.1
        elif T < 0.5:
            step_size = 0.3
    
    # Final polish: local optimization
    if time.time() < deadline - 5.0:
        # Gradient-free local search
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                for dx in [-0.1, 0, 0.1]:
                    for dy in [-0.1, 0, 0.1]:
                        trial_inner = [row[:] for row in best_inner]
                        trial_inner[i][0] += dx
                        trial_inner[i][1] += dy
                        trial_side = _side_length_for_inner(trial_inner, best_center, best_angle)
                        if _valid_config(trial_inner, best_center, trial_side, best_angle):
                            if trial_side < best_side - 1e-8:
                                best_side = trial_side
                                best_inner = trial_inner
                                improved = True
            if not improved:
                break
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END

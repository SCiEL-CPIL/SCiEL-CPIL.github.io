"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Helper functions for geometry
    
    def get_hex_vertices(center, angle_deg, side=1.0):
        """Returns list of (x, y) vertices for a regular hexagon."""
        cx, cy = center
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, center, angle_deg, side=1.0):
        """Check if point (px, py) is inside the hexagon defined by center, angle, side."""
        cx, cy = center
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        inradius = side * sqrt3 / 2
        
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = lx * nx + ly * ny
            if proj > inradius + 1e-9:
                return False
        return True

    def dist_point_to_segment(px, py, ax, ay, bx, by):
        """Calculate minimum distance from point P to segment AB."""
        # Vector AB
        vx = bx - ax
        vy = by - ay
        # Vector AP
        wx = px - ax
        wy = py - ay
        
        # Length squared of AB
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - ax, py - ay)
        
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - bx, py - by)
        
        # Project P onto AB
        b = c1 / c2
        px_proj = ax + b * vx
        py_proj = ay + b * vy
        
        return math.hypot(px - px_proj, py - py_proj)

    def min_distance_hexagons(v1, v2):
        """
        Calculate the minimum distance between the boundaries of two hexagons.
        Returns a negative value if they overlap (approximation of penetration depth).
        Returns a positive value if they are disjoint.
        
        Strategy:
        1. Check if any vertex of v1 is inside v2. If so, distance is negative.
           To estimate penetration depth, we can use the minimum distance from that vertex to the boundary of v2.
        2. Similarly for v2 vertices inside v1.
        3. If no vertices are inside, the minimum distance is the minimum distance between any edge of v1 and any edge of v2.
           Since they are convex and disjoint, the minimum distance will be between a vertex of one and an edge of the other, or between two parallel edges.
           Checking all vertex-edge pairs (6*6 + 6*6 = 72 checks) covers all cases for minimum distance between two convex polygons.
        """
        min_dist = float('inf')
        overlap_depth = 0.0
        
        # Check vertices of v1 against edges of v2
        for p1 in v1:
            px, py = p1
            # Check if p1 is inside v2
            # We can use point_in_hexagon, but we need the center and angle.
            # It's easier to just calculate distance to edges and check sign.
            # However, for "inside" check, we need to know if it's strictly inside.
            # Let's compute distance to each edge of v2.
            # If all distances to edges (with consistent normal direction) are <= 0, it's inside.
            # But we don't have normals here easily.
            
            # Alternative: Just compute min distance from p1 to all edges of v2.
            # If p1 is inside v2, the distance to the boundary is the min distance to edges.
            # But how do we know it's inside?
            # If p1 is inside, the "signed distance" to the boundary is negative.
            # We can determine if a point is inside a convex polygon by checking if it's on the same side of all edges.
            
            # Let's implement a robust point-in-convex-polygon check using cross products.
            # For a convex polygon defined by vertices in CCW order, point P is inside
            # if cross(edge, P - vertex) > 0 for all edges.
            
            def point_in_convex_poly(px, py, poly):
                n = len(poly)
                sign = None
                for i in range(n):
                    ax, ay = poly[i]
                    bx, by = poly[(i + 1) % n]
                    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                    if cross != 0:
                        if sign is None:
                            sign = 1 if cross > 0 else -1
                        else:
                            if (cross > 0 and sign == -1) or (cross < 0 and sign == 1):
                                return False
                return True

            if point_in_convex_poly(px, py, v2):
                # p1 is inside v2. Calculate min distance to boundary of v2.
                d_min = float('inf')
                for i in range(6):
                    ax, ay = v2[i]
                    bx, by = v2[(i + 1) % 6]
                    d = dist_point_to_segment(px, py, ax, ay, bx, by)
                    if d < d_min:
                        d_min = d
                # The penetration depth is roughly d_min.
                # We want to push them apart, so penalty should be based on d_min.
                overlap_depth = max(overlap_depth, d_min)
                # Note: We don't return immediately because other vertices might be deeper.
                # But for penalty, max penetration is a good metric.
                
        # Check vertices of v2 against edges of v1
        for p2 in v2:
            px, py = p2
            if point_in_convex_poly(px, py, v1):
                d_min = float('inf')
                for i in range(6):
                    ax, ay = v1[i]
                    bx, by = v1[(i + 1) % 6]
                    d = dist_point_to_segment(px, py, ax, ay, bx, by)
                    if d < d_min:
                        d_min = d
                overlap_depth = max(overlap_depth, d_min)

        if overlap_depth > 0:
            return -overlap_depth

        # If no vertices are inside, calculate min distance between edges
        # It suffices to check distance from each vertex of v1 to each edge of v2,
        # and each vertex of v2 to each edge of v1.
        for p1 in v1:
            px, py = p1
            for i in range(6):
                ax, ay = v2[i]
                bx, by = v2[(i + 1) % 6]
                d = dist_point_to_segment(px, py, ax, ay, bx, by)
                if d < min_dist:
                    min_dist = d
                    
        for p2 in v2:
            px, py = p2
            for i in range(6):
                ax, ay = v1[i]
                bx, by = v1[(i + 1) % 6]
                d = dist_point_to_segment(px, py, ax, ay, bx, by)
                if d < min_dist:
                    min_dist = d

        return min_dist

    def boundary_violation(c, a, s_outer):
        """
        Check if hexagon at c, angle a is inside outer hexagon with side s_outer, center (0,0), rotation 0.
        Returns max violation amount (0 if inside).
        """
        verts = get_hex_vertices(c, a)
        max_viol = 0.0
        inradius_outer = s_outer * sqrt3 / 2
        
        for p in verts:
            px, py = p
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = px * nx + py * ny
                if proj > inradius_outer:
                    viol = proj - inradius_outer
                    if viol > max_viol:
                        max_viol = viol
        return max_viol

    def objective(params):
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Precompute vertices for all hexagons to avoid recalculating in loops
        verts_list = []
        for i in range(N):
            verts_list.append(get_hex_vertices(pos[i], angs[i]))
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                # Quick bounding box check
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.hypot(dx, dy)
                if dist > 2.0 + 1e-9:
                    continue
                
                min_d = min_distance_hexagons(verts_list[i], verts_list[j])
                if min_d < 0:
                    # Overlapping. Penalty is proportional to penetration depth squared.
                    # Using squared penalty makes the landscape smoother near 0.
                    penalty += 100.0 * (-min_d) ** 2
                    
        # Check boundary
        for i in range(N):
            viol = boundary_violation(pos[i], angs[i], s)
            if viol > 0:
                penalty += 1000.0 * viol ** 2
        
        return s + penalty

    # Initialization
    best_solution = None
    best_score = float('inf')
    
    configs = []
    
    # Generate multiple initial configurations
    
    # Config 1: 3-4-4 grid
    rows = [3, 4, 4]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Config 2: 4-3-4 grid
    rows = [4, 3, 4]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Config 3: Random Jittered Grid
    rows = [3, 4, 4]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3 + random.uniform(-0.2, 0.2)
            y = y_offset + random.uniform(-0.2, 0.2)
            pos_init.append([x, y])
            ang_init.append(random.uniform(-10, 10))
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))

    # Run DE for each config
    for idx, (p_init, a_init) in enumerate(configs):
        bounds = []
        for i in range(N):
            bounds.append((-6, 6))
            bounds.append((-6, 6))
        for i in range(N):
            bounds.append((-180, 180))
        bounds.append((2.5, 10.0)) # Side length bound
        
        x0 = []
        for i in range(N):
            x0.append(p_init[i][0])
            x0.append(p_init[i][1])
        for i in range(N):
            x0.append(a_init[i])
        
        # Estimate initial S
        max_dist = 0
        for p in p_init:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        x0.append(max_dist + 1.5)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=150, # Increased iterations
                popsize=30,  # Increased population
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42 + idx
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
        except Exception as e:
            print(f"DE failed for config {idx}: {e}")
            
    if best_solution is None:
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

    # Extract solution
    pos = []
    for i in range(N):
        pos.append([best_solution[2*i], best_solution[2*i+1]])
    angs = [best_solution[2*N + i] for i in range(N)]
    s = best_solution[3*N]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END

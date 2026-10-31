"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

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
        # Transform point to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        # Rotate point by -angle
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # In local coords, hexagon vertices are at 0, 60, ...
        # Edges are perpendicular to directions 30, 90, ...
        # The hexagon is defined by 6 half-planes.
        # Normal to edge k is at angle 30 + 60k degrees.
        # Distance from center to edge is inradius = side * sqrt(3) / 2
        inradius = side * sqrt3 / 2
        
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            # Projection of local point onto normal
            proj = lx * nx + ly * ny
            if proj > inradius + 1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        
        # Handle collinear cases (touching)
        # If points are collinear and on the segment, they touch.
        # For packing, touching is allowed (disjoint interiors).
        # So we only care if they cross.
        return False

    def hexagons_overlap(c1, a1, c2, a2):
        """
        Check if two hexagons overlap.
        Returns True if they overlap (interiors intersect).
        """
        # Quick bounding box check
        # Max distance from center to vertex is 1.0
        # If distance between centers > 2.0, no overlap.
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.hypot(dx, dy)
        if dist > 2.0 + 1e-9:
            return False
        
        v1 = get_hex_vertices(c1, a1)
        v2 = get_hex_vertices(c2, a2)
        
        # Check if any vertex of v1 is inside v2
        for p in v1:
            if point_in_hexagon(p[0], p[1], c2, a2):
                return True
                
        # Check if any vertex of v2 is inside v1
        for p in v2:
            if point_in_hexagon(p[0], p[1], c1, a1):
                return True
                
        # Check if any edges intersect
        for i in range(6):
            p1a = v1[i]
            p1b = v1[(i + 1) % 6]
            for j in range(6):
                p2a = v2[j]
                p2b = v2[(j + 1) % 6]
                if segments_intersect(p1a, p1b, p2a, p2b):
                    return True
        return False

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
            # Check against 6 edges of outer hexagon
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
        # params: 2*N positions + N angles + 1 side
        # Outer center is fixed at (0,0), rotation 0.
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(pos[i], angs[i], pos[j], angs[j]):
                    # Add penalty. We want to push them apart.
                    # Calculate a rough penalty based on center distance?
                    # Or just a large constant.
                    penalty += 100.0
                    
        # Check boundary
        for i in range(N):
            viol = boundary_violation(pos[i], angs[i], s)
            if viol > 0:
                penalty += 1000.0 * viol
        
        return s + penalty

    # Initialization
    # Try a few different initial configurations and pick the best one
    best_solution = None
    best_score = float('inf')
    
    # Configuration 1: 3-4-4 grid
    # Configuration 2: 4-4-3 grid
    # Configuration 3: Random
    
    configs = []
    
    # 3-4-4
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
    # Center
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # 4-3-4
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
    
    # Run DE for each config
    for idx, (p_init, a_init) in enumerate(configs):
        bounds = []
        for i in range(N):
            bounds.append((-5, 5))
            bounds.append((-5, 5))
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
                maxiter=100,
                popsize=20,
                tol=1e-5,
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
        # Fallback
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2  # inradius of unit hexagon
    
    def get_hex_vertices(center, angle_deg, side=1.0):
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
        cx, cy = center
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        inr = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = lx * nx + ly * ny
            if proj > inr + 1e-9:
                return False
        return True
    
    def segments_intersect(p1, p2, p3, p4):
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
    
    def hexagons_overlap(c1, a1, c2, a2):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.hypot(dx, dy)
        if dist > 2.0 + 1e-9:
            return False
        v1 = get_hex_vertices(c1, a1)
        v2 = get_hex_vertices(c2, a2)
        for p in v1:
            if point_in_hexagon(p[0], p[1], c2, a2):
                return True
        for p in v2:
            if point_in_hexagon(p[0], p[1], c1, a1):
                return True
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
    
    def compute_min_side(pos, angs):
        """Compute minimum side length needed to contain all hexagons at given positions/angles."""
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_vertices(pos[i], angs[i])
            for p in verts:
                for k in range(6):
                    theta = math.radians(30 + 60 * k)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = p[0] * nx + p[1] * ny
                    if proj > max_proj:
                        max_proj = proj
        return max_proj * 2 / sqrt3
    
    def objective(params):
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(pos[i], angs[i], pos[j], angs[j]):
                    penalty += 50.0
        
        # Compute actual required side
        s_required = compute_min_side(pos, angs)
        
        # Add boundary penalty if hexagons extend beyond a reasonable bound
        # We'll use s_required as our target and add penalty for large s
        if s_required > 6.0:
            penalty += 100.0 * (s_required - 6.0)
        
        return s_required + penalty
    
    # Generate diverse initial configurations
    configs = []
    
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
    
    # Config 3: 4-4-3 grid
    rows = [4, 4, 3]
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
    
    # Config 4: Compact hexagonal arrangement
    # Center + 6 around + 4 more
    pos_init = [[0, 0]]
    ang_init = [0.0]
    for k in range(6):
        angle = math.radians(60 * k)
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        pos_init.append([x, y])
        ang_init.append(0.0)
    # Add 4 more in the gaps
    for k in range(4):
        angle = math.radians(60 * k + 30)
        x = 2.0 * sqrt3 * math.cos(angle)
        y = 2.0 * sqrt3 * math.sin(angle)
        pos_init.append([x, y])
        ang_init.append(0.0)
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Run DE for each config
    best_solution = None
    best_score = float('inf')
    
    for idx, (p_init, a_init) in enumerate(configs):
        bounds = []
        for i in range(N):
            bounds.append((-6, 6))
            bounds.append((-6, 6))
        for i in range(N):
            bounds.append((-180, 180))
        
        x0 = []
        for i in range(N):
            x0.append(p_init[i][0])
            x0.append(p_init[i][1])
        for i in range(N):
            x0.append(a_init[i])
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=200,
                popsize=25,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42 + idx,
                polish=True
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x.copy()
        except Exception as e:
            pass
    
    if best_solution is None:
        # Fallback
        rows = [3, 4, 4]
        y_offset = 0.0
        pos_init = []
        for i, count in enumerate(rows):
            x_start = -(count - 1) * sqrt3 / 2
            for j in range(count):
                x = x_start + j * sqrt3
                y = y_offset
                pos_init.append([x, y, 0.0])
            y_offset += 1.5
        xs = [p[0] for p in pos_init]
        ys = [p[1] for p in pos_init]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pos_init:
            p[0] -= cx
            p[1] -= cy
        return pos_init, (0.0, 0.0), 8.0, 0.0
    
    # Extract solution
    pos = []
    for i in range(N):
        pos.append((best_solution[2*i], best_solution[2*i+1]))
    angs = [best_solution[2*N + i] for i in range(N)]
    
    # Compute the actual minimum side length
    s = compute_min_side(pos, angs)
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
    
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END

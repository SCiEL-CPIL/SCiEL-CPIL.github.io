"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Returns 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def support_function_diff(c1, a1, c2, a2):
        """
        Compute minimum separation distance between two hexagons using support functions.
        Returns negative if overlapping, positive if separated.
        """
        # Quick rejection: if centers too far, no overlap
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.hypot(dx, dy)
        if dist > 2.0 + 1e-9:
            return dist - 2.0
        
        # Compute support function difference for 6 edge directions
        # For each edge normal direction, compute max projection of each hex
        min_sep = float('inf')
        
        # The 6 edge normal directions (perpendicular to edges)
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            # Support function of hex1 in direction (nx, ny)
            # = max over vertices of (v - c1) dot (nx, ny)
            # For a unit hexagon with angle a1, the support in direction theta is:
            # inradius * |cos(theta - a1 + 30k)| ... actually let's compute directly
            
            v1 = hex_vertices(c1[0], c1[1], a1)
            v2 = hex_vertices(c2[0], c2[1], a2)
            
            # Max projection of v1 in direction (nx, ny)
            s1 = max((vx - c1[0]) * nx + (vy - c1[1]) * ny for vx, vy in v1)
            # Min projection of v2 in direction (nx, ny)
            s2_min = min((vx - c2[0]) * nx + (vy - c2[1]) * ny for vx, vy in v2)
            
            # Separation in this direction
            sep = s1 - s2_min
            
            # Also check opposite direction
            s1_rev = min((vx - c1[0]) * nx + (vy - c1[1]) * ny for vx, vy in v1)
            s2_max = max((vx - c2[0]) * nx + (vy - c2[1]) * ny for vx, vy in v2)
            sep_rev = s2_max - s1_rev
            
            # The actual separation in this normal direction is max(sep, sep_rev)
            # But we want the minimum over all directions
            min_sep = min(min_sep, sep)
        
        return min_sep

    def hex_in_container(cx, cy, angle_deg, s_outer):
        """
        Check if hexagon fits in container. Returns max violation (0 if inside).
        Container is centered at origin, rotation 0.
        """
        inradius_outer = s_outer * sqrt3 / 2
        v = hex_vertices(cx, cy, angle_deg)
        max_viol = 0.0
        
        for px, py in v:
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
        """
        Objective: minimize outer side length + penalties.
        params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, s]
        """
        pos = []
        angs = []
        for i in range(N):
            pos.append([params[3*i], params[3*i+1]])
            angs.append(params[3*i+2])
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
        
        penalty = 0.0
        
        # Pairwise overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                sep = support_function_diff(pos[i], angs[i], pos[j], angs[j])
                if sep < 0:
                    penalty += 1000.0 * sep * sep
        
        # Container boundary violations
        for i in range(N):
            viol = hex_in_container(pos[i][0], pos[i][1], angs[i], s)
            if viol > 0:
                penalty += 1000.0 * viol * viol
        
        return s + penalty

    # Multi-start with different initial configurations
    best_solution = None
    best_score = float('inf')
    
    # Generate diverse initial configurations
    configs = []
    
    # Config 1: Hexagonal lattice packing
    pos_init = []
    ang_init = []
    # 3-4-4 arrangement
    rows = [3, 4, 4]
    y_offset = 0.0
    for count in rows:
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
    
    # Config 2: 4-3-4 arrangement
    pos_init = []
    ang_init = []
    rows = [4, 3, 4]
    y_offset = 0.0
    for count in rows:
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
    
    # Config 3: Random with some structure
    np.random.seed(123)
    pos_init = []
    ang_init = []
    for i in range(N):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        a = np.random.choice([0, 30, 60, 90, 120, 150])
        pos_init.append([x, y])
        ang_init.append(a)
    configs.append((pos_init, ang_init))
    
    # Run differential evolution for each config
    for idx, (p_init, a_init) in enumerate(configs):
        bounds = []
        for i in range(N):
            bounds.append((-5, 5))  # x
            bounds.append((-5, 5))  # y
            bounds.append((-180, 180))  # angle
        bounds.append((2.5, 10.0))  # side length
        
        x0 = []
        for i in range(N):
            x0.append(p_init[i][0])
            x0.append(p_init[i][1])
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
                maxiter=150,
                popsize=25,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
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
    angs = []
    for i in range(N):
        pos.append([best_solution[3*i], best_solution[3*i+1]])
        angs.append(best_solution[3*i+2])
    s = best_solution[3*N]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END

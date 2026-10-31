"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
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
        
        inradius = side * sqrt3 / 2
        
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = lx * nx + ly * ny
            if proj > inradius + 1e-9:
                return False
        return True

    def vertex_depth_in_hex(px, py, center, angle_deg, side=1.0):
        """
        Returns the signed distance from point (px, py) to the boundary of the hexagon.
        Positive if outside, negative if inside.
        """
        cx, cy = center
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        inradius = side * sqrt3 / 2
        
        min_dist_to_boundary = float('inf')
        # The distance to the boundary is the minimum over the 6 edges.
        # However, for a convex polygon, the signed distance is:
        # If inside, it's -min(proj - inradius) ?? No.
        # The condition for being inside is proj <= inradius for all 6 normals.
        # The "depth" of penetration is max(0, max_k (proj_k - inradius)).
        # Wait, if it's inside, all proj_k <= inradius.
        # The distance to the boundary from an interior point is min_k (inradius - proj_k).
        # So signed distance = min_k (inradius - proj_k) if inside, else max_k (proj_k - inradius) if outside?
        # Actually, for a regular hexagon, the distance to the boundary from a point is:
        # d = min_k (inradius - proj_k) if point is inside.
        # If point is outside, it's outside at least one edge. The distance is min_k (proj_k - inradius) for the violated edges?
        # Let's just compute the distance to each of the 6 lines containing the edges.
        # The distance to the line with normal n_k and offset inradius is |proj_k - inradius|.
        # But we need to ensure we are measuring to the *edge segment*, not the infinite line.
        # For small penetrations, the line distance is a good approximation.
        # For robustness, let's check if the point is inside.
        is_inside = True
        max_viol = 0.0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = lx * nx + ly * ny
            if proj > inradius:
                is_inside = False
                viol = proj - inradius
                if viol > max_viol:
                    max_viol = viol
        
        if not is_inside:
            # Outside: distance is the minimum distance to any edge segment.
            # Approximation: use the max violation? No, that's the depth of the worst edge.
            # The actual distance to the polygon boundary is the min distance to any edge.
            # Let's compute distance to the 6 edge segments.
            verts = get_hex_vertices(center, angle_deg, side)
            min_d = float('inf')
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1)%6]
                # Distance from (lx, ly) to segment p1-p2 in local coords?
                # No, (lx, ly) is local. p1, p2 are global? No, get_hex_vertices returns global.
                # Let's work in local coords for the point, but verts are global.
                # Transform verts to local? Or transform point to global?
                # We already have lx, ly.
                # Let's get local vertices.
                # Actually, easier: transform point to global, then compute distance to global edges.
                gx = cx + lx * math.cos(math.radians(angle_deg)) - ly * math.sin(math.radians(angle_deg))
                gy = cy + lx * math.sin(math.radians(angle_deg)) + ly * math.cos(math.radians(angle_deg))
                # This is just (px, py).
                # So we can just compute distance from (px, py) to the 6 global edges.
                for i in range(6):
                    p1 = verts[i]
                    p2 = verts[(i+1)%6]
                    # Vector p1 to p2
                    dx_e = p2[0] - p1[0]
                    dy_e = p2[1] - p1[1]
                    len_sq = dx_e*dx_e + dy_e*dy_e
                    if len_sq == 0: continue
                    t = ((px - p1[0])*dx_e + (py - p1[1])*dy_e) / len_sq
                    t = max(0, min(1, t))
                    proj_x = p1[0] + t * dx_e
                    proj_y = p1[1] + t * dy_e
                    dist = math.hypot(px - proj_x, py - proj_y)
                    if dist < min_d:
                        min_d = dist
            return min_d
        else:
            # Inside: distance is min_k (inradius - proj_k)
            min_d = float('inf')
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = lx * nx + ly * ny
                d = inradius - proj
                if d < min_d:
                    min_d = d
            return -min_d # Negative because inside

    def boundary_violation_smooth(c, a, s_outer):
        """Returns sum of squared boundary violations for one hexagon."""
        verts = get_hex_vertices(c, a)
        inradius_outer = s_outer * sqrt3 / 2
        total_viol = 0.0
        for p in verts:
            px, py = p
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = px * nx + py * ny
                if proj > inradius_outer:
                    viol = proj - inradius_outer
                    total_viol += viol * viol
        return total_viol

    def overlap_penalty_smooth(c1, a1, c2, a2):
        """
        Returns sum of squared penetration depths for vertices of one hex in the other.
        """
        v1 = get_hex_vertices(c1, a1)
        v2 = get_hex_vertices(c2, a2)
        
        pen1 = 0.0
        for p in v1:
            d = vertex_depth_in_hex(p[0], p[1], c2, a2)
            if d < 0:
                pen1 += (-d) * (-d)
                
        pen2 = 0.0
        for p in v2:
            d = vertex_depth_in_hex(p[0], p[1], c1, a1)
            if d < 0:
                pen2 += (-d) * (-d)
                
        return pen1 + pen2

    def objective_de(params):
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Hard overlap check
        for i in range(N):
            for j in range(i+1, N):
                # Quick check
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                if math.hypot(dx, dy) < 1.8: # Loose bound
                    if overlap_penalty_smooth(pos[i], angs[i], pos[j], angs[j]) > 1e-6:
                        penalty += 100.0
                        
        # Hard boundary check
        for i in range(N):
            if boundary_violation_smooth(pos[i], angs[i], s) > 1e-6:
                penalty += 1000.0

        return s + penalty

    def objective_refine(params):
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Smooth overlap
        for i in range(N):
            for j in range(i+1, N):
                penalty += 10.0 * overlap_penalty_smooth(pos[i], angs[i], pos[j], angs[j])
                
        # Smooth boundary
        for i in range(N):
            penalty += 10.0 * boundary_violation_smooth(pos[i], angs[i], s)

        return s + penalty

    # Initial configuration: 3-4-4 grid centered
    def get_initial_config():
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
        return pos_init, ang_init

    pos_init, ang_init = get_initial_config()
    
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
    for i in range(N):
        bounds.append((-180, 180))
    bounds.append((2.5, 10.0))
    
    x0 = []
    for i in range(N):
        x0.append(pos_init[i][0])
        x0.append(pos_init[i][1])
    for i in range(N):
        x0.append(ang_init[i])
        
    max_dist = 0
    for p in pos_init:
        max_dist = max(max_dist, math.hypot(p[0], p[1]))
    x0.append(max_dist + 1.5)

    # Run DE
    try:
        result_de = differential_evolution(
            objective_de,
            bounds,
            x0=x0,
            maxiter=200,
            popsize=30,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42
        )
        best_x = result_de.x
    except Exception:
        best_x = x0

    # Refine with Powell
    try:
        result_refine = minimize(
            objective_refine,
            best_x,
            method='Powell',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if result_refine.fun < objective_refine(best_x):
            best_x = result_refine.x
    except Exception:
        pass

    # Extract solution
    pos = []
    for i in range(N):
        pos.append([best_x[2*i], best_x[2*i+1]])
    angs = [best_x[2*N + i] for i in range(N)]
    s = best_x[3*N]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Helper: Calculate support function of a regular hexagon with side 1, center c, angle a, in direction u
    # h(u) = max_{v in vertices} dot(v, u)
    # Vertices of unit hexagon (side 1) are at distance 1 from center.
    # Vertex k is at angle (a + 60*k) degrees.
    # h(u) = max_k (dot(c, u) + dot(vertex_k, u)) = dot(c, u) + max_k dot(vertex_k, u)
    # For a regular hexagon, max dot(vertex_k, u) depends on the angle of u relative to the hexagon.
    # However, for the Minkowski sum/difference approach, we can precompute or compute on the fly.
    
    def support_hex(c, a_deg, u):
        """
        c: center (x, y)
        a_deg: rotation angle in degrees
        u: unit vector (ux, uy)
        Returns max dot(v, u) for vertices v of the hexagon.
        """
        ux, uy = u
        # The maximum projection of a regular hexagon (side 1) onto a direction u
        # is given by the support function.
        # For a hexagon with vertices at angles theta_k = a + 60*k,
        # h(u) = max_k (c_x ux + c_y uy + cos(theta_k)ux + sin(theta_k)uy)
        # = c_x ux + c_y uy + max_k (cos(theta_k)ux + sin(theta_k)uy)
        # The term max_k (cos(theta_k)ux + sin(theta_k)uy) is the support function of the unit hexagon centered at origin.
        # This value is periodic with period 60 degrees.
        # We can compute it by checking the 6 vertices.
        base_angle = math.radians(a_deg)
        max_proj = 0.0
        for k in range(6):
            theta = base_angle + math.radians(60 * k)
            vx = math.cos(theta)
            vy = math.sin(theta)
            proj = vx * ux + vy * uy
            if proj > max_proj:
                max_proj = proj
        return c[0] * ux + c[1] * uy + max_proj

    def check_overlap(c1, a1, c2, a2):
        """
        Check if two unit hexagons with centers c1, c2 and angles a1, a2 overlap.
        They are disjoint if for all directions u, h1(u) + h2(-u) <= dot(c2-c1, u) is NOT the right condition.
        Correct condition for disjointness of convex sets A, B:
        min_{u} (h_A(u) + h_B(-u) - dot(c_B - c_A, u)) >= 0 ? No.
        
        The distance between two convex sets A and B is:
        d(A, B) = min_{a in A, b in B} ||a - b||
        This is equivalent to:
        min_{||u||=1} (h_A(u) + h_B(-u) - dot(c_B - c_A, u)) ? No, that's for the Minkowski difference.
        
        Let's use the property: Two convex polygons P and Q are disjoint iff there exists a separating line.
        A separating line is perpendicular to some edge normal of P or Q.
        So we only need to check the edge normals of P and Q.
        
        For a regular hexagon, the edge normals are at angles:
        n_k = a + 60*k + 30 (if vertices are at a+60k, edges are between them)
        Wait, vertices are at angles a + 60k.
        The edge between vertex k and k+1 is perpendicular to the angle bisector, which is a + 60k + 30.
        So the outward normals are at angles a + 30 + 60k.
        
        We check separation along these 12 directions (6 from hex 1, 6 from hex 2).
        """
        c1x, c1y = c1
        c2x, c2y = c2
        dx = c2x - c1x
        dy = c2y - c1y
        
        # Edge normals for hex 1
        base1 = math.radians(a1) + math.radians(30)
        # Edge normals for hex 2
        base2 = math.radians(a2) + math.radians(30)
        
        for k in range(6):
            # Normal from hex 1
            ang1 = base1 + math.radians(60 * k)
            ux = math.cos(ang1)
            uy = math.sin(ang1)
            # h1(u) is max dot(v1, u)
            # h2(-u) is max dot(v2, -u) = - min dot(v2, u)
            # Distance along u is dot(c2-c1, u)
            # Separation condition: h1(u) + h2(-u) <= dot(c2-c1, u)
            # Let's compute h1(u) and h2(-u)
            
            # h1(u) = dot(c1, u) + max_{v1} dot(v1, u)
            # h2(-u) = dot(c2, -u) + max_{v2} dot(v2, -u)
            
            # Pre-compute max dot for unit hexagon in direction u
            def max_dot_unit_hex(a_deg, ux, uy):
                base = math.radians(a_deg)
                m = 0.0
                for j in range(6):
                    theta = base + math.radians(60 * j)
                    vx = math.cos(theta)
                    vy = math.sin(theta)
                    p = vx * ux + vy * uy
                    if p > m: m = p
                return m
            
            m1 = max_dot_unit_hex(a1, ux, uy)
            m2_neg = max_dot_unit_hex(a2, -ux, -uy) # max dot(v2, -u)
            
            h1 = c1x * ux + c1y * uy + m1
            h2_neg = c2x * (-ux) + c2y * (-uy) + m2_neg
            
            # The projection of the vector from c1 to c2 onto u is dot(c2-c1, u)
            dist_proj = dx * ux + dy * uy
            
            # If h1 + h2_neg > dist_proj, they overlap in this direction
            # Actually, the condition for NO overlap is that there exists a u such that h1(u) + h2(-u) < dot(c2-c1, u) is FALSE?
            # No.
            # Let's use the standard formula:
            # Two convex bodies A, B are disjoint if and only if:
            # min_{||u||=1} (h_A(u) + h_B(-u) - dot(c_B - c_A, u)) > 0 ? No.
            
            # Let's stick to the separating axis theorem.
            # Project all vertices of A and B onto the axis u.
            # If the intervals [min_A, max_A] and [min_B, max_B] are disjoint, they are separated.
            # Since they are convex, if they are disjoint, there is a separating axis parallel to an edge of A or B.
            
            # Interval A on u: [min_{v in A} dot(v, u), max_{v in A} dot(v, u)]
            # min_A = dot(c1, u) + min_{v1} dot(v1, u) = dot(c1, u) - max_{v1} dot(v1, -u)
            # max_A = dot(c1, u) + max_{v1} dot(v1, u)
            
            # Interval B on u: [min_B, max_B]
            # min_B = dot(c2, u) - max_{v2} dot(v2, -u)
            # max_B = dot(c2, u) + max_{v2} dot(v2, u)
            
            # Disjoint if max_A < min_B or max_B < min_A.
            
            max_A = c1x * ux + c1y * uy + m1
            min_B = c2x * ux + c2y * uy - m2_neg # Note: min_B = dot(c2, u) + min_{v2} dot(v2, u) = dot(c2, u) - max_{v2} dot(v2, -u)
            
            if max_A < min_B - 1e-9:
                return False # Separated
            
            # Check the other direction (B before A)
            # We also need to check if max_B < min_A.
            # Due to symmetry, checking all 12 axes covers this.
            # But let's be safe and check the other side too for this axis?
            # No, if they are separated, one interval is strictly to the left or right.
            # Checking max_A < min_B covers "A is to the left of B".
            # If "B is to the left of A", then max_B < min_A.
            # Is max_B < min_A equivalent to max_A' < min_B' for axis -u?
            # Yes. So checking all 12 axes (6 from A, 6 from B) is sufficient.
            
        return True # Overlapping

    def is_inside_hex(point, center, side, angle_deg):
        """
        Check if point is inside regular hexagon with center, side, angle.
        Use support function / half-plane test.
        Point p is inside if for all edge normals n_k, dot(p - center, n_k) <= inradius.
        Edge normals are at angles angle_deg + 30 + 60*k.
        Inradius = side * sqrt(3) / 2.
        """
        cx, cy = center
        px, py = point
        dx = px - cx
        dy = py - cy
        inradius = side * sqrt3 / 2.0
        
        base = math.radians(angle_deg) + math.radians(30)
        for k in range(6):
            ang = base + math.radians(60 * k)
            nx = math.cos(ang)
            ny = math.sin(ang)
            proj = dx * nx + dy * ny
            if proj > inradius + 1e-9:
                return False
        return True

    def objective(params):
        # params: 2*N positions + N angles + 2 center + 1 side
        n = len(positions)
        pos = []
        for i in range(n):
            pos.append((params[2*i], params[2*i+1]))
        angs = [params[2*n + i] for i in range(n)]
        cx = params[3*n]
        cy = params[3*n + 1]
        s = params[3*n + 2]
        
        if s < 2.0:
            return 1e6
        
        penalty = 0.0
        
        # Check overlap
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap(pos[i], angs[i], pos[j], angs[j]):
                    # Estimate violation magnitude
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = math.hypot(dx, dy)
                    # Approximate violation
                    penalty += max(0, 1.7 - dist) * 100
                    if penalty > 1e5:
                        return 1e6 + penalty
        
        # Check containment
        for i in range(n):
            # Check all 6 vertices of inner hexagon i
            base_ang = math.radians(angs[i])
            for k in range(6):
                theta = base_ang + math.radians(60 * k)
                vx = pos[i][0] + math.cos(theta)
                vy = pos[i][1] + math.sin(theta)
                if not is_inside_hex((vx, vy), (cx, cy), s, 0.0):
                    # Estimate violation
                    # Distance from center to edge
                    dx = vx - cx
                    dy = vy - cy
                    # Find max projection violation
                    # For simplicity, just add a large penalty
                    penalty += 100.0
                    if penalty > 1e5:
                        return 1e6 + penalty
        
        return s + penalty

    # Generate multiple initial heuristics for multi-start
    def get_initial_guess(seed_idx):
        """
        Generate an initial guess for DE.
        seed_idx: 0, 1, 2...
        """
        pos = []
        angs = []
        
        if seed_idx == 0:
            # Hexagonal packing 3-4-4
            rows = [3, 4, 4]
            y_offset = 0.0
            for i, count in enumerate(rows):
                x_start = -(count - 1) * sqrt3 / 2
                for j in range(count):
                    x = x_start + j * sqrt3
                    y = y_offset
                    pos.append([x, y])
                    angs.append(0.0)
                y_offset += 1.5
        elif seed_idx == 1:
            # Circular packing
            for i in range(N):
                ang = 2 * math.pi * i / N
                r = 2.0
                pos.append([r * math.cos(ang), r * math.sin(ang)])
                angs.append(math.degrees(ang))
        elif seed_idx == 2:
            # Grid packing
            for i in range(N):
                x = (i % 3) * 2.0 - 2.0
                y = (i // 3) * 2.0 - 2.0
                pos.append([x, y])
                angs.append(0.0)
        else:
            # Random
            pos = [[(np.random.rand() - 0.5) * 6, (np.random.rand() - 0.5) * 6] for _ in range(N)]
            angs = [np.random.rand() * 360 for _ in range(N)]
        
        # Center
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pos:
            p[0] -= cx
            p[1] -= cy
        
        # Estimate s
        max_dist = 0
        for p in pos:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        s_init = max_dist + 1.5
        
        x0 = []
        for i in range(N):
            x0.append(pos[i][0])
            x0.append(pos[i][1])
        for i in range(N):
            x0.append(angs[i])
        x0.append(0.0)
        x0.append(0.0)
        x0.append(s_init)
        
        return x0

    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
    for i in range(N):
        bounds.append((-180, 180))
    bounds.append((-5, 5))
    bounds.append((-5, 5))
    bounds.append((2.5, 10.0))
    
    best_result = None
    best_fun = float('inf')
    
    # Multi-start DE
    for seed_idx in range(3):
        x0 = get_initial_guess(seed_idx)
        # Use different random seeds for DE
        res = differential_evolution(
            objective,
            bounds,
            x0=x0,
            maxiter=30,
            popsize=10,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42 + seed_idx,
            polish=True
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res
    
    if best_result is None or best_result.fun > 100:
        # Fallback
        pos = get_initial_guess(0)
        # Extract from x0
        pos_list = []
        for i in range(N):
            pos_list.append([pos[2*i], pos[2*i+1]])
        angs_list = [pos[2*N + i] for i in range(N)]
        cx = 0.0
        cy = 0.0
        s = 5.0 # Guess
        # Recalculate s properly
        max_dist = 0
        for p in pos_list:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        s = max_dist + 1.5
        inner = [[p[0], p[1], a] for p, a in zip(pos_list, angs_list)]
        return inner, (cx, cy), s, 0.0
        
    best_params = best_result.x
    pos = []
    for i in range(N):
        pos.append([best_params[2*i], best_params[2*i+1]])
    angs = [best_params[2*N + i] for i in range(N)]
    cx = best_params[3*N]
    cy = best_params[3*N + 1]
    s = best_params[3*N + 2]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
    
    return inner, (cx, cy), s, 0.0
# EVOLVE-BLOCK-END

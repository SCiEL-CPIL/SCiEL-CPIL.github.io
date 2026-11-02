import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 115.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial guess: hexagonal lattice pattern
    # For N=11, a 3x4 grid with one removed or similar
    # Let's use a standard hexagonal packing initial guess
    # Positions in Cartesian for a hex grid
    def hex_to_cart(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Define initial positions for 11 hexagons
    # A common dense packing for 11 is a 3x3 center plus 8 around, or a specific pattern
    # Let's try a 3x4 grid minus one corner
    initial_axial = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Convert to Cartesian
    init_pts = np.array([hex_to_cart(q, r) for q, r in initial_axial])
    init_angles = np.zeros(N)
    
    # Parameters: x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, D
    # Total 3N + 1 = 34 parameters
    # We minimize D (diameter of outer hexagon apothem + unit apothem)
    # Actually, minimize S = (D + APOTHEM) / APOTHEM
    # Subject to:
    # 1. Non-overlap: distance between centers >= 2 * min_dist(angle_diff)
    #    For two unit hexagons with relative rotation theta, the minimum distance
    #    to avoid overlap is 2 * (1 + cos(theta/2)) / (1 + 2*cos(theta/2))? No.
    #    The Minkowski sum of two hexagons is a hexagon (or dodecagon if rotated).
    #    The support function of the Minkowski sum is the sum of support functions.
    #    h_{A+B}(u) = h_A(u) + h_B(u).
    #    For a unit hexagon centered at origin, h(u) = max_{v in V} v . u.
    #    If hexagon A is at p_a with angle a, and B at p_b with angle b.
    #    Overlap if (p_b - p_a) is inside the Minkowski difference (which is same as sum for symmetric shapes).
    #    So we need ||p_b - p_a||_u >= h_A(u) + h_B(u) for all unit u?
    #    Actually, non-overlap means the distance between centers in every direction u is at least the sum of support functions.
    #    This is equivalent to: for all u, (p_b - p_a) . u >= h_A(u) AND (p_a - p_b) . u >= h_B(u)? No.
    #    The condition is: p_b - p_a is NOT in the interior of (A - B) = A + (-B).
    #    Since A and B are centrally symmetric, -B = B (rotated).
    #    So we need p_b - p_a to be outside the Minkowski sum of A and B (rotated).
    #    This is a non-convex constraint generally, but for hexagons it's manageable.
    #    
    #    Simpler approach: Use the fact that the "distance" between two hexagons is the minimum distance between their boundaries.
    #    We can enforce that the distance between centers is greater than the sum of their "radii" in the direction of the center vector.
    #    Let v = p_b - p_a. Let d = ||v||. Let u = v/d.
    #    The support function of hexagon A in direction u is h_A(u).
    #    The support function of hexagon B in direction -u is h_B(-u).
    #    If d >= h_A(u) + h_B(-u), they don't overlap in that direction.
    #    But this is only sufficient, not necessary, unless we check all directions?
    #    Actually, for convex sets, A and B are disjoint iff for all u, (p_b - p_a) . u >= h_A(u) - h_B(-u) ? No.
    #    A and B are disjoint iff 0 is not in (A - B).
    #    0 is in (A - B) iff for all u, h_{A-B}(u) >= 0.
    #    h_{A-B}(u) = h_A(u) + h_B(-u).
    #    So they are disjoint iff there exists a direction u such that (p_b - p_a) . u > h_A(u) + h_B(-u) is FALSE?
    #    No, they are disjoint iff the distance from 0 to (A-B) is > 0.
    #    This is equivalent to: min_{||u||=1} ( (p_b - p_a) . u - h_A(u) - h_B(-u) ) > 0.
    #    
    #    This is a global constraint. To make it differentiable, we can use a smooth approximation of the min.
    #    Or, we can use the fact that for hexagons, the "critical" directions are the normals to the edges.
    #    There are 6+6=12 critical directions (normals of A and B).
    #    If we satisfy the constraint for these 12 directions, is it sufficient?
    #    Yes, because the support function is convex and piecewise linear. The minimum of a linear function over a polygon occurs at a vertex, or the constraint is violated at a vertex of the Minkowski sum.
    #    Actually, the condition "p is outside K" is equivalent to "exists u such that p.u > h_K(u)".
    #    We want "p is outside K", so we want "exists u such that p.u > h_K(u)".
    #    This is a disjunction, which is hard.
    #    Instead, we can minimize the violation: max_u (h_K(u) - p.u). We want this to be <= 0.
    #    So we add a penalty: max(0, max_u (h_K(u) - p.u))^2.
    #    
    #    Let's implement this.
    #    h_K(u) where K = A + B. h_K(u) = h_A(u) + h_B(u).
    #    h_A(u) for a hexagon with vertices v_i: max_i v_i . u.
    #    We can compute this analytically or by checking the 6 vertices.
    
    def hex_support(vertices, u):
        # vertices: (6, 2) array
        # u: (2,) array
        dots = vertices @ u
        return np.max(dots)
    
    def get_hex_vertices(angle):
        # Vertices of a unit regular hexagon centered at origin, rotated by angle
        # Side length 1.
        # Vertices at distance 1 from center? No, side length 1 means circumradius 1.
        # Yes, for regular hexagon, side = circumradius.
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def non_overlap_penalty(p1, a1, p2, a2):
        v = p2 - p1
        # We need to check if v is outside K = A + B.
        # This is equivalent to: min_{||u||=1} (v . u - h_A(u) - h_B(u)) > 0.
        # Let f(u) = v . u - h_A(u) - h_B(u).
        # We want min f(u) > 0.
        # Penalty is max(0, -min f(u))^2.
        
        # Sample directions? Or use the normals?
        # The function f(u) is concave in u? No, h is convex, so -h is concave. v.u is linear.
        # So f(u) is concave. The minimum of a concave function on a circle is at the boundary?
        # Actually, we want to find if f(u) <= 0 for all u.
        # If f(u) > 0 for some u, they are separated.
        # If f(u) <= 0 for all u, they overlap.
        # So the "depth of overlap" is max(0, -min f(u)).
        
        # To compute min f(u), we can use the fact that the minimum occurs at a direction where the subgradient of h_A + h_B contains v.
        # Or simply sample the 12 edge normals of A and B.
        
        verts_a = get_hex_vertices(a1)
        verts_b = get_hex_vertices(a2)
        
        # Edge normals of A
        # The normal to edge between v_i and v_{i+1} is perpendicular to v_{i+1}-v_i.
        # For a regular hexagon, the normals are at angles a + pi/6 + k*pi/3.
        normals_a = np.column_stack([np.cos(a1 + np.pi/6 + np.arange(6)*np.pi/3),
                                     np.sin(a1 + np.pi/6 + np.arange(6)*np.pi/3)])
        normals_b = np.column_stack([np.cos(a2 + np.pi/6 + np.arange(6)*np.pi/3),
                                     np.sin(a2 + np.pi/6 + np.arange(6)*np.pi/3)])
        
        all_normals = np.vstack([normals_a, normals_b])
        
        # Also check the direction of v itself, and maybe some intermediate ones?
        # The minimum of f(u) = v.u - h_A(u) - h_B(u) on the unit circle.
        # Since h_A and h_B are piecewise linear, f is piecewise linear in u (if we parametrize by angle, it's not linear).
        # But the minimum of v.u - h_K(u) is related to the distance from v to K.
        # Actually, min_{||u||=1} (v.u - h_K(u)) = - dist(v, K) if v is outside? No.
        # If v is outside K, there exists u such that v.u > h_K(u).
        # The "slack" is max_u (v.u - h_K(u)).
        # We want slack > 0.
        # So penalty is max(0, -max_u (v.u - h_K(u)))^2.
        
        # Let's compute g(u) = v.u - h_A(u) - h_B(u) for the candidate normals.
        # Is the maximum of g(u) attained at one of the normals?
        # g(u) is concave in u (on the circle). The maximum of a concave function is at the interior or boundary.
        # Actually, h_K is convex, so -h_K is concave. v.u is linear.
        # The maximum of a concave function on a convex set (circle is not convex) ...
        # Let's just evaluate at the 12 normals and the direction of v.
        
        candidates_u = all_normals
        if np.linalg.norm(v) > 1e-9:
            u_v = v / np.linalg.norm(v)
            candidates_u = np.vstack([candidates_u, u_v])
        
        vals = []
        for u in candidates_u:
            h_a = hex_support(verts_a, u)
            h_b = hex_support(verts_b, u)
            val = v @ u - h_a - h_b
            vals.append(val)
        
        max_val = max(vals)
        # If max_val > 0, they are separated.
        # If max_val <= 0, they overlap.
        # Penalty is (max(0, -max_val))^2.
        if max_val < 0:
            return (-max_val) ** 2
        else:
            return 0.0

    def outer_hex_penalty(params, D):
        # params: 3N floats
        # D: apothem of outer hexagon
        # We need all hexagons to be inside the outer hexagon.
        # Outer hexagon centered at (0,0) for simplicity? 
        # The task allows translating the outer hexagon.
        # Let's fix the outer hexagon center at (0,0) by symmetry?
        # No, the inner hexagons can be shifted.
        # Let's add outer center (cx, cy) and outer angle (theta) to the optimization?
        # That adds 3 more parameters.
        # Total 3N + 3 + 1 (D) = 37 parameters.
        
        # Actually, we can just optimize the inner positions relative to a fixed outer hexagon?
        # No, the score is the side length. We want to minimize it.
        # So D is the variable.
        # We need to ensure all inner hexagons are inside the outer hexagon.
        # For each inner hexagon i, its vertices must be inside the outer hexagon.
        # This is equivalent to: for each vertex v of inner hex i, and for each of the 6 outer normals u_k:
        # (v - c_outer) . u_k <= D_outer_apothem.
        
        # Let's include outer center and angle in the parameters.
        # P = [x1, y1, a1, ..., xN, yN, aN, cx, cy, theta, D]
        
        pass

    # Let's restructure the optimization.
    # Variables: 
    # 1. Inner hex centers: x_i, y_i (2N)
    # 2. Inner hex angles: a_i (N)
    # 3. Outer hex center: cx, cy (2)
    # 4. Outer hex angle: theta (1)
    # 5. Outer hex apothem: D (1)
    # Total: 3N + 4 = 37 variables.
    
    def objective(params):
        # params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, theta, D]
        D = params[-1]
        theta = params[-2]
        cy = params[-3]
        cx = params[-4]
        
        # Extract inner hexagons
        inner_pts = []
        inner_angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_pts.append([x, y])
            inner_angles.append(a)
        
        inner_pts = np.array(inner_pts)
        inner_angles = np.array(inner_angles)
        
        penalty = 0.0
        
        # 1. Non-overlap penalty
        for i in range(N):
            for j in range(i+1, N):
                p1 = inner_pts[i]
                p2 = inner_pts[j]
                a1 = inner_angles[i]
                a2 = inner_angles[j]
                penalty += non_overlap_penalty(p1, a1, p2, a2)
        
        # 2. Containment penalty
        # Outer hexagon vertices/normals
        # Normals of outer hexagon
        outer_normals = np.column_stack([np.cos(theta + np.pi/6 + np.arange(6)*np.pi/3),
                                         np.sin(theta + np.pi/6 + np.arange(6)*np.pi/3)])
        
        for i in range(N):
            verts = get_hex_vertices(inner_angles[i]) + inner_pts[i]
            for v in verts:
                rel_v = v - np.array([cx, cy])
                for u in outer_normals:
                    proj = rel_v @ u
                    if proj > D:
                        penalty += (proj - D) ** 2
        
        # 3. Regularization to keep D small (handled by minimizing D in params? No, D is a variable)
        # We want to minimize D, but the penalty constraints must be satisfied.
        # So the objective should be D + penalty.
        return D + penalty

    # Initial guess
    init_params = []
    for i in range(N):
        init_params.append(init_pts[i][0])
        init_params.append(init_pts[i][1])
        init_params.append(0.0)
    init_params.extend([0.0, 0.0, 0.0, 5.0]) # cx, cy, theta, D
    
    # Bounds
    # D must be positive.
    bounds = [(None, None)] * (3*N + 3) + [(0.1, None)]
    
    # Start optimization
    best_result = None
    best_obj = float('inf')
    
    # Multiple starts
    starts = [init_params]
    # Add some random starts
    for _ in range(5):
        rand_params = []
        for i in range(N):
            rand_params.append(np.random.uniform(-2, 2))
            rand_params.append(np.random.uniform(-2, 2))
            rand_params.append(np.random.uniform(0, np.pi/3))
        rand_params.extend([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0, 5.0])
        starts.append(rand_params)
    
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        try:
            res = minimize(objective, start, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Extract solution
    params = best_result.x
    D = params[-1]
    theta = params[-2]
    cy = params[-3]
    cx = params[-4]
    
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        # Convert angle to degrees
        a_deg = math.degrees(a)
        inner.append([float(x), float(y), float(a_deg)])
    
    center = [float(cx), float(cy)]
    side = (D + APOTHEM) / APOTHEM
    rotation = float(math.degrees(theta))
    
    return inner, center, side, rotation

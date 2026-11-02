import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Hexagon vertices in local coordinates (unit hexagon, side=1)
    def hex_vertices(theta):
        # 6 vertices of a regular hexagon with side 1, rotated by theta
        angles = theta + np.arange(6) * math.pi / 3.0
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def hex_apothem_directions(theta):
        # 6 outward normal directions (apothem directions)
        angles = theta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def distance_between_hexagons(x1, y1, theta1, x2, y2, theta2):
        """
        Compute the minimum distance between two unit hexagons.
        Uses support function approach: for convex polygons, the minimum distance
        can be computed via the Minkowski difference.
        
        For two regular hexagons, the distance is the maximum over all pairs of
        (support function of A - support function of B) projected onto directions.
        
        Actually, for non-overlapping convex bodies, we need to ensure that
        the Minkowski difference does not contain the origin.
        
        A simpler approach: compute the maximum of the support functions of the
        Minkowski difference A - B in all directions. If this max is <= 0, they overlap.
        The actual distance is related to how far the origin is from the Minkowski difference.
        
        For regular hexagons, we can use the fact that the support function of a hexagon
        with center c, rotation theta is:
        h(u) = c · u + APOTHEM * |u · n_k| max over the 6 face normals n_k
        
        Actually, for a regular hexagon with side 1 and apothem a, the support function in
        direction u is:
        h(u) = c · u + a * max_k (u · n_k) where n_k are the 6 outward unit normals
        
        Wait, that's not quite right. Let me think more carefully.
        
        For a regular hexagon centered at origin with rotation theta, the support function
        in direction (cos phi, sin phi) is:
        h(phi) = APOTHEM * max_k cos(phi - (theta + pi/6 + k*pi/3))
        
        For the Minkowski difference A - B, the support function is h_A(u) + h_B(-u).
        The two bodies are disjoint if and only if the origin is not in A - B, i.e.,
        h_{A-B}(u) > 0 for all u... no, that's not right either.
        
        The bodies A and B are disjoint if and only if the origin is not in A - B.
        For convex bodies, this is equivalent to: there exists a direction u such that
        h_A(u) < -h_B(u), i.e., h_A(u) + h_B(-u) < 0... 
        
        Actually, A and B are disjoint iff the origin is not in A - B iff
        for all u, h_{A-B}(u) >= 0 doesn't work.
        
        Let me use a different approach. The minimum distance between two convex polygons
        can be found by checking all edge-vertex and vertex-vertex pairs, or by using
        the fact that for regular hexagons, the distance is determined by the 12 
        combinations of face normals.
        
        For simplicity and efficiency, I'll compute the minimum distance by sampling
        the support function of the Minkowski difference in a fine grid of directions,
        and finding the minimum. If the minimum is > 0, the distance is that minimum.
        """
        # Support function of hexagon A in direction u
        def support_A(u_angle):
            # u_angle is the angle of direction u
            max_val = -np.inf
            for k in range(6):
                n_angle = theta1 + math.pi / 6.0 + k * math.pi / 3.0
                proj = math.cos(u_angle - n_angle)
                max_val = max(max_val, proj)
            return x1 * math.cos(u_angle) + y1 * math.sin(u_angle) + APOTHEM * max_val

        def support_B_neg(u_angle):
            # Support function of -B in direction u, which is -h_B(-u)
            # h_B(v) = x2*cos(v) + y2*sin(v) + APOTHEM * max_k cos(v - n2_k)
            # -h_B(-u) = -[x2*cos(-u) + y2*sin(-u) + APOTHEM * max_k cos(-u - n2_k)]
            # = -x2*cos(u) + y2*sin(u) - APOTHEM * max_k cos(u + n2_k)... 
            # Wait, cos(-u - n) = cos(u + n)
            # So -h_B(-u) = -x2*cos(u) + y2*sin(u) - APOTHEM * max_k cos(u + n2_k)
            # Hmm, let me just compute h_B(-u) and negate.
            v_angle = u_angle + math.pi  # -u direction
            max_val = -np.inf
            for k in range(6):
                n_angle = theta2 + math.pi / 6.0 + k * math.pi / 3.0
                proj = math.cos(v_angle - n_angle)
                max_val = max(max_val, proj)
            h_B = x2 * math.cos(v_angle) + y2 * math.sin(v_angle) + APOTHEM * max_val
            return -h_B

        # The support function of A - B in direction u is h_A(u) + h_{-B}(u) = h_A(u) - h_B(-u)
        # A and B are disjoint iff 0 not in A - B iff min_u (h_A(u) + h_{-B}(u)) > 0
        # The minimum distance is max(0, min_u (h_A(u) + h_{-B}(u)))... 
        # Actually, the distance from origin to A-B is min_{p in A-B} |p|, and for convex sets
        # containing the origin in their interior if they overlap, the distance is 
        # min_u h_{A-B}(u) when this is positive... 
        
        # Let me just compute min over directions of the support function.
        # Sample 360 directions
        angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
        min_support = np.inf
        for ang in angles:
            s = support_A(ang) + support_B_neg(ang)
            if s < min_support:
                min_support = s
        
        return min_support

    def total_distance(params, positions, thetas):
        """
        params: [cx, cy, S, beta, x0, y0, x1, y1, ..., xN-1, yN-1, theta0, theta1, ..., thetaN-1]
        """
        cx, cy = params[0], params[1]
        S = params[2]
        beta = params[3]
        
        # Outer hexagon apothem
        outer_apothem = S * APOTHEM
        
        # Check if all inner hexagons are inside the outer hexagon
        # For each inner hexagon, check its support function in the 6 outer normal directions
        outer_normals = hex_apothem_directions(beta)
        
        for i in range(N):
            xi, yi = positions[i]
            theta_i = thetas[i]
            for k in range(6):
                nx, ny = outer_normals[k]
                # Support function of inner hex i in direction (nx, ny)
                max_proj = -np.inf
                for j in range(6):
                    n_angle = theta_i + math.pi / 6.0 + j * math.pi / 3.0
                    proj = nx * math.cos(n_angle) + ny * math.sin(n_angle)
                    max_proj = max(max_proj, proj)
                # Distance from inner hex center to outer hex boundary in this direction
                dist_to_boundary = outer_apothem - (xi * nx + yi * ny + APOTHEM * max_proj)
                if dist_to_boundary < -EPS:
                    return -dist_to_boundary + 1.0  # Penalize going outside
        
        # Check pairwise distances between inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                d = distance_between_hexagons(positions[i][0], positions[i][1], thetas[i],
                                              positions[j][0], positions[j][1], thetas[j])
                if d < -EPS:
                    return -d + 1.0  # Penalize overlap
        
        # If all constraints satisfied, return 0 (or a small positive value)
        return 0.0

    def objective(params):
        """
        Objective: minimize S (the outer side length)
        params: [cx, cy, S, beta, x0, y0, ..., xN-1, yN-1, theta0, ..., thetaN-1]
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        S = params[2]
        if S < 1.0:
            S = 1.0
        
        positions = np.array([[params[4 + 2*i], params[5 + 2*i]] for i in range(N)])
        thetas = np.array([params[4 + 2*N + i] for i in range(N)])
        
        # Penalty for constraint violations
        penalty = total_distance(params, positions, thetas)
        if penalty > 0:
            return S + 1000.0 * penalty
        
        return S

    # Initial candidate configurations in axial coordinates
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate a few good initial configurations
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]

    axial_4 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]

    candidates_axial = [axial_1, axial_2, axial_3, axial_4, axial_5]
    
    best_S = float('inf')
    best_params = None
    
    for axial in candidates_axial:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        
        # Estimate initial S
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx0) * nx + (pts[:, 1] - cy0) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S0 = (max_proj + APOTHEM) / APOTHEM
        
        # Initial parameters
        initial = [cx0, cy0, S0, 0.0]
        for i in range(N):
            initial.append(pts[i][0])
            initial.append(pts[i][1])
        for i in range(N):
            initial.append(0.0)  # initial rotation
        
        # Optimize
        try:
            # Use differential evolution for global search
            bounds = [(-5, 5), (-5, 5), (1.0, 6.0), (-math.pi/6, math.pi/6)]
            for i in range(N):
                bounds.append((cx0 - 3, cx0 + 3))
                bounds.append((cy0 - 3, cy0 + 3))
            for i in range(N):
                bounds.append((-math.pi/6, math.pi/6))
            
            de_res = differential_evolution(objective, bounds, maxiter=50, popsize=10,
                                           tol=1e-6, seed=42, polish=False)
            
            # Refine with L-BFGS-B
            res = minimize(objective, de_res.x, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < 1e10 and res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # If no good result, fall back to a simple configuration
    if best_params is None:
        pts = axial_to_cartesian(axial_1)
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx0) * nx + (pts[:, 1] - cy0) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S0 = (max_proj + APOTHEM) / APOTHEM
        
        best_params = [cx0, cy0, S0, 0.0]
        for i in range(N):
            best_params.append(pts[i][0])
            best_params.append(pts[i][1])
        for i in range(N):
            best_params.append(0.0)
        best_S = S0
    
    # Extract final configuration
    cx = best_params[0]
    cy = best_params[1]
    S = best_params[2]
    beta = best_params[3]
    
    inner = []
    for i in range(N):
        x = best_params[4 + 2*i]
        y = best_params[5 + 2*i]
        theta = best_params[4 + 2*N + i]
        inner.append([float(x), float(y), float(math.degrees(theta))])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation

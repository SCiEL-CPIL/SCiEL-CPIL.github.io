import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Constants
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3  # Min distance between centers for non-overlap (aligned)
    
    # Initial positions: Hexagonal cluster
    # 1 center
    # 6 neighbors
    # 4 more (approximate positions for 2nd layer)
    
    pos = np.zeros((N, 2))
    # Center
    pos[0] = [0.0, 0.0]
    
    # 6 neighbors
    for k in range(6):
        angle = k * np.pi / 3
        pos[k+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
        
    # 4 more: place them at the "corners" of the 2nd layer
    # 2nd layer is at distance 2*sqrt3
    # Angles: 30, 90, 150, 210 degrees (offset by 30 from the neighbor axes)
    # Or just place them further out and let optimizer pull them in
    for k in range(4):
        angle = (k * 90 + 30) * np.pi / 180
        pos[k+7] = [2.5 * sqrt3 * np.cos(angle), 2.5 * sqrt3 * np.sin(angle)]
        
    # Outer hexagon parameters
    S = 5.0  # Initial guess
    theta_out = 0.0
    
    x0 = np.concatenate([pos.flatten(), [S, theta_out]])
    
    def objective(x):
        if time.time() > deadline:
            return 1e9
        pos_flat = x[:N*2]
        p = pos_flat.reshape((N, 2))
        S_val = x[N*2]
        theta_val = x[N*2+1]
        
        # Calculate max required S for current positions and theta_out
        # To make this differentiable, we can just return S_val and use constraints
        # But SLSQP minimizes objective. We want to minimize S.
        # So objective is S.
        return S_val

    def constraints(x):
        if time.time() > deadline:
            return []
        pos_flat = x[:N*2]
        p = pos_flat.reshape((N, 2))
        S_val = x[N*2]
        theta_val = x[N*2+1]
        
        cons = []
        
        # 1. Pairwise distance constraints
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                # d >= min_dist
                cons.append({'type': 'ineq', 'fun': lambda d=d, min_dist=min_dist: d - min_dist})
                
        # 2. Containment constraints
        # Outer hexagon vertices
        # Vertices of regular hexagon with side S, center (0,0), rotation theta
        # Vertex k: S * (cos(theta + k*pi/3), sin(theta + k*pi/3))
        # The hexagon is convex. A point (px, py) is inside if it satisfies 6 inequalities.
        # Inequality for side k:
        # Normal vector n_k = (cos(theta + k*pi/3 + pi/6), sin(theta + k*pi/3 + pi/6)) ?
        # Let's use the support function.
        # Max dot product of point with normal <= S * sqrt(3)/2 ?
        # For a regular hexagon of side S, the distance from center to side is S*sqrt(3)/2.
        # The normals to the sides are at angles theta + k*pi/3 + pi/6?
        # Let's define the 6 normals.
        # If vertex is at angle alpha, the side opposite is perpendicular to radius at alpha+pi.
        # Side normal angle: theta + k*pi/3.
        # Let's verify.
        # If theta=0, vertices at 0, 60, 120, 180, 240, 300.
        # Side between 0 and 60 deg vertices.
        # Midpoint angle 30 deg.
        # Normal angle 30 deg.
        # So normals are at theta + 30, theta + 90, ...
        # Distance to side = S * sqrt(3) / 2.
        
        half_width = S_val * sqrt3 / 2.0
        
        for k in range(6):
            normal_angle = theta_val + np.pi/6 + k * np.pi / 3
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            
            # All vertices of all inner hexagons must satisfy:
            # dot(v, n) <= half_width
            # Inner hexagon vertices (aligned, angle 0)
            # Relative vertices:
            # (0, 1), (sqrt3/2, 0.5), (sqrt3/2, -0.5), (0, -1), (-sqrt3/2, -0.5), (-sqrt3/2, 0.5)
            # We can just check the 6 vertices of each inner hex.
            # To reduce constraints, note that the max dot product for a convex polygon is at a vertex.
            # So we check all 6 vertices of each of the 11 hexagons.
            # 11 * 6 = 66 constraints per normal direction.
            # 6 directions -> 396 constraints.
            # This is a lot for SLSQP.
            
            # Optimization: Precompute inner vertices relative to center.
            # inner_verts = np.array([...])
            # For each inner hex i, verts = p[i] + inner_verts
            # dot(verts, n) <= half_width
            # This is equivalent to:
            # dot(p[i], n) + max(dot(inner_verts, n)) <= half_width
            # max(dot(inner_verts, n)) is the support function of the unit hexagon in direction n.
            # For a unit hexagon, support function h(n) is:
            # If n is aligned with vertex (angle 0, 60, ...), h=1.
            # If n is aligned with side normal (30, 90, ...), h=sqrt(3)/2.
            # Generally, h(n) = max(dot(v, n)) for v in vertices.
            
            # Let's compute h(n) for the 6 normals.
            # Since normals are fixed relative to theta, we can compute h for each.
            
            # But theta is a variable. So we must compute inside the loop.
            
            # To speed up, we can approximate the inner hexagon as a circle of radius 1?
            # No, that's too loose.
            # Let's just compute the 6 vertices for each inner hex.
            
            # Vectorized check for all inner hexes for this normal
            # inner_verts (6x2)
            # p (11x2)
            # verts (11x6x2)
            # dots (11x6)
            # max_dots (11)
            # p_dot_n (11)
            # constraint: p_dot_n + max_dots <= half_width
            
            # We'll add the constraint for the worst-case inner hex?
            # No, each inner hex must be inside.
            # So we need 11 constraints per normal.
            
            # Let's use a simpler constraint:
            # Distance from center of inner hex to side + inradius <= half_width?
            # No, that assumes the hexagon is centered on the side normal.
            # If rotated, the extent is larger.
            
            # Let's stick to the vertex check but vectorize it.
            
            # Precompute unit hex vertices
            # (0,1), (sqrt3/2, 0.5), ...
            
            # For performance, I will implement the constraint function carefully.
            
            pass # Placeholder

    # Due to complexity of 400+ constraints, SLSQP might be slow.
    # Alternative: Minimize S such that max_i (dist(p_i, boundary)) <= 0?
    # Or minimize S + penalty(violations).
    
    # Let's use a penalty method or just check validity at the end.
    # Better: Use a coarser constraint.
    # Constraint: Distance from center of inner hex to each side >= inradius?
    # This is valid if the inner hex is aligned with the container.
    # If we fix inner angle to 0, and container angle is variable, this approximation is valid ONLY if the container is also aligned?
    # No, if container rotates, the "inradius" direction changes.
    # However, the support function of a hexagon in any direction is between sqrt(3)/2 and 1.
    # If we use radius=1 (circumradius), we guarantee containment but it's too conservative (S will be ~1.15x larger).
    # If we use radius=sqrt(3)/2 (inradius), we might have violations if the hexagon is not aligned with the side.
    
    # Given the time, I will use the Circumradius approximation (R=1) for the inner hexagons.
    # This means we treat inner hexagons as circles of radius 1.
    # This ensures they fit, but might result in a slightly larger S than optimal.
    # However, it reduces constraints to:
    # dot(p_i, n_k) + 1 <= half_width for all i, k.
    # 11 * 6 = 66 constraints. Much better.
    
    def constraints_penalty(x):
        pos_flat = x[:N*2]
        p = pos_flat.reshape((N, 2))
        S_val = x[N*2]
        theta_val = x[N*2+1]
        
        cons = []
        
        # Pairwise distance
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                cons.append({'type': 'ineq', 'fun': lambda d=d: d - min_dist})
                
        # Containment (Circle approximation R=1)
        half_width = S_val * sqrt3 / 2.0
        
        for k in range(6):
            normal_angle = theta_val + np.pi/6 + k * np.pi / 3
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            
            # dot(p, n) + 1 <= half_width
            # dot(p, n) <= half_width - 1
            p_dots = p[:, 0] * nx + p[:, 1] * ny
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda pd=p_dots[i], hw=half_width: hw - 1 - pd})
                
        return cons

    # Objective
    def objective(x):
        if time.time() > deadline:
            return 1e9
        return x[N*2]

    # Bounds
    # S > 0
    # theta in [0, pi/3]
    # p in [-10, 10]
    bounds = [(-10, 10)] * (N*2) + [(0.1, 20), (0, np.pi/3)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_penalty, options={'maxiter': 200, 'ftol': 1e-6})
        
        if res.success:
            best_S = res.x[N*2]
            best_p = res.x[:N*2].reshape((N, 2))
            best_theta = res.x[N*2+1]
            
            # Verify and Polish
            # The circle approximation is conservative.
            # We can try to reduce S slightly by checking actual hexagon vertices.
            
            # Refinement: Local search on S with exact vertex check
            # If S is reduced, check if any vertex is outside.
            
            def is_valid(S_test, p_test, theta_test):
                # Check pairwise
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(p_test[i] - p_test[j]) < min_dist - 1e-6:
                            return False
                # Check containment
                half_width = S_test * sqrt3 / 2.0
                for k in range(6):
                    normal_angle = theta_test + np.pi/6 + k * np.pi / 3
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    
                    # Inner hex vertices (angle 0)
                    # (0,1), (sqrt3/2, 0.5), (sqrt3/2, -0.5), (0, -1), (-sqrt3/2, -0.5), (-sqrt3/2, 0.5)
                    # We can just compute max dot product for each inner hex
                    for i in range(N):
                        cx, cy = p_test[i]
                        # 6 vertices
                        # v0: (0, 1) -> dot = cy
                        # v1: (sqrt3/2, 0.5) -> dot = cx*sqrt3/2 + cy*0.5
                        # v2: (sqrt3/2, -0.5) -> dot = cx*sqrt3/2 - cy*0.5
                        # v3: (0, -1) -> dot = -cy
                        # v4: (-sqrt3/2, -0.5) -> dot = -cx*sqrt3/2 - cy*0.5
                        # v5: (-sqrt3/2, 0.5) -> dot = -cx*sqrt3/2 + cy*0.5
                        
                        max_dot = max(
                            cy,
                            cx*sqrt3/2 + cy*0.5,
                            cx*sqrt3/2 - cy*0.5,
                            -cy,
                            -cx*sqrt3/2 - cy*0.5,
                            -cx*sqrt3/2 + cy*0.5
                        )
                        
                        if max_dot > half_width + 1e-6:
                            return False
                return True

            # Binary search / Step down S
            current_S = best_S
            step = 0.01
            for _ in range(100):
                if time.time() > deadline:
                    break
                if is_valid(current_S - step, best_p, best_theta):
                    current_S -= step
                else:
                    step /= 2
                    if step < 1e-5:
                        break
                        
            return_val = current_S
            inner_list = [[float(p_test[0]), float(p_test[1]), 0.0] for p_test in best_p]
            return inner_list, (0.0, 0.0), float(return_val), float(best_theta)
        else:
            # Return initial guess if optimization fails
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    except Exception as e:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END

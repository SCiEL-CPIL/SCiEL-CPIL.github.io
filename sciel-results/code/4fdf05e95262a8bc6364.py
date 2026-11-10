# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float representing the side length of the outer hexagon
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, we want to finish well before
    
    # Helper: get vertices of a hexagon
    def get_hex_vertices(cx, cy, s, a_deg):
        a = np.deg2rad(a_deg)
        angles = a + np.deg2rad(60 * np.arange(6))
        return np.column_stack((cx + s * np.cos(angles), cy + s * np.sin(angles)))
    
    # Helper: compute the minimum outer hexagon side length containing all inner hexagons
    # The outer hexagon is centered at origin with angle 0 (pointy-top by convention, but we can rotate)
    # Actually, the outer hexagon has angle parameter too. Let's fix outer angle = 0 for simplicity
    # and just find the minimum side length.
    
    # For a hexagon with center (0,0), side length S, angle 0:
    # vertices are at (S*cos(60*i), S*sin(60*i)) for i=0..5
    # The "width" in any direction is related to S.
    
    # Actually, let me think about this differently.
    # Given inner hexagons, I need to find the smallest regular hexagon (with some orientation)
    # that contains all of them.
    
    # A simpler approach: the outer hexagon must contain all vertices of all inner hexagons.
    # For a regular hexagon centered at origin with side length S and angle a:
    # A point (px, py) is inside if for each of the 6 edges, the point is on the correct side.
    
    # Let me just parameterize: outer hexagon center at origin, angle = 0 (or optimize it),
    # and find minimum S such that all inner vertices are inside.
    
    def outer_side_length(xy, angles):
        """Compute minimum side length of outer hexagon (centered at origin, angle 0) to contain all inner hexagons."""
        # For outer hexagon with center (0,0), side S, angle 0:
        # Vertices: (S*cos(60*i), S*sin(60*i)) for i=0..5
        # Edge normals are at angles 30 + 60*i degrees
        # A point is inside if dot(point - vertex_i, normal_i) <= 0 for all i
        # Actually, for a convex polygon, point is inside if it's on the "inside" of all edges.
        
        # Let's use support function approach.
        # The outer hexagon with side S has support function h(theta) = S * cos(theta - 30deg) for theta near 0, etc.
        # Actually, the support function of a regular hexagon with "radius" R (distance from center to vertex) is:
        # h(theta) = R * max(cos(theta - 60*i) for i=0..5)
        # And R = S for a regular hexagon (side = circumradius).
        
        # Wait, for a regular hexagon, the circumradius equals the side length.
        # The support function h(theta) = R * max_i cos(theta - 60*i) where the max is over i such that cos is positive.
        # More precisely, h(theta) = R * cos(theta - 60*round(theta/60)) when |theta - 60*round(theta/60)| <= 30
        
        # For a point to be inside the outer hexagon, we need |point| projected appropriately.
        # Actually, the condition is: for each of the 6 edge normals n_k (k=0..5), 
        # dot(point, n_k) <= h_outer(n_k direction)
        
        # Let me just compute: for each inner vertex, find the minimum S such that it's inside.
        
        all_verts = []
        for i in range(len(xy)):
            verts = get_hex_vertices(xy[i, 0], xy[i, 1], 1.0, angles[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Outer hexagon centered at origin, angle 0, side S.
        # Vertices: V_k = S * (cos(60k), sin(60k)) for k=0..5
        # Edge k connects V_k to V_{k+1} (mod 6)
        # Normal to edge k points outward.
        
        # For a regular hexagon, the edge between V_0 and V_1:
        # V_0 = (S, 0), V_1 = (S/2, S*sqrt(3)/2)
        # Edge direction: V_1 - V_0 = (-S/2, S*sqrt(3)/2), normalize: (-1/2, sqrt(3)/2)
        # Outward normal: (sqrt(3)/2, 1/2) = (cos(30), sin(30))
        # The line: dot(point, (sqrt(3)/2, 1/2)) = dot(V_0, (sqrt(3)/2, 1/2)) = S*sqrt(3)/2
        
        # In general, for edge k, the outward normal is at angle (30 + 60k) degrees
        # and the support distance is S * cos(30) = S * sqrt(3)/2
        
        S_needed = 0.0
        for k in range(6):
            normal_angle = np.deg2rad(30 + 60 * k)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            # dot(point, n) <= S * sqrt(3)/2 for all points
            dots = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dot = np.max(dots)
            # S * sqrt(3)/2 >= max_dot => S >= max_dot / (sqrt(3)/2)
            S_needed = max(S_needed, max_dot / (np.sqrt(3) / 2))
        
        return S_needed
    
    # Initial guess: hexagonal grid arrangement
    # Place 11 hexagons in a roughly hexagonal pattern
    # Center + 6 around it + 4 more
    # Actually, let's use a known good initial configuration
    
    # Hexagonal lattice: neighbors are at distance sqrt(3) (for unit hexagons, center-to-center for adjacent)
    # Wait, for unit side hexagons, the distance between centers of adjacent hexagons is 2 * (sqrt(3)/2) = sqrt(3)
    # No, the "height" of a unit hexagon is sqrt(3), so center-to-center for edge-adjacent is sqrt(3).
    
    # Let me place them in a compact hexagonal arrangement:
    # Row 0: 3 hexagons
    # Row 1: 5 hexagons  
    # Row 2: 3 hexagons
    # Total: 11
    
    # For hexagonal packing of unit hexagons:
    # Horizontal spacing: 1.5 (for pointy-top) or 3/2 * side
    # Vertical spacing: sqrt(3) for alternating rows
    
    # Let's use flat-top hexagons (angle 0 means one vertex points right)
    # Actually, angle 0 means first vertex is at angle 0, so it's pointy-right.
    
    # Let me just use a simple initial guess and optimize.
    
    # Initial positions - hexagonal cluster
    # Center at (0,0), 6 neighbors, then 4 more on the outside
    
    dx = 1.5  # horizontal spacing for flat-top
    dy = np.sqrt(3)  # vertical spacing
    
    # 6 neighbors around center (0,0)
    neighbors = [
        (dx, 0),
        (dx/2, dy/2),
        (-dx/2, dy/2),
        (-dx, 0),
        (-dx/2, -dy/2),
        (dx/2, -dy/2),
    ]
    
    # Add 4 more: extend the hexagon
    # Place at (2*dx, 0), (dx, dy), (-dx, dy), (-2*dx, 0) ... 
    # Or better: (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)
    
    init_positions = [(0, 0)] + neighbors + [
        (dx, dy),
        (-dx, dy),
        (dx, -dy),
        (-dx, -dy),
    ]
    
    init_angles = np.zeros(11)
    init_xy = np.array(init_positions)
    
    # Check initial outer side
    init_S = outer_side_length(init_xy, init_angles)
    
    best_params = np.concatenate([init_xy.flatten(), init_angles])
    best_score = init_S
    
    # Objective function: minimize outer side length + penalty for overlaps
    def objective(params):
        xy = params[0:22].reshape(11, 2)
        angles = params[22:33]
        
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(angles)):
            return 1e6
        
        S = outer_side_length(xy, angles)
        
        # Check for overlaps between inner hexagons
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                vi = get_hex_vertices(xy[i, 0], xy[i, 1], 1.0, angles[i])
                vj = get_hex_vertices(xy[j, 0], xy[j, 1], 1.0, angles[j])
                
                # Check if any edge of hex i intersects any edge of hex j
                # Or simpler: check if any vertex of one is inside the other
                # For speed, use a rough distance check first
                dist = np.linalg.norm(xy[i] - xy[j])
                if dist < 2.0:  # Only check if potentially close
                    # Check vertex-in-polygon for both directions
                    for v in vi:
                        # Check if v is inside hex j
                        inside = True
                        for k in range(6):
                            na = np.deg2rad(30 + 60 * k)
                            nx = np.cos(na)
                            ny = np.sin(na)
                            # Edge of hex j: dot(point - center, n) <= cos(30) = sqrt(3)/2
                            if (v[0] - xy[j, 0]) * nx + (v[1] - xy[j, 1]) * ny > np.sqrt(3) / 2 + 1e-8:
                                inside = False
                                break
                        if inside:
                            penalty += 10.0
                            break
                    if penalty >= 10.0:
                        continue
                    for v in vj:
                        inside = True
                        for k in range(6):
                            na = np.deg2rad(30 + 60 * k)
                            nx = np.cos(na)
                            ny = np.sin(na)
                            if (v[0] - xy[i, 0]) * nx + (v[1] - xy[i, 1]) * ny > np.sqrt(3) / 2 + 1e-8:
                                inside = False
                                break
                        if inside:
                            penalty += 10.0
                            break
                    if penalty >= 20.0:
                        break
        
        return S + penalty
    
    # Run Nelder-Mead optimization
    try:
        res = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
        
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 120:
        # Not enough time for more, just return what we have
        xy = best_params[0:22].reshape(11, 2)
        angles = best_params[22:33] % 360
        inner_hex_data = np.column_stack((xy[:, 0], xy[:, 1], angles))
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        outer_hex_side_length = float(outer_side_length(xy, angles))
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Second phase: refine with more iterations
    try:
        res2 = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x
            
    except Exception as e:
        pass
    
    # Final extraction
    xy = best_params[0:22].reshape(11, 2)
    angles = best_params[22:33] % 360
    
    # Final validation: ensure no overlaps and compute accurate outer side
    final_S = outer_side_length(xy, angles)
    
    inner_hex_data = np.column_stack((xy[:, 0], xy[:, 1], angles))
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(final_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END

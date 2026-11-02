import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Initial configuration: Hexagonal lattice
    # We use a 3x4 block of hexagons, centered.
    # Hex coords (q, r) for a compact shape.
    # Let's try a 3x4 grid:
    # q in {-1, 0, 1, 2}, r in {0, 1, 2, 3} ? No, that's 12.
    # Let's use a standard hex grid generation.
    # A 3x4 rectangle in hex grid is 12 cells.
    # q: 0, 1, 2, 3
    # r: 0, 1, 2
    # Wait, 4*3 = 12.
    # Let's use q in [0, 3] and r in [0, 2].
    
    hex_coords = []
    for q in range(4):
        for r in range(3):
            hex_coords.append((q, r))
            
    # Convert to Cartesian
    # x = 1.5 * q + sqrt(3)/2 * r
    # y = sqrt(3) * (r + q/2)
    sqrt3 = np.sqrt(3.0)
    init_pos = np.zeros((N, 2))
    init_angles = np.zeros(N)
    for i, (q, r) in enumerate(hex_coords):
        init_pos[i, 0] = 1.5 * q + sqrt3 / 2.0 * r
        init_pos[i, 1] = sqrt3 * (r + q / 2.0)
        
    # Center the initial configuration
    init_pos -= init_pos.mean(axis=0)
    
    def get_hex_vertices(cx, cy, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack([x, y])

    def check_overlap(pos, angles):
        # Check if any two hexagons overlap
        # Approximation: Distance between centers must be >= sqrt(3) for non-rotated?
        # No, for arbitrary rotations, we need to check vertex-edge or vertex-vertex distances.
        # A simpler, robust check: 
        # For two unit hexagons, the minimum distance between any point in hex A and any point in hex B is 0 if they touch/overlap.
        # We can check if the distance between centers is less than the sum of in-radii? No, that's for circles.
        # For hexagons, if center distance < sqrt(3) - epsilon, they might overlap if aligned, but if rotated, the "radius" varies.
        # The maximum distance from center to vertex is 1.
        # If center distance < 2.0, they *might* overlap.
        # If center distance < sqrt(3) ~ 1.732, they definitely overlap (since min width is sqrt(3)).
        # Let's use a pairwise distance check with a safe threshold.
        # If dist < 1.732, definitely overlap.
        # If 1.732 < dist < 2.0, check vertices.
        
        for i in range(N):
            vi = get_hex_vertices(pos[i, 0], pos[i, 1], angles[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(pos[j, 0], pos[j, 1], angles[j])
                # Quick reject
                d_centers = np.linalg.norm(pos[i] - pos[j])
                if d_centers > 2.0:
                    continue
                if d_centers < 1.732:
                    return False
                
                # Check if any vertex of i is inside j or vice versa?
                # Or check min distance between vertices.
                # For efficiency, we can check if any vertex of i is within distance < 1.0 (inradius+margin?) of center of j? No.
                # Let's just check pairwise vertex distances.
                # If min dist between any v_i and v_j is < 1e-6, they touch/overlap.
                # Actually, if they are disjoint, min dist > 0.
                # To be safe and fast, we can check if the convex hulls intersect? Too slow.
                # Let's use the fact that if dist < sqrt(3), they overlap.
                # If sqrt(3) <= dist < 2, we need to be careful.
                # Let's compute min distance between vertices.
                # 36 pairs of vertices.
                # This is acceptable for N=12 (66 pairs * 36 checks = ~2400 ops per eval).
                
                # Optimization: Only check if d_centers < 2.0
                # Compute min dist
                min_d = np.inf
                for vi_pt in vi:
                    d = np.linalg.norm(vj - vi_pt, axis=1)
                    min_d = min(min_d, np.min(d))
                    if min_d < 1e-6:
                        return False
                if min_d < 1e-6:
                    return False
        return True

    def calculate_outer_side(pos, angles):
        # Find the minimum side length of a regular hexagon that encloses all points.
        # The outer hexagon can be rotated.
        # The width of the union in direction theta is W(theta).
        # The side length S of a regular hexagon with "width" W (distance between parallel sides) is W / sqrt(3).
        # Wait, width of regular hex side s is sqrt(3)*s. So s = W / sqrt(3).
        # We need to find the rotation of the outer hex that minimizes max(W(theta + alpha)).
        # The function W(theta) is periodic with period pi/3.
        # We can sample theta.
        
        # Get all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            v = get_hex_vertices(pos[i, 0], pos[i, 1], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Sample outer hexagon orientations
        # The critical directions are likely aligned with the vertices or edges of the inner hexagons.
        # Let's sample 360 degrees with fine steps.
        best_s = np.inf
        # We can optimize the outer angle as well.
        # For a fixed set of points, the minimum enclosing hexagon side is min_alpha (max_theta (proj_theta(points) - min_theta(points)) / sqrt(3) * 2? No.
        # Let's stick to the definition:
        # A regular hexagon centered at C with side S and rotation Phi.
        # Vertices are C + S * (cos(Phi + k*pi/3), sin(Phi + k*pi/3)).
        # The constraint is that all inner points P must satisfy:
        # For each of the 6 normals n_k(Phi), dot(P - C, n_k) <= S * cos(pi/6) = S * sqrt(3)/2.
        # So S >= max_k (max_i dot(P_i - C, n_k(Phi))) / (sqrt(3)/2).
        # We want to minimize S over C and Phi.
        # For a fixed Phi, the optimal C is the center of the bounding box defined by the projections?
        # Actually, for a fixed set of half-planes, the minimal enclosing regular hexagon can be found.
        # However, since N is small, we can just optimize Phi and C.
        # C should be the centroid of the points? Not necessarily, but a good start.
        # Let's fix C to be the centroid of all vertices for simplicity in the inner loop, 
        # and optimize Phi. Then refine C?
        # Actually, the center of the minimal enclosing hexagon is the intersection of the "tightest" constraints.
        
        # Simplified approach:
        # 1. Calculate centroid of all vertices.
        # 2. For a range of outer angles Phi, calculate the required S if centered at centroid.
        # 3. Take the min S.
        # This is an approximation. To be more accurate, we can optimize C as well.
        
        centroid = all_verts.mean(axis=0)
        
        # Try a few outer angles. The optimal angle is often related to the symmetry of the packing.
        # Let's sample 0 to 30 degrees (periodicity pi/3 = 60 deg, but symmetry of hex is 60 deg).
        # Actually, the function S(Phi) is periodic with 60 degrees.
        # Let's sample 0, 5, 10, ..., 55.
        min_s = np.inf
        for phi_deg in range(0, 60, 5):
            phi = np.radians(phi_deg)
            # Normals
            angles_n = phi + np.radians(np.arange(0, 360, 60))
            n = np.column_stack([np.cos(angles_n), np.sin(angles_n)])
            
            # Project all points onto normals
            # p = (all_verts - centroid) @ n.T
            # p shape (M, 6)
            p = (all_verts - centroid) @ n.T
            # For each normal k, max projection is max(p[:, k])
            # The constraint is max(p[:, k]) <= S * sqrt(3)/2
            # Also min(p[:, k]) >= -S * sqrt(3)/2
            # So S * sqrt(3)/2 >= max( max(p[:, k]), -min(p[:, k]) )
            # Let R_k = max( max(p[:, k]), -min(p[:, k]) )
            # S >= max(R_k) / (sqrt(3)/2)
            
            max_pos = np.max(p, axis=0)
            min_neg = -np.min(p, axis=0)
            R = np.maximum(max_pos, min_neg)
            S_req = np.max(R) / (sqrt3 / 2.0)
            
            if S_req < min_s:
                min_s = S_req
                
        return min_s

    def total_cost(pos, angles):
        if not check_overlap(pos, angles):
            return 1e10
        s = calculate_outer_side(pos, angles)
        return s

    # Simulated Annealing
    current_pos = init_pos.copy()
    current_angles = init_angles.copy()
    
    # Initial cost
    current_cost = total_cost(current_pos, current_angles)
    best_pos = current_pos.copy()
    best_angles = current_angles.copy()
    best_cost = current_cost
    
    T = 0.5
    T_min = 1e-6
    alpha = 0.99
    
    # Step size
    step_size = 0.2
    
    while time.time() < deadline and T > T_min:
        # Propose move
        idx = np.random.randint(N)
        move_type = np.random.randint(3)
        
        new_pos = current_pos.copy()
        new_angles = current_angles.copy()
        
        if move_type == 0:
            # Move x
            new_pos[idx, 0] += np.random.normal(0, step_size)
        elif move_type == 1:
            # Move y
            new_pos[idx, 1] += np.random.normal(0, step_size)
        else:
            # Rotate
            new_angles[idx] += np.random.normal(0, step_size * 0.5)
            
        new_cost = total_cost(new_pos, new_angles)
        
        # Acceptance
        if new_cost < current_cost:
            current_pos = new_pos
            current_angles = new_angles
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = current_pos.copy()
                best_angles = current_angles.copy()
        else:
            # Metropolis criterion
            diff = new_cost - current_cost
            if diff < 0 or np.random.rand() < np.exp(-diff / T):
                current_pos = new_pos
                current_angles = new_angles
                current_cost = new_cost
        
        T *= alpha
        # Reduce step size slightly
        step_size *= 0.999
        
    # Final result
    inner = [[best_pos[i, 0], best_pos[i, 1], np.degrees(best_angles[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    # Recalculate side for the best solution
    outer_side = calculate_outer_side(best_pos, best_angles)
    outer_angle_deg = 0.0 # The calculation above assumed we can rotate the outer hex freely, 
                          # but the interface asks for a specific angle.
                          # Since we minimized over phi in calculate_outer_side, we should return the optimal phi.
                          # Let's re-find the optimal phi for the best solution to return the correct angle.
    
    # Re-optimize outer angle for the best solution to get the correct angle_deg
    all_verts = []
    for i in range(N):
        v = get_hex_vertices(best_pos[i, 0], best_pos[i, 1], best_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    centroid = all_verts.mean(axis=0)
    
    min_s = np.inf
    best_phi = 0.0
    for phi_deg in range(0, 60, 1):
        phi = np.radians(phi_deg)
        angles_n = phi + np.radians(np.arange(0, 360, 60))
        n = np.column_stack([np.cos(angles_n), np.sin(angles_n)])
        p = (all_verts - centroid) @ n.T
        max_pos = np.max(p, axis=0)
        min_neg = -np.min(p, axis=0)
        R = np.maximum(max_pos, min_neg)
        S_req = np.max(R) / (sqrt3 / 2.0)
        if S_req < min_s:
            min_s = S_req
            best_phi = phi_deg
            
    outer_side = min_s
    outer_angle_deg = best_phi

    return inner, outer_center, outer_side, outer_angle_deg

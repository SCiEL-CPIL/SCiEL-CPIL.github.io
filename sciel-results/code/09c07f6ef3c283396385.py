import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # Helper: Get vertices of a unit regular hexagon centered at (cx, cy)
    # Orientation: flat-top (vertices at 0, 60, 120, 180, 240, 300 degrees)
    def hex_vertices(cx, cy):
        angles = np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + np.cos(angles),
            cy + np.sin(angles)
        ])

    def compute_objective_and_grad(x_flat):
        """
        x_flat: 2N array of coordinates [x0, y0, x1, y1, ...]
        Returns: (objective_value, gradient)
        
        Objective:
        1. Outer hexagon side length (approximated by max projection on 6 axes)
        2. Collision penalty (sum of squared overlaps for all pairs)
        """
        coords = x_flat.reshape(N, 2)
        
        # 1. Compute Outer Side Length
        # The outer hexagon is centered at the origin (we can shift coords later if needed, 
        # but minimizing side length usually implies centering around centroid or origin).
        # To be safe, we compute the required side length for a hexagon centered at (0,0) 
        # containing all points. If the centroid drifts, the side length increases, 
        # so the optimizer will naturally center the cluster.
        
        sqrt3 = np.sqrt(3.0)
        
        # Support functions for a regular hexagon of side S centered at origin:
        # The condition that point (x,y) is inside hexagon of side S is:
        # |x| <= S
        # |y| <= S * sqrt(3)
        # |x/2 + y*sqrt(3)/2| <= S  (wait, let's use standard axes)
        
        # Standard flat-top hexagon vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), ...
        # The half-widths in the 6 normal directions are:
        # Normal 1 (0 deg): S
        # Normal 2 (60 deg): S * cos(30) = S * sqrt(3)/2 ? No.
        
        # Let's derive the support function h(u) for a regular hexagon of side S.
        # The hexagon is the intersection of 6 strips.
        # The distance from center to side (apothem) is S * sqrt(3) / 2.
        # The normals are at angles 30, 90, 150, 210, 270, 330 degrees.
        # Let's verify.
        # Vertex at (S, 0). Normal to edge connecting (S,0) and (S/2, S*sqrt(3)/2) is at 30 deg?
        # Edge vector: (-S/2, S*sqrt(3)/2). Normal: (S*sqrt(3)/2, S/2). Angle 30 deg.
        # Yes. The 6 normals are at 30 + k*60 degrees.
        
        angles = np.radians(30 + np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)]) # 6x2
        
        # For a point p to be inside hexagon of side S:
        # p . n_k <= S * sqrt(3) / 2 for all k.
        # So S >= max_k ( p . n_k ) / (sqrt(3)/2) = 2/sqrt(3) * max_k ( p . n_k )
        
        # We need to contain ALL vertices of ALL inner hexagons.
        # This is expensive to compute exactly for every gradient step if we use all 72 vertices.
        # However, the max projection of the Minkowski sum (union of convex hulls) is the max of individual projections.
        # The union of hexagons is contained in a hexagon of side S iff every vertex of every hexagon is inside.
        # Actually, it suffices to check the max projection of the set of all vertices.
        
        # Let's compute the max projection of all vertices.
        # There are N hexagons, 6 vertices each. Total 72 points.
        # For performance in L-BFGS, we can approximate or just compute it. 72 points is small.
        
        all_verts = np.zeros((N * 6, 2))
        for i in range(N):
            all_verts[i*6:(i+1)*6] = hex_vertices(coords[i, 0], coords[i, 1])
            
        # Project all vertices onto the 6 normals
        # Projections: (72, 6)
        projs = all_verts @ normals.T
        max_proj = np.max(projs)
        
        # Smooth max approximation for differentiability
        # soft_max(z, beta) = (1/beta) * log(sum(exp(beta * z)))
        beta = 20.0
        max_proj_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * max_proj)))
        
        outer_side = (2.0 / sqrt3) * max_proj_smooth
        
        # Gradient of outer_side w.r.t coords
        # d(outer_side)/d(coords) = (2/sqrt3) * d(max_proj_smooth)/d(coords)
        # d(max_proj_smooth)/d(coords) = (1/beta) * d(log(sum(exp(beta * max_proj))))/d(coords)
        # = (1/beta) * (sum(exp(beta * max_proj) * beta * d(max_proj)/d(coords))) / sum(exp(beta * max_proj))
        # = sum(exp(beta * max_proj) * d(max_proj)/d(coords)) / sum(exp(beta * max_proj))
        
        # max_proj is the max over vertices. The gradient of the smooth max is a weighted average of the gradients of the individual projections.
        # weights = exp(beta * projs) / sum(exp(beta * projs))
        
        # We need d(projs)/d(coords).
        # projs[i, k] = v_i . n_k
        # v_i depends on coords[i//6]
        
        # Let's compute the gradient of the outer side part.
        exp_vals = np.exp(beta * max_proj)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp # (72, 6)
        
        # grad_outer = (2/sqrt3) * sum_i sum_k weights[i,k] * d(projs[i,k])/d(coords)
        # d(projs[i,k])/d(coords) is zero except for the 2 coords of hexagon i.
        # For hexagon i, vertex j (local index), v_j = c_i + r_j
        # d(v_j)/d(c_i) = I
        # So d(projs[i,k])/d(c_i) = n_k
        
        # grad_c_i = (2/sqrt3) * sum_k weights[i*6 + local_j, k] * n_k  ... wait, sum over local_j too.
        # Actually, max_proj = max_{i, j, k} ( (c_i + r_j) . n_k )
        # The smooth max weights are distributed over all (i, j, k) combinations.
        # Let w_{i,j,k} be the weight.
        # grad_c_i = (2/sqrt3) * sum_{j,k} w_{i,j,k} * n_k
        
        grad_outer = np.zeros((N, 2))
        # weights is (72, 6). Reshape to (N, 6, 6) -> (N, 36)
        w_reshaped = weights.reshape(N, 6, 6)
        # Sum over j (local vertex index) and k (normal index)
        # grad_c_i = (2/sqrt3) * sum_j sum_k w_{i,j,k} n_k
        # This is equivalent to (2/sqrt3) * (sum_j w_{i,j,k} for each k) dot n_k
        # sum_j w_{i,j,k} is the sum of weights for hexagon i, normal k.
        sum_weights_per_normal = np.sum(w_reshaped, axis=1) # (N, 6)
        
        # grad_c_i = (2/sqrt3) * sum_k (sum_weights_per_normal[i,k] * n_k)
        # = (2/sqrt3) * (sum_weights_per_normal @ normals.T)
        
        grad_outer = (2.0 / sqrt3) * (sum_weights_per_normal @ normals.T)
        
        # 2. Collision Penalty
        # We need to ensure hexagons are disjoint.
        # Two convex polygons are disjoint if there exists a separating axis.
        # For optimization, we use a penalty based on the "overlap" depth.
        # A common differentiable proxy is to check if the distance between centers is less than the sum of radii,
        # but hexagons are not circles.
        
        # Better: Use the support function.
        # Two convex sets A and B are disjoint iff max_{u} (h_A(u) + h_B(-u)) < 0 ? No.
        # They are disjoint iff there exists a direction u such that max_{a in A} a.u < min_{b in B} b.u.
        # This is equivalent to: max_{u} (h_A(u) - h_B(u)) < 0 ? No.
        
        # Let's use the "minimum distance" approximation.
        # Or, simpler: Penalize if the centers are too close.
        # For regular hexagons of side 1, the minimum distance between centers for disjointness depends on orientation.
        # Since we are fixing orientation to 0 (flat-top) for simplicity in this first pass,
        # the "Minkowski sum" of two hexagons is a dodecagon.
        # The centers must be outside this dodecagon.
        
        # The dodecagon formed by the Minkowski sum of two unit hexagons (flat-top) has vertices at:
        # The support function of the sum is h_sum(u) = h_hex(u) + h_hex(u) = 2 * h_hex(u).
        # The hexagon is defined by |x| <= 1, |y| <= sqrt(3), |x/2 + y*sqrt(3)/2| <= 1 ... etc.
        # Actually, the condition for disjointness of two identical hexagons at c1, c2 is:
        # c1 - c2 is NOT in the Minkowski difference H - H.
        # H - H is a regular dodecagon centered at origin.
        # The support function of H - H in direction u is 2 * h_H(u).
        # The vertices of H - H are the sums of vertices of H.
        
        # Let's compute the exact distance from the vector d = c1 - c2 to the boundary of the dodecagon H-H.
        # If d is inside, penalty is positive.
        
        # The dodecagon H-H is the intersection of strips defined by the normals of H.
        # The normals are the same 6 normals as before.
        # The width of the strip for H is 2 * (sqrt(3)/2) = sqrt(3).
        # So for H-H, the width is 2 * sqrt(3).
        # The condition for a point p to be inside H-H is:
        # |p . n_k| <= sqrt(3) for all k.
        
        # Wait, is H-H just the hexagon scaled by 2?
        # No, H is a hexagon. H-H is a dodecagon.
        # The support function h_{H-H}(u) = h_H(u) + h_H(-u).
        # For a centrally symmetric polygon like H, h_H(u) = h_H(-u).
        # So h_{H-H}(u) = 2 h_H(u).
        # The hexagon H is defined by max_k (x . n_k) <= sqrt(3)/2.
        # So h_H(n_k) = sqrt(3)/2.
        # Therefore, the dodecagon H-H is defined by max_k (x . n_k) <= sqrt(3).
        
        # This means H-H is actually a hexagon with side length 2!
        # Let's verify.
        # Vertices of H: (1,0), (0.5, sqrt(3)/2), ...
        # Sum of two vertices: (2,0), (1.5, sqrt(3)), ...
        # Distance from origin to (2,0) is 2.
        # Distance from origin to (1.5, sqrt(3)) is sqrt(2.25 + 3) = sqrt(5.25) ~ 2.29.
        # The hexagon with side 2 has vertices at (2,0), (1, sqrt(3)), ...
        # (1, sqrt(3)) distance is 2.
        # So H-H is indeed a regular hexagon of side 2.
        
        # So, two unit hexagons are disjoint if and only if the vector between their centers
        # is OUTSIDE the hexagon of side 2 centered at the origin.
        
        # The condition for a point p to be INSIDE the hexagon of side 2 is:
        # max_k (p . n_k) <= sqrt(3)
        
        # We want to penalize if p is inside.
        # Let val_k = p . n_k.
        # If max(val_k) < sqrt(3), it is inside.
        # We want a penalty that is 0 if max(val_k) >= sqrt(3) and > 0 if max(val_k) < sqrt(3).
        
        # Let soft_max_val = smooth_max(val_k).
        # If soft_max_val < sqrt(3), penalty = (sqrt(3) - soft_max_val)^2.
        # Otherwise 0.
        
        penalty = 0.0
        grad_penalty = np.zeros((N, 2))
        
        for i in range(N):
            for j in range(i + 1, N):
                d = coords[i] - coords[j]
                projs = d @ normals.T # (6,)
                max_val = (1.0 / beta) * np.log(np.sum(np.exp(beta * projs)))
                
                if max_val < sqrt3:
                    diff = sqrt3 - max_val
                    penalty += diff * diff
                    
                    # Gradient of penalty w.r.t d
                    # d(penalty)/d(max_val) = -2 * diff
                    # d(max_val)/d(projs) = weights (same as before)
                    # d(projs)/d(d) = normals
                    # So d(penalty)/d(d) = -2 * diff * (weights @ normals.T)
                    
                    exp_vals = np.exp(beta * projs)
                    sum_exp = np.sum(exp_vals)
                    weights = exp_vals / sum_exp
                    grad_d = -2.0 * diff * (weights @ normals.T)
                    
                    grad_penalty[i] += grad_d
                    grad_penalty[j] -= grad_d
        
        total_obj = outer_side + 10.0 * penalty
        
        # Combine gradients
        grad_total = grad_outer + 10.0 * grad_penalty
        
        return total_obj, grad_total.flatten()

    # Initial configurations
    # 1. Hexagonal lattice
    sqrt3 = np.sqrt(3.0)
    coords_init = []
    # Center hexagon
    coords_init.append([0.0, 0.0])
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        coords_init.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)]) # Side 2 hexagon vertices?
        # Distance between centers in lattice is sqrt(3) for side 1?
        # No, for side 1 hexagons to touch, center distance is sqrt(3).
        # So place them at distance sqrt(3) + epsilon.
    # Let's use a proper lattice.
    # Basis vectors for hex lattice:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    init_coords = np.array([
        [0.0, 0.0],
        [sqrt3, 0.0],
        [0.0, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [2*sqrt3, 0.0],
        [1.5*sqrt3, 1.5],
        [0.5*sqrt3, 3.0],
        [-0.5*sqrt3, 3.0],
        [-1.5*sqrt3, 1.5]
    ])
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    init_coords += np.random.uniform(-0.1, 0.1, init_coords.shape)
    
    best_x = init_coords.flatten()
    best_val, best_grad = compute_objective_and_grad(best_x)
    
    # L-BFGS-B Optimization
    try:
        res = minimize(
            compute_objective_and_grad,
            best_x,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
        )
        if res.fun < best_val:
            best_x = res.x
            best_val, _ = compute_objective_and_grad(best_x)
    except Exception:
        pass
    
    # Final Validation and Extraction
    coords_final = best_x.reshape(N, 2)
    
    # Calculate exact side length (non-smooth)
    all_verts = np.zeros((N * 6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = hex_vertices(coords_final[i, 0], coords_final[i, 1])
        
    angles = np.radians(30 + np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    final_side = (2.0 / sqrt3) * max_proj
    
    # Check validity (SAT)
    def check_validity(positions):
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    if not check_validity(coords_final):
        # Fallback to initial valid config if optimization failed validity
        # The initial lattice config should be valid
        if check_validity(init_coords):
            coords_final = init_coords
            # Recalculate side
            all_verts = np.zeros((N * 6, 2))
            for i in range(N):
                all_verts[i*6:(i+1)*6] = hex_vertices(init_coords[i, 0], init_coords[i, 1])
            projs = all_verts @ normals.T
            max_proj = np.max(projs)
            final_side = (2.0 / sqrt3) * max_proj
        else:
            # Last resort: spread out
            coords_final = np.array([[i * 3.0, 0.0] for i in range(N)])
            all_verts = np.zeros((N * 6, 2))
            for i in range(N):
                all_verts[i*6:(i+1)*6] = hex_vertices(coords_final[i, 0], coords_final[i, 1])
            projs = all_verts @ normals.T
            max_proj = np.max(projs)
            final_side = (2.0 / sqrt3) * max_proj

    inner = []
    for i in range(N):
        inner.append([float(coords_final[i, 0]), float(coords_final[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END

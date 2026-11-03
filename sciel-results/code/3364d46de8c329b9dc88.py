import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3)
    
    # Vertices of a unit regular hexagon (side 1, circumradius 1)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([i * np.pi / 3 for i in range(6)])
    unit_verts = np.array([np.cos(angles), np.sin(angles)]).T
    
    # Normals for the 6 edges of a regular hexagon
    # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([np.pi/6 + i * np.pi / 3 for i in range(6)])
    normals = np.array([np.cos(normal_angles), np.sin(normal_angles)]).T
    
    # For a regular hexagon with side 1, the distance from center to edge (apothem) is sqrt(3)/2
    # The support function h(u) = max(v . u) for v in hex is 1/2 for |u|=1? No.
    # For unit hex (circumradius 1), apothem a = sqrt(3)/2.
    # The half-plane constraint for a hexagon centered at origin is n_i . p <= sqrt(3)/2.
    # So the support function in direction n_i is sqrt(3)/2.
    # Wait, let's verify.
    # Vertex at (1, 0). Normal to edge connecting (1,0) and (1/2, sqrt(3)/2) is at 30 deg.
    # n = (sqrt(3)/2, 1/2).
    # n . (1,0) = sqrt(3)/2.
    # n . (1/2, sqrt(3)/2) = 3/4 + 1/4 = 1.
    # Wait. The max dot product is 1?
    # Let's check the edge. The edge is between (1,0) and (0.5, 0.866).
    # The line equation: x*sqrt(3)/2 + y*1/2 = 1?
    # (1,0) -> sqrt(3)/2 != 1.
    # Ah, the circumradius is 1, so vertices are on unit circle.
    # The apothem is cos(30) = sqrt(3)/2.
    # The support function h(u) = max_{v in P} v . u.
    # For a regular hexagon with circumradius R=1, h(u) = R * max_{v} (v/R) . u.
    # The maximum of v . u for v on vertices is 1 if u aligns with a vertex.
    # But u is a normal to an edge.
    # For u = n_edge, the max is attained at the two vertices adjacent to that edge.
    # Let's calculate n_edge . v_vertex.
    # n_edge is at 30 deg. v_vertex is at 0 deg.
    # Dot product = cos(30) = sqrt(3)/2.
    # So h(n_edge) = sqrt(3)/2.
    # This is the apothem.
    
    # So for a hexagon of side s, circumradius is s. Apothem is s * sqrt(3)/2.
    # The constraint for point p to be inside hex centered at c with side s is:
    # n_i . (p - c) <= s * sqrt(3)/2 for all i.
    # Or: n_i . p - n_i . c <= s * sqrt(3)/2.
    
    # Distance between two hexagons (non-overlap):
    # Two convex sets A and B are disjoint if d(A, B) > 0.
    # d(A, B) = min_{a in A, b in B} ||a - b||.
    # This is a non-smooth function.
    # However, we can use a smooth approximation or a differentiable penalty.
    # A common trick for Minkowski sum of convex polygons is to use the support functions.
    # The distance is related to the gap in the Minkowski difference.
    # Alternatively, we can penalize the maximum penetration depth.
    # Penetration depth = max(0, -d(A, B)).
    # For regular hexagons, we can check the 6 directions of normals.
    # If for some normal n, the projection of A extends beyond B in direction n, they might overlap.
    # Actually, if A and B are disjoint, there exists a separating axis. For regular hexagons, the separating axis must be one of the 12 edge normals (or 6 if aligned).
    # If they are aligned (same angle), the separating axis is one of the 6 normals.
    # If they are rotated, it's more complex.
    # Let's assume all inner hexagons have angle 0. This is a strong assumption but often optimal for packing identical regular polygons in a regular polygon container due to symmetry.
    # If all angles are 0, the Minkowski difference is also a regular hexagon (scaled).
    # The condition for non-overlap of two unit hexagons at c1 and c2 is:
    # ||c1 - c2||_hex >= 2 * (apothem) ? No.
    # The Minkowski sum of two unit hexagons is a hexagon with circumradius 2.
    # The condition is that c2 - c1 is NOT inside the Minkowski sum of (hex1) and (-hex2).
    # Since hex1 and hex2 are identical and aligned, -hex2 is just hex2 rotated 180, which is hex2.
    # So the "forbidden zone" for c2 relative to c1 is a hexagon with circumradius 2 (side 2).
    # Wait, Minkowski sum of two unit hexagons (side 1) is a hexagon with side 2.
    # The apothem of this sum hexagon is 2 * sqrt(3)/2 = sqrt(3).
    # So c2 - c1 must be outside the hexagon defined by n_i . (c2 - c1) <= sqrt(3).
    # If n_i . (c2 - c1) > sqrt(3) for some i, they are separated in that direction.
    # If n_i . (c2 - c1) <= sqrt(3) for ALL i, they overlap (or touch).
    # So the "violation" for a pair is:
    # max_i ( n_i . (c2 - c1) - sqrt(3) )
    # If this max is < 0, they are disjoint. If > 0, they overlap.
    # This function is convex and piecewise linear. It is not smooth, but it is Lipschitz.
    # We can smooth it: f(x) = logsumexp(beta * x) / beta.
    
    def smooth_max(x, beta=20.0):
        # x is a vector of values
        # To avoid overflow, subtract max
        x_max = np.max(x)
        if x_max == -np.inf:
            return -np.inf
        return x_max + np.log(np.sum(np.exp(x - x_max))) / beta

    def objective(pos):
        # pos: 2N array of [x1, y1, x2, y2, ...]
        # Extract centers
        centers = pos.reshape(N, 2)
        
        # 1. Containment in outer hexagon
        # Outer hexagon centered at (0,0), side R.
        # We want to minimize R.
        # The required R for a point p to be inside is:
        # R >= max_i ( n_i . p ) / (sqrt(3)/2) = max_i ( n_i . p ) * 2/sqrt(3)
        # For a hexagon with center c and side 1, the vertices are c + v_k.
        # The required R for the whole hexagon is the max over its vertices.
        # R_req = max_{k} ( max_i ( n_i . (c + v_k) ) * 2/sqrt(3) )
        # = max_{k, i} ( n_i . c + n_i . v_k ) * 2/sqrt(3)
        
        # Precompute n_i . v_k
        # v_k are unit vertices
        # dots_nv = normals.T @ unit_verts (6x6)
        dots_nv = normals.T @ unit_verts
        
        max_projection = 0.0
        for i in range(N):
            c = centers[i]
            # n_i . c is a vector of 6 values
            n_dot_c = normals.T @ c
            # For each vertex k, the projection is n_dot_c + dots_nv[:, k]
            # We want the max over all i, k
            # proj = n_dot_c + dots_nv.T (broadcasting: 6x1 + 6x6 -> 6x6? No)
            # dots_nv is 6x6. dots_nv[j, k] = n_j . v_k.
            # We want max_j (n_j . c + n_j . v_k) for each k, then max over k.
            # Or simply max over j, k.
            # The term is n_j . (c + v_k).
            # Let's compute the max projection of the hexagon onto the normals.
            # This is equivalent to the support function of the hexagon in directions n_j.
            # h_j(c) = max_k ( n_j . (c + v_k) ) = n_j . c + max_k ( n_j . v_k )
            # max_k ( n_j . v_k ) is just the support of the unit hex in direction n_j, which is sqrt(3)/2.
            # Wait.
            # h_j(unit_hex) = max_k ( n_j . v_k ).
            # Since n_j is a normal to an edge, the max is attained at the two vertices adjacent to that edge.
            # As calculated before, n_j . v_k = sqrt(3)/2 for those vertices.
            # So h_j(unit_hex) = sqrt(3)/2.
            # Therefore, the projection of the inner hexagon i onto normal j is:
            # n_j . c_i + sqrt(3)/2.
            
            # So the condition for containment in outer hex of side R is:
            # n_j . c_i + sqrt(3)/2 <= R * sqrt(3)/2 for all i, j.
            # R >= (n_j . c_i + sqrt(3)/2) / (sqrt(3)/2) = (n_j . c_i) * 2/sqrt(3) + 1.
            
            # So R_req_i = max_j ( (n_j . c_i) * 2/sqrt(3) + 1 )
            # Total R_req = max_i ( R_req_i )
            
            n_dot_c = normals.T @ c
            r_req = np.max(n_dot_c) * 2.0 / sqrt3 + 1.0
            if r_req > max_projection:
                max_projection = r_req
        
        # 2. Non-overlap penalty
        # For each pair (i, j), i < j:
        # diff = c_j - c_i
        # proj = n_k . diff
        # violation = max_k ( proj - sqrt(3) )
        # If violation > 0, they overlap.
        # We use smooth_max to make it differentiable.
        
        penalty = 0.0
        beta = 10.0 # Smoothing parameter
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[j] - centers[i]
                proj = normals.T @ diff
                # We want max(proj - sqrt(3))
                # If max(proj - sqrt(3)) < 0, no overlap.
                # We can use smooth_max on (proj - sqrt(3)).
                # But smooth_max of negative numbers is close to the max.
                # To penalize only positive violations, we can use:
                # max(0, smooth_max(proj - sqrt(3)))
                # Or just smooth_max(proj - sqrt(3)) and rely on the optimizer to push it negative.
                # If we use smooth_max, it's always positive if any term is positive.
                # If all terms are negative, it's close to the max (least negative).
                # We want the penalty to be 0 when disjoint.
                # Let x = proj - sqrt(3).
                # If max(x) < 0, they are disjoint.
                # smooth_max(x) approximates max(x).
                # So if max(x) < 0, smooth_max(x) < 0.
                # We can take max(0, smooth_max(x)).
                # Or, to be smooth, use a softplus-like function.
                # Let's use: penalty += max(0, smooth_max(proj - sqrt(3)))**2
                # Squaring makes it zero if negative, and smooth if positive.
                
                sm = smooth_max(proj - sqrt3, beta)
                if sm > 0:
                    penalty += sm * sm
        return max_projection + 10.0 * penalty

    # Initial guess: Honeycomb lattice
    # Hexagon side 1. Distance between centers in honeycomb is 2 * apothem = sqrt(3).
    # Wait, in a honeycomb tiling, the centers are at distance sqrt(3) * 2?
    # Let's check.
    # Hexagon side 1. Apothem sqrt(3)/2.
    # Distance between parallel edges is sqrt(3).
    # In a honeycomb, the centers form a triangular lattice.
    # The distance between adjacent centers is 2 * apothem = sqrt(3).
    # Let's verify.
    # Two hexagons sharing an edge. The edge is at distance sqrt(3)/2 from each center.
    # So the distance between centers is sqrt(3)/2 + sqrt(3)/2 = sqrt(3).
    # Yes.
    
    # Generate 12 centers in a honeycomb pattern centered at (0,0).
    # Triangular lattice basis:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5) ? No.
    # Angle 60 deg.
    # v2 = (sqrt(3) * cos(60), sqrt(3) * sin(60)) = (sqrt(3)/2, 3/2).
    
    # Let's generate points in a hexagonal shape of radius 2 (in lattice steps).
    # Center (0,0).
    # Ring 0: (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points. We need 12 total. So Center + 11 from Ring 1/2?
    # Or just 12 points in a compact cluster.
    # Let's take the 12 points with smallest norm in the lattice.
    
    lattice_points = []
    # Generate points in a box
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Convert axial coordinates (i, j) to Cartesian
            # x = sqrt(3) * (i + j/2)
            # y = 1.5 * j
            x = sqrt3 * (i + j / 2.0)
            y = 1.5 * j
            lattice_points.append((x, y))
    
    # Sort by distance from origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    # Take first 12
    init_centers = lattice_points[:N]
    
    # Center them
    cx = np.mean([p[0] for p in init_centers])
    cy = np.mean([p[1] for p in init_centers])
    init_pos = []
    for p in init_centers:
        init_pos.append(p[0] - cx)
        init_pos.append(p[1] - cy)
    init_pos = np.array(init_pos)
    
    best_pos = init_pos
    best_score = objective(best_pos)
    
    # Optimization
    # Use L-BFGS-B
    # Bounds: roughly within the outer hexagon.
    # Outer side R ~ 4-5.
    # x, y in [-5, 5].
    bounds = [(-5, 5)] * (2 * N)
    
    try:
        res = minimize(objective, init_pos, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_pos = res.x
    except Exception as e:
        pass
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        
        # Perturb initial positions
        random_pos = init_pos + np.random.normal(0, 0.5, 2*N)
        
        try:
            res = minimize(objective, random_pos, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_pos = res.x
        except Exception as e:
            pass
            
    # Extract results
    centers = best_pos.reshape(N, 2)
    
    # Calculate final outer side
    # R = max_i ( max_j ( n_j . c_i ) * 2/sqrt(3) + 1 )
    max_r = 0.0
    for i in range(N):
        c = centers[i]
        n_dot_c = normals.T @ c
        r_req = np.max(n_dot_c) * 2.0 / sqrt3 + 1.0
        if r_req > max_r:
            max_r = r_req
    
    # Construct inner hexagons
    # All angles 0
    inner = []
    for i in range(N):
        x, y = centers[i]
        inner.append([x, y, 0.0])
    
    outer_center = [0.0, 0.0]
    outer_side = max_r
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END

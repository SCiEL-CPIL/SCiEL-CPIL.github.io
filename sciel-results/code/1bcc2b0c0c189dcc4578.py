# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Strategy:
    1. Assume the inner hexagons are aligned with the outer hexagon (angle 0).
    2. Assume the inner hexagons are placed on a hexagonal lattice with spacing sqrt(3) (touching).
    3. Search over the set of 11-vertex subsets of a sufficiently large hexagonal lattice.
    4. For each subset, determine the minimal outer hexagon side length S and orientation/center
       that encloses them, ensuring S is minimized.
    5. Refine the best candidate using local optimization of the outer hexagon's position and angle.
    """
    start_time = time.time()
    time_limit = 630.0
    n = 11
    
    # Geometry constants
    # Inner hexagons: side 1, angle 0.
    # Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    inner_angles_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles_rad])
    
    # Lattice basis for hexagonal packing of unit hexagons (side 1)
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Let a = sqrt(3).
    # Basis vectors:
    # u = (a, 0)
    # v = (a/2, a*sqrt(3)/2) = (sqrt(3)/2, 3/2)
    # Note: The angle between u and v is 60 degrees.
    a = np.sqrt(3)
    u = np.array([a, 0.0])
    v = np.array([a/2, 3.0/2]) # a * sqrt(3)/2 = sqrt(3)*sqrt(3)/2 = 3/2
    
    # Generate lattice points in a hexagonal region large enough to contain 11 points
    # We need to search for the "shape" of the 11 points.
    # Max radius in lattice steps: R_max = 3 or 4 is usually enough for 11 points.
    # Let's generate points (i, j) such that the hexagonal distance <= R_max.
    # Hex distance: (|i| + |j| + |i+j|) / 2 <= R_max
    R_max = 4
    lattice_points = []
    for i in range(-R_max, R_max + 1):
        for j in range(-R_max, R_max + 1):
            # Check hex distance
            dist_hex = (abs(i) + abs(j) + abs(i + j)) / 2.0
            if dist_hex <= R_max:
                # Convert to Cartesian
                p = i * u + j * v
                lattice_points.append(p)
    
    lattice_points = np.array(lattice_points)
    num_points = len(lattice_points)
    
    # We need to choose 11 points from this lattice.
    # To make this tractable, we can restrict to "connected" or "compact" shapes.
    # However, the number of combinations C(num_points, 11) can be large.
    # num_points for R_max=4: 1 + 6*(1+2+3+4) = 1 + 6*10 = 61 points.
    # C(61, 11) is huge (~ 2e11).
    
    # Strategy Adjustment:
    # Instead of combinatorial search, we can use a "greedy" or "optimization" approach on the lattice indices.
    # Or, we can observe that the optimal packing of 11 hexagons in a hexagon is likely the "hexagonal number" shape plus some extras, or a "diamond" shape.
    # Known optimal packings for small N in a hexagon often follow the hexagonal lattice.
    
    # Let's try a different approach:
    # The objective is to minimize the side length S of the outer hexagon.
    # If we fix the set of 11 lattice points, the minimal enclosing hexagon (aligned with them) is determined by the max extent in the 3 axial directions.
    # However, the outer hexagon can be rotated.
    
    # Let's define the search space:
    # 1. A set of 11 lattice points.
    # 2. The orientation of the outer hexagon.
    
    # To avoid combinatorial explosion, let's restrict the lattice points to a smaller region, 
    # or use a heuristic to find a good set of points.
    # A common compact shape for 11 is the "center + 6 ring + 4 more".
    # Or simply, let's use a local search on the *positions* of the 11 hexagons, 
    # but constrained to the lattice? No, continuous positions might be better if they allow "sliding" to fit tighter?
    # Actually, for convex containers and convex objects, optimal packing often has objects touching the boundary and each other.
    # If the container is a hexagon and objects are hexagons, and they are aligned, 
    # the optimal configuration is likely a subset of the lattice.
    
    # Let's generate candidate sets of 11 points.
    # We can use a "beam search" or just sample random subsets and keep the best ones.
    
    # Precompute relative positions for quick evaluation
    # For a set of points P, the required side length S for an aligned outer hexagon centered at C is:
    # S = max_{p in P} (max_k dot(p - C, n_k)) / (sqrt(3)/2)
    # where n_k are the normals.
    # To minimize S, we should center the outer hexagon at the "center" of the set P.
    # For a hexagon, the optimal center is the intersection of the midlines of the bounding box in the 3 axial directions.
    
    def evaluate_set(points, outer_angle_deg):
        """
        Given a set of 11 points (lattice coords) and an outer angle,
        find the best center and return the side length S.
        """
        # Rotate points to align with outer hexagon? 
        # No, the points are fixed in the global frame (lattice aligned with 0 deg).
        # The outer hexagon is rotated by outer_angle_deg.
        
        # If outer angle is 0, the normals are at 30, 90, ...
        # If outer angle is theta, the normals are at 30+theta, 90+theta, ...
        
        # The condition for point p to be inside hexagon (center C, side S, angle theta) is:
        # max_k dot(p - C, n_k(theta)) <= S * sqrt(3) / 2
        # To minimize S for a fixed C, S = max_k (max_p dot(p, n_k) - dot(C, n_k)) / (sqrt(3)/2)
        # This is a linear programming problem for C.
        # To minimize S, we can find the Chebyshev center of the convex hull of the points in the metric defined by the hexagon.
        
        # Simpler heuristic: The optimal center C is the centroid of the points?
        # Or the intersection of the midlines of the bounding box?
        # Let's compute the bounding box in the 3 axial directions.
        
        # Axial directions for outer hexagon with angle theta:
        # The "width" of the hexagon in direction n is S * sqrt(3).
        # We need to cover the projection of the points onto the 3 pairs of opposite normals.
        
        # Let's just compute the minimal S by finding the center C that minimizes the max projection.
        # This is equivalent to finding the smallest enclosing hexagon.
        
        # For a fixed orientation, the problem is:
        # Minimize S
        # Subject to:
        # dot(C, n_k) + S*sqrt(3)/2 >= max_p dot(p, n_k)  for k=0,1,2
        # dot(C, -n_k) + S*sqrt(3)/2 >= max_p dot(-p, n_k) => -dot(C, n_k) + S*sqrt(3)/2 >= -min_p dot(p, n_k)
        
        # Let M_k = max_p dot(p, n_k) and m_k = min_p dot(p, n_k).
        # Constraints:
        # dot(C, n_k) >= M_k - S*sqrt(3)/2
        # dot(C, n_k) <= m_k + S*sqrt(3)/2
        
        # This implies:
        # M_k - m_k <= S * sqrt(3)
        # So S >= (M_k - m_k) / sqrt(3) for all k.
        
        # Wait, is this sufficient?
        # If S = max_k (M_k - m_k) / sqrt(3), can we find a C?
        # We need an intersection of intervals for dot(C, n_k).
        # Interval for dot(C, n_k) is [M_k - S*sqrt(3)/2, m_k + S*sqrt(3)/2].
        # Length of interval: (m_k + S*sqrt(3)/2) - (M_k - S*sqrt(3)/2) = S*sqrt(3) - (M_k - m_k).
        # Since S >= (M_k - m_k)/sqrt(3), the length is non-negative.
        # However, we need the intersection of the 3 intervals (for n_0, n_1, n_2) to be non-empty.
        # This is not always guaranteed just by the individual widths.
        
        # Let's use a simple optimization for C given S.
        # Or, just use the centroid as a good approximation for C, and calculate S.
        
        centroid = np.mean(points, axis=0)
        
        # Calculate S for this C
        max_proj = 0
        for k in range(3):
            n_k = np.array([np.cos(np.deg2rad(30 + k*60 + outer_angle_deg)), np.sin(np.deg2rad(30 + k*60 + outer_angle_deg))])
            projs = points @ n_k
            M_k = np.max(projs)
            m_k = np.min(projs)
            # Required S_k
            S_k = (M_k - m_k) / np.sqrt(3)
            max_proj = max(max_proj, S_k)
            
        # Check if centroid is a valid center for S = max_proj
        # We need to verify that there exists a C such that all points are inside.
        # If we use the centroid, the projections might be asymmetric, requiring a larger S.
        # Let's calculate S precisely for C = centroid.
        
        S_required = 0
        for k in range(3):
            n_k = np.array([np.cos(np.deg2rad(30 + k*60 + outer_angle_deg)), np.sin(np.deg2rad(30 + k*60 + outer_angle_deg))])
            projs = (points - centroid) @ n_k
            # Max absolute projection
            max_abs_proj = np.max(np.abs(projs))
            # S * sqrt(3) / 2 >= max_abs_proj
            S_req = max_abs_proj * 2 / np.sqrt(3)
            S_required = max(S_required, S_req)
            
        return S_required, centroid

    # Search for the best subset of 11 points and best angle.
    # Since C(61, 11) is too large, we will:
    # 1. Generate a pool of "good" candidate sets.
    #    - The "hexagonal" cluster of 7 + 4 extensions.
    #    - Random samples.
    # 2. For each candidate, optimize the angle.
    
    # Generate specific candidate sets
    candidates = []
    
    # Candidate 1: Center + 6 neighbors + 4 next-nearest
    # Indices in lattice (i,j)
    # Center: (0,0)
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next-nearest: (2,0), (1,1), (0,2), (-1,2) ... 
    # Let's just take the 11 closest points to the origin in the lattice.
    dists = []
    for idx, p in enumerate(lattice_points):
        # Use hex distance or Euclidean? Hex distance is better for lattice connectivity.
        # We generated based on hex distance, so we can just sort by index if we generated in order?
        # No, we generated in loop order. Let's compute Euclidean distance to origin for sorting.
        dists.append(np.linalg.norm(p))
    dists = np.array(dists)
    sorted_indices = np.argsort(dists)
    
    # Take top 11
    top_11_indices = sorted_indices[:11]
    candidates.append(lattice_points[top_11_indices])
    
    # Take top 12, remove one to create variations?
    # Or take top 15 and sample subsets?
    
    # Let's add some random subsets from the top 30 closest points.
    top_30_indices = sorted_indices[:30]
    np.random.seed(42)
    num_random = 500
    for _ in range(num_random):
        idx_subset = np.random.choice(top_30_indices, size=11, replace=False)
        candidates.append(lattice_points[idx_subset])
        
    # Also add "ring" based candidates
    # Center + 6 + 4 from the next ring
    # Ring 1: 6 points
    # Ring 2: 12 points
    # We need 11. Center(1) + Ring1(6) + 4 from Ring2.
    # Let's identify points in Ring 1 and Ring 2.
    ring1_indices = []
    ring2_indices = []
    for idx, p in enumerate(lattice_points):
        # Hex distance
        # We need to recover i,j from p? 
        # Or just use the generation loop?
        # Let's re-calculate hex distance roughly.
        # Actually, we can just use the sorted indices.
        pass
        
    # Simpler: Just use the top 11 and random samples from top 40.
    top_40_indices = sorted_indices[:40]
    for _ in range(1000):
        idx_subset = np.random.choice(top_40_indices, size=11, replace=False)
        candidates.append(lattice_points[idx_subset])

    best_S = float('inf')
    best_points = None
    best_angle = 0.0
    best_center = None
    
    # Evaluate candidates
    # We can vectorize the angle search a bit, or just loop.
    # Angles to check: 0, 5, 10, ... 60.
    angles = np.arange(0, 60, 5.0)
    
    for points in candidates:
        for ang in angles:
            S, C = evaluate_set(points, ang)
            if S < best_S:
                best_S = S
                best_points = points
                best_angle = ang
                best_center = C
                
    if time.time() - start_time > time_limit - 30:
        return _format_result(best_points, best_center, best_angle, best_S)

    # Refine the best candidate
    # We can try to optimize the angle more finely.
    # And maybe allow the points to move slightly off the lattice? 
    # The problem allows any position. 
    # If we move off the lattice, we might be able to reduce S slightly?
    # Usually, for symmetric shapes in symmetric containers, the lattice is optimal.
    # But let's try a local optimization on the positions of the 11 hexagons 
    # starting from the best lattice configuration.
    
    # Objective: Minimize S(centers, angle, center)
    # Variables: 11*2 (centers) + 2 (outer center) + 1 (angle) = 25 variables.
    
    def get_S_from_params(params):
        centers = params[:22].reshape(11, 2)
        outer_center = params[22:24]
        angle_deg = params[24]
        
        # Calculate S
        # Normals
        base_normals = np.deg2rad([30, 90, 150])
        rotated_normals = base_normals + np.deg2rad(angle_deg)
        normals = np.array([[np.cos(a), np.sin(a)] for a in rotated_normals])
        
        # Vertices of inner hexagons (relative to center)
        # We need to check if all vertices of all inner hexagons are inside the outer hexagon.
        # Or, more efficiently, check the max projection of the *vertices* of the inner hexagons.
        
        # The outer hexagon is defined by: max_k dot(p - C, n_k) <= S * sqrt(3) / 2
        # So S = max_{i, v, k} dot(v_i + c_i - C, n_k) / (sqrt(3)/2)
        
        max_proj = 0
        for i in range(11):
            c_i = centers[i]
            # Vertices of hexagon i
            verts = inner_verts_rel + c_i
            # Project onto normals
            for k in range(3):
                n_k = normals[k]
                projs = (verts - outer_center) @ n_k
                max_proj = max(max_proj, np.max(projs))
                
        S = max_proj * 2 / np.sqrt(3)
        
        # Penalty for overlap
        # We need to ensure inner hexagons don't overlap.
        # Use the Minkowski sum condition.
        # For each pair i, j:
        # max_k dot(c_j - c_i, n_k) >= sqrt(3)
        # Wait, the normals for the Minkowski sum of two aligned hexagons are the same as the outer normals?
        # Yes, if they are aligned.
        # If they are not aligned (we fixed angle 0 for inner, so they are aligned with each other),
        # the separation condition is:
        # max_{m=0..5} dot(c_j - c_i, n_m) >= sqrt(3)
        # where n_m are the 6 normals of the unit hexagon.
        
        penalty = 0
        hex_normals = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in [30, 90, 150, 210, 270, 330]])
        for i in range(11):
            for j in range(i+1, 11):
                diff = centers[j] - centers[i]
                projs = diff @ hex_normals.T
                max_proj_sep = np.max(projs)
                if max_proj_sep < np.sqrt(3):
                    penalty += (np.sqrt(3) - max_proj_sep)**2
                    
        return S + 100 * penalty

    # Initial params
    params0 = np.concatenate([best_points.flatten(), best_center, [best_angle]])
    
    try:
        res = minimize(get_S_from_params, params0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_S:
            # Extract best
            best_params = res.x
            best_S = get_S_from_params(best_params) # Recalculate to be sure
            # Update best_points etc for formatting? 
            # We just need to return the formatted result.
            # Let's store the best params
            final_params = best_params
        else:
            final_params = params0
    except Exception:
        final_params = params0
        
    # Format result
    return _format_from_params(final_params)

def _format_from_params(params):
    n = 11
    centers = params[:2*n].reshape(n, 2)
    outer_center = params[2*n:2*n+2]
    angle_deg = params[2*n+2]
    
    # Recalculate S
    inner_angles_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles_rad])
    
    # Normals for outer hexagon
    base_normals = np.deg2rad([30, 90, 150])
    rotated_normals = base_normals + np.deg2rad(angle_deg)
    normals = np.array([[np.cos(a), np.sin(a)] for a in rotated_normals])
    
    max_proj = 0
    for i in range(n):
        c_i = centers[i]
        verts = inner_verts_rel + c_i
        for k in range(3):
            n_k = normals[k]
            projs = (verts - outer_center) @ n_k
            max_proj = max(max_proj, np.max(projs))
            
    S = max_proj * 2 / np.sqrt(3)
    S += 1e-5 # Safety margin
    
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0 # Inner hexagons are aligned with lattice (angle 0)
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], angle_deg])
    
    return inner_hex_data, outer_hex_data, float(S)

def _format_result(points, center, angle, S):
    # Fallback formatter if optimization fails or times out
    # points: (11, 2)
    # center: (2,)
    # angle: float
    # S: float
    
    n = 11
    centers = points
    
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([center[0], center[1], angle])
    
    return inner_hex_data, outer_hex_data, float(S + 1e-5)

# EVOLVE-BLOCK-END

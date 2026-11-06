"""Improved program using targeted greedy search for max area multiplicity."""
import numpy as np
from collections import Counter

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Initialize with a structured grid.
    2. Identify the dominant area A*.
    3. Iteratively swap points to increase the count of triples with area A*.
    """
    
    def compute_areas_and_max(points):
        """
        Compute all doubled areas for triples and return the max count and the area value.
        Returns: (max_count, best_area, counter_dict)
        """
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        
        # Generate indices for all triples
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        
        i_idx = np.array(i_idx)
        j_idx = np.array(j_idx)
        k_idx = np.array(k_idx)
        
        # Vectorized area calculation
        xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
        xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
        xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Filter positive areas
        positive_mask = areas > 0
        if not np.any(positive_mask):
            return 0, 0, {}
        
        pos_areas = areas[positive_mask]
        counter = Counter(pos_areas.tolist())
        if not counter:
            return 0, 0, {}
        
        best_area, max_count = max(counter.items(), key=lambda item: item[1])
        return max_count, best_area, counter

    def get_point_scores(points, target_area):
        """
        For each point p in points, count how many pairs {u,v} in points\{p}
        form a triple with p that has area == target_area.
        """
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        scores = np.zeros(n, dtype=np.int64)
        
        # We can vectorize this:
        # For each pair (u,v), compute area with all p.
        # But simpler: just loop over points since n=30 is small, but use numpy for inner loop.
        
        for i in range(n):
            # Get other points
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others = pts[mask]
            
            if len(others) < 2:
                continue
                
            # Generate pairs of others
            # Since n=30, others has 29 points. Pairs = 29*28/2 = 406.
            # We can just compute areas for all triples involving i.
            # Actually, let's just compute the area for all triples (i, j, k) where j != i, k != i, j < k.
            # This is effectively what compute_areas_and_max does but filtered by i.
            
            # Efficient way:
            # Construct array of all pairs from 'others'
            # This might be overkill. Let's just use the global triple list logic but mask.
            # Or simpler:
            # For each j in others:
            #   For each k in others (j<k):
            #      if area(i,j,k) == target: scores[i]++
            
            # Vectorized approach for a single i:
            # Create indices for all pairs in 'others'
            # But 'others' is a view/subset.
            
            # Let's stick to a semi-vectorized loop for clarity and speed (30*406 is small)
            # Actually, let's precompute all pair indices for the full set once?
            # No, let's just do a simple loop. 30 points * 406 pairs = 12k ops. Very fast.
            
            o_idx = np.where(mask)[0]
            for u_idx in range(len(o_idx)):
                for v_idx in range(u_idx + 1, len(o_idx)):
                    u = pts[o_idx[u_idx]]
                    v = pts[o_idx[v_idx]]
                    p = pts[i]
                    # Area of triangle (p, u, v)
                    area = np.abs(p[0]*(u[1]-v[1]) + u[0]*(v[1]-p[1]) + v[0]*(p[1]-u[1]))
                    if area == target_area:
                        scores[i] += 1
                        
        return scores

    def find_best_swap(points, target_area, candidate_pool):
        """
        Find the best point to replace (p) and the best replacement (q)
        to maximize the count of target_area.
        """
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        current_scores = get_point_scores(points, target_area)
        
        # Precompute T(S, q, A) for all q in candidate_pool
        # T(S, q, A) = number of pairs {u,v} in S such that Area(q,u,v) == A
        
        # To do this efficiently:
        # Iterate over all pairs {u,v} in S.
        # For each pair, find how many q in pool satisfy Area(q,u,v) == A.
        # Area(q,u,v) = A => q lies on a line.
        # Instead of geometric line intersection, we can just compute area for all q in pool for each pair.
        # Pairs in S: 435.
        # Pool size: ~3700.
        # Total ops: 435 * 3700 = 1.6M. Fast.
        
        pool_arr = np.array(candidate_pool, dtype=np.int64)
        num_pool = len(pool_arr)
        
        if num_pool == 0:
            return -1, -1, 0
            
        t_scores = np.zeros(num_pool, dtype=np.int64)
        
        # Get all pairs in S
        pair_indices = []
        for i in range(n):
            for j in range(i+1, n):
                pair_indices.append((i, j))
                
        # Vectorize pair processing
        # Create arrays for u and v coordinates for all pairs
        u_coords = np.array([pts[i] for i, j in pair_indices])
        v_coords = np.array([pts[j] for i, j in pair_indices])
        
        # For each q in pool, compute areas with all pairs
        # q_coords: (num_pool, 2)
        # u_coords: (num_pairs, 2)
        # v_coords: (num_pairs, 2)
        
        # We want area(q, u, v) for all q, u, v.
        # This is a 3D array (num_pool, num_pairs, 2) effectively.
        # Let's reshape to broadcast.
        
        # q_x, q_y shape (num_pool, 1)
        # u_x, u_y shape (1, num_pairs)
        # v_x, v_y shape (1, num_pairs)
        
        q_x = pool_arr[:, 0:1]
        q_y = pool_arr[:, 1:2]
        
        u_x = u_coords[:, 0:1].T
        u_y = u_coords[:, 1:2].T
        v_x = v_coords[:, 0:1].T
        v_y = v_coords[:, 1:2].T
        
        # Area formula: | qx(uy-vy) + ux(vy-qy) + vx(qy-uy) |
        # = | qx*uy - qx*vy + ux*vy - ux*qy + vx*qy - vx*uy |
        
        terms = (q_x * u_y - q_x * v_y + u_x * v_y - u_x * q_y + v_x * q_y - v_x * u_y)
        areas_matrix = np.abs(terms)
        
        # Count matches for target_area
        matches = (areas_matrix == target_area).astype(np.int64)
        t_scores = np.sum(matches, axis=1)
        
        # Now, for each p in S and q in Pool:
        # Delta = T(S, q) - Interact(q, p) - current_scores[p]
        # We want to maximize Delta.
        
        best_delta = -1
        best_p_idx = -1
        best_q_idx = -1
        
        # Interact(q, p) is the number of pairs {p, u} in S such that Area(q, p, u) == target_area.
        # Note: T(S, q) counts pairs {u,v} in S.
        # Pairs in S \ {p} are all pairs in S EXCEPT those involving p.
        # So C_new(q, S\{p}) = T(S, q) - |{ u in S\{p} : Area(q,p,u) == target }|.
        
        # We can precompute interaction terms?
        # Interact(q, p) depends on q and p.
        # Let's just loop over p (30) and vectorize over q (3700).
        
        for p_idx in range(n):
            p = pts[p_idx]
            
            # Calculate interaction: for each q in pool, count u in S\{p} such that Area(q,p,u)==target
            # u ranges over S\{p}.
            # Let's create an array of all u in S\{p}
            mask_u = np.ones(n, dtype=bool)
            mask_u[p_idx] = False
            u_coords_p = pts[mask_u]
            
            if len(u_coords_p) == 0:
                continue
                
            u_x_p = u_coords_p[:, 0:1].T # (1, 29)
            u_y_p = u_coords_p[:, 1:2].T # (1, 29)
            
            # Area(q, p, u)
            # q_x, q_y: (num_pool, 1)
            # p_x, p_y: scalar
            # u_x_p, u_y_p: (1, 29)
            
            px = p[0]
            py = p[1]
            
            terms_int = (q_x * u_y_p - q_x * (np.array([[py]])) + 
                         u_x_p * (np.array([[py]])) - u_x_p * q_y + 
                         px * q_y - px * u_y_p)
            # Wait, formula: qx(uy-py) + ux(py-qy) + px(qy-uy)
            # = qx*uy - qx*py + ux*py - ux*qy + px*qy - px*uy
            
            areas_int = np.abs(terms_int)
            
            # Count matches
            matches_int = (areas_int == target_area).astype(np.int64)
            interact_counts = np.sum(matches_int, axis=1) # (num_pool,)
            
            # Delta for this p and all q
            # Delta_q = t_scores[q] - interact_counts[q] - current_scores[p_idx]
            deltas = t_scores - interact_counts - current_scores[p_idx]
            
            max_delta_idx = np.argmax(deltas)
            max_delta_val = deltas[max_delta_idx]
            
            if max_delta_val > best_delta:
                best_delta = max_delta_val
                best_p_idx = p_idx
                best_q_idx = max_delta_idx
                
        return best_p_idx, best_q_idx, best_delta

    # Initialization
    # Use a 6x5 grid centered at 0
    best_points = []
    for x in range(-2, 4):
        for y in range(-2, 3):
            best_points.append((x, y))
            
    # Ensure 30 points
    while len(best_points) < 30:
        # Add points on the boundary of the grid
        best_points.append((len(best_points) // 6, len(best_points) % 6))
        
    # Ensure distinct and in bounds
    valid_set = set()
    final_points = []
    for p in best_points:
        if p not in valid_set and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
            valid_set.add(p)
            final_points.append(p)
            
    # If we still don't have 30, fill with random
    if len(final_points) < 30:
        all_pts = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
        for p in all_pts:
            if len(final_points) >= 30:
                break
            if p not in valid_set:
                valid_set.add(p)
                final_points.append(p)
                
    best_points = final_points[:30]
    
    # Initial Evaluation
    max_count, best_area, _ = compute_areas_and_max(best_points)
    
    # Candidate pool: all valid points not in best_points
    all_valid = set([(x, y) for x in range(-30, 31) for y in range(-30, 31)])
    current_set = set(best_points)
    candidate_pool = [p for p in all_valid if p not in current_set]
    
    # Optimization loop
    for _ in range(50): # Max 50 swaps
        if max_count == 0:
            break
            
        p_idx, q_idx, delta = find_best_swap(best_points, best_area, candidate_pool)
        
        if p_idx == -1 or delta <= 0:
            break
            
        # Perform swap
        old_point = best_points[p_idx]
        new_point = candidate_pool[q_idx]
        
        best_points[p_idx] = new_point
        current_set.remove(old_point)
        current_set.add(new_point)
        
        # Update candidate pool
        candidate_pool.remove(new_point)
        candidate_pool.append(old_point)
        
        # Re-evaluate max area
        max_count, best_area, _ = compute_areas_and_max(best_points)
        
    return best_points

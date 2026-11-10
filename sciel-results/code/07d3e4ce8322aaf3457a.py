import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy safe-removal heuristic 
    on the complement set S.
    
    Strategy:
    1. Start with S = F_49^2 (all points in S, N is empty).
    2. The condition for N to be a Nikodym set is equivalent to:
       For every point x in S and every direction v, there exists t != 0 
       such that x + t*v is also in S.
       (i.e., no point in S is "isolated" in any direction within S).
    3. We greedily remove points from S if they are "safe" to remove.
       A point x is safe to remove if, after removal, no other point y in S 
       loses its last neighbor in any direction.
       Specifically, for every direction v and every y on the line through x in 
       direction v (y != x), y must have at least one neighbor in S (other than x) 
       in direction v.
    4. Repeat removals in multiple passes until no more points can be removed.
    5. Return N = F_49^2 \ S.
    """
    p = 7
    q = p * p
    total_points = q * q
    
    # Field arithmetic helpers
    # Element is (a, b) -> a + b*alpha, alpha^2 = 3
    # Index = a * p + b
    
    # Precompute addition and multiplication tables for F_q
    # add[i, j] = i + j
    # mul[i, j] = i * j
    # We can just use index arithmetic if we map carefully, but tables are safer.
    
    def elem_to_idx(a, b):
        return a * p + b
    
    def idx_to_elem(idx):
        return (idx // p, idx % p)
    
    # Build addition table
    add_tbl = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        a1, b1 = idx_to_elem(i)
        for j in range(q):
            a2, b2 = idx_to_elem(j)
            a_sum = (a1 + a2) % p
            b_sum = (b1 + b2) % p
            add_tbl[i, j] = elem_to_idx(a_sum, b_sum)
            
    # Build multiplication table
    mul_tbl = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        a1, b1 = idx_to_elem(i)
        for j in range(q):
            a2, b2 = idx_to_elem(j)
            # (a1 + b1*alpha) * (a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
            # alpha^2 = 3
            a_prod = (a1 * a2 + b1 * b2 * 3) % p
            b_prod = (a1 * b2 + a2 * b1) % p
            mul_tbl[i, j] = elem_to_idx(a_prod, b_prod)
            
    # Directions: v = (1, m) for m in F_q, and v = (0, 1)
    # We represent a direction by its vector (dx, dy) in F_q x F_q.
    # There are q + 1 directions.
    
    # Precompute for each point and each direction, the list of other points on that line.
    # line_points[point_idx, dir_idx] = array of indices of other points on the line.
    # There are q-1 other points on each line.
    
    # To save memory and time, we can compute the line membership on the fly or precompute a boolean matrix.
    # Given q=49, total points=2401, directions=50.
    # line_mask[point_idx, dir_idx] is a boolean vector of length total_points? 
    # 2401 * 50 * 2401 booleans is ~28 MB. Acceptable.
    # But checking "does y have a neighbor in S" requires summing over the line.
    
    # Let's precompute the lines as lists of indices.
    # lines[dir_idx][point_idx] = list of q indices (including point_idx)
    # Or better: for each point and direction, store the set of neighbors.
    
    # Efficient Check:
    # We maintain an array `in_S` of size total_points (boolean).
    # We also maintain `neighbor_count[point_idx, dir_idx]` = number of points in S on the line through point in dir, excluding point itself.
    
    # Initialization
    in_S = np.ones(total_points, dtype=bool)
    
    # Precompute line neighbors
    # For each direction v, and each point x, find the other q-1 points.
    # Direction v = (dx, dy).
    # Points on line: x + t*v for t in F_q.
    
    # List of directions
    directions = []
    # v = (0, 1)
    directions.append((0, 1))
    # v = (1, m) for m in F_q
    for m_idx in range(q):
        # m is an element of F_q
        # v = (1, m)
        directions.append((1, m_idx))
        
    num_dirs = len(directions)
    
    # Precompute neighbors for each point and direction
    # neighbors[point_idx, dir_idx] = array of indices of the q-1 other points
    neighbors = np.zeros((total_points, num_dirs, q-1), dtype=np.int32)
    
    for d_idx, (dx_idx, dy_idx) in enumerate(directions):
        for x_idx in range(total_points):
            # Generate points x + t*v for t in F_q, t != 0
            count = 0
            for t_idx in range(q):
                if t_idx == 0:
                    continue
                # t is an element of F_q
                # point = x + t * v
                # x = (xa, xb), v = (dx, dy)
                # t*v = (t*dx, t*dy)
                tv_x = mul_tbl[t_idx, dx_idx]
                tv_y = mul_tbl[t_idx, dy_idx]
                # x + t*v
                nx = add_tbl[x_idx, tv_x]
                ny = add_tbl[x_idx, tv_y]
                # Wait, x_idx is index of point (xa, xb). 
                # My add_tbl is defined on field elements, not plane points.
                # I need to decompose x_idx into (xa, xb) and (ya, yb)?
                # No, x_idx is just an index 0..2400.
                # Let's map point index to (x_elem, y_elem).
                # point_idx = x_elem * q + y_elem.
                
                # Let's restart the neighbor precomputation with correct indexing.
                pass
                
    # Correct Precomputation
    # Point index P = x_elem * q + y_elem
    # x_elem, y_elem in 0..48
    
    neighbors = np.zeros((total_points, num_dirs, q-1), dtype=np.int32)
    
    for d_idx, (dx_idx, dy_idx) in enumerate(directions):
        for x_elem in range(q):
            for y_elem in range(q):
                point_idx = x_elem * q + y_elem
                count = 0
                for t_idx in range(q):
                    if t_idx == 0:
                        continue
                    # t is field element
                    # v = (dx, dy)
                    # t*v = (t*dx, t*dy)
                    tv_x = mul_tbl[t_idx, dx_idx]
                    tv_y = mul_tbl[t_idx, dy_idx]
                    
                    # new_x = x_elem + tv_x
                    # new_y = y_elem + tv_y
                    nx = add_tbl[x_elem, tv_x]
                    ny = add_tbl[y_elem, tv_y]
                    
                    n_point_idx = nx * q + ny
                    neighbors[point_idx, d_idx, count] = n_point_idx
                    count += 1
                    
    # Initialize neighbor counts
    # neighbor_count[point_idx, dir_idx]
    neighbor_count = np.zeros((total_points, num_dirs), dtype=np.int32)
    for p_idx in range(total_points):
        for d_idx in range(num_dirs):
            # All neighbors are in S initially
            neighbor_count[p_idx, d_idx] = q - 1
            
    # Greedy Removal
    # We try to remove points. 
    # To be safe, we iterate until no changes or time limit.
    
    deadline = time.time() + 500  # 500s limit, leaving margin
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to remove each point
        # Randomize order to avoid bias? Or just iterate.
        # Iterating is deterministic. Let's shuffle indices for variety if we want, but deterministic is fine for now.
        for p_idx in range(total_points):
            if not in_S[p_idx]:
                continue
                
            # Check if p_idx is safe to remove
            # For each direction d:
            #   For each neighbor y in neighbors[p_idx, d]:
            #     if not in_S[y]: continue (y is already removed, doesn't care about p_idx)
            #     if neighbor_count[y, d] == 1 and the only neighbor is p_idx:
            #        Then removing p_idx makes y isolated in direction d. Unsafe.
            #     else: safe for y.
            
            # How to check "only neighbor is p_idx"?
            # neighbor_count[y, d] counts neighbors in S.
            # If neighbor_count[y, d] == 1, we need to know if p_idx is that neighbor.
            # We can check if p_idx is in the list of neighbors of y in direction d.
            # Since lines are symmetric, if p_idx is neighbor of y in dir d, then y is neighbor of p_idx in dir d.
            # So we can just check: is p_idx in neighbors[y, d]?
            
            safe = True
            for d_idx in range(num_dirs):
                # Check all neighbors of p_idx in this direction
                # If any neighbor y (which is in S) has neighbor_count[y, d_idx] == 1, 
                # it means y's only neighbor in S in this direction is p_idx (since they are on the same line).
                # Wait, if y has count 1, and p_idx is its neighbor, then p_idx is the unique neighbor.
                # So we just need to check:
                # For all y in neighbors[p_idx, d_idx]:
                #    if in_S[y] and neighbor_count[y, d_idx] == 1:
                #        safe = False
                #        break
                
                # Optimization: 
                # If neighbor_count[p_idx, d_idx] == 0, p_idx is already isolated? 
                # No, p_idx is in S. If count is 0, the current S is invalid?
                # No, we start valid and only remove safe points, so S remains valid.
                # So neighbor_count[x, d] >= 1 for all x in S.
                
                # We iterate over neighbors of p_idx in dir d_idx
                for k in range(q-1):
                    y_idx = neighbors[p_idx, d_idx, k]
                    if in_S[y_idx]:
                        if neighbor_count[y_idx, d_idx] == 1:
                            # y's only neighbor in this direction is p_idx
                            # (Because y is on the line with p_idx, and count is 1)
                            safe = False
                            break
                if not safe:
                    break
                    
            if safe:
                # Remove p_idx
                in_S[p_idx] = False
                changed = True
                
                # Update neighbor counts for all neighbors of p_idx
                for d_idx in range(num_dirs):
                    for k in range(q-1):
                        y_idx = neighbors[p_idx, d_idx, k]
                        if in_S[y_idx]:
                            neighbor_count[y_idx, d_idx] -= 1
                            
                # Early exit if time is tight
                if time.time() > deadline:
                    break
                    
        if time.time() > deadline:
            break

    # Construct N = complement of S
    # N points are those where in_S is False
    n_indices = np.where(~in_S)[0]
    
    if len(n_indices) == 0:
        # Fallback: return a single point? No, must be valid Nikodym set.
        # If S is full space, N is empty, which is invalid (k>=1).
        # But our greedy removal should remove at least some points.
        # If it doesn't, we force remove one point? 
        # If we force remove, we might break validity.
        # However, starting from full space, is it always possible to remove a point?
        # In F_49^2, yes, likely.
        
        # If no points removed, we have a problem.
        # Let's just return the current N. If empty, it's invalid.
        # But the problem says k>=1.
        # Let's assume the algorithm removes points.
        
        pass

    # Convert indices to coordinates
    # Index P = x_elem * q + y_elem
    # x_elem = a + b*alpha -> (a, b)
    # y_elem = c + d*alpha -> (c, d)
    
    points = []
    for idx in n_indices:
        x_elem = idx // q
        y_elem = idx % q
        
        xa = x_elem // p
        xb = x_elem % p
        ya = y_elem // p
        yb = y_elem % p
        
        points.append([[xa, xb], [ya, yb]])
        
    if not points:
        # Emergency fallback: return a known valid small set?
        # Or just return the full space minus 1 point? 
        # Actually, if no points were removed, the loop didn't run or found no safe points.
        # Let's just return an array of shape (0, 2, 2) which is invalid, 
        # but hopefully the algorithm works.
        # To be safe, let's return a single point if nothing else? No, invalid.
        # Let's return the result of the current S complement.
        pass

    result = np.array(points, dtype=np.int64)
    
    # If result is empty, we must handle it.
    # But given the structure, it should not be empty.
    
    return result

# EVOLVE-BLOCK-END

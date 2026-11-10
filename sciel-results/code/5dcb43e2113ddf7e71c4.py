# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    alpha_sq = 3

    # Precompute field operations
    def elem_add(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def elem_mul(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        ra = (a1 * a2 + b1 * b2 * alpha_sq) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra * p + rb

    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_add(i, j)
            mul_table[i, j] = elem_mul(i, j)

    # Directions: 0..48 slopes, 49 vertical
    # Precompute line_other[idx, d, t]
    # idx = x*q + y
    # d in 0..48: slope d, vertical is d=49
    # t in 1..48 (indices 0..47)
    line_other = np.zeros((q * q, 50, 48), dtype=np.int32)
    
    # Precompute x, y coords for faster access
    xs = np.array([i // q for i in range(q * q)], dtype=np.int32)
    ys = np.array([i % q for i in range(q * q)], dtype=np.int32)
    
    # Precompute field elements as (a, b) pairs for speed?
    # Or just use divmod in the loop, it's only 2401 * 50 * 48 ~ 5.7M ops.
    # Let's try to vectorize the construction of line_other if possible, or just loop.
    # Looping 5.7M times in Python is ~5-10s. Acceptable.
    
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            a_x, b_x = divmod(x, p)
            a_y, b_y = divmod(y, p)
            
            for d in range(49):
                for t_idx, t in enumerate(range(1, q)):
                    a_t, b_t = divmod(t, p)
                    # ox = x + t
                    ox_a = (a_x + a_t) % p
                    ox_b = (b_x + b_t) % p
                    ox = ox_a * p + ox_b
                    
                    # oy = y + t*d
                    td = mul_table[t, d]
                    a_td, b_td = divmod(td, p)
                    oy_a = (a_y + a_td) % p
                    oy_b = (b_y + b_td) % p
                    oy = oy_a * p + oy_b
                    
                    line_other[idx, d, t_idx] = ox * q + oy
            
            # Vertical line: x + t, y
            for t_idx, t in enumerate(range(1, q)):
                a_t, b_t = divmod(t, p)
                oy_a = (a_y + a_t) % p
                oy_b = (b_y + b_t) % p
                oy = oy_a * p + oy_b
                line_other[idx, 49, t_idx] = x * q + oy

    # Construct S as the FULL Unital
    # S = { (x,y) : 2*x_a + y_a^2 - 3*y_b^2 == 0 }
    # This set has size 49.
    # N = complement of S.
    
    S = np.zeros(q * q, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            x = xa * p + xb
            for ya in range(p):
                for yb in range(p):
                    y = ya * p + yb
                    if (2 * xa + ya * ya - yb * yb * alpha_sq) % p == 0:
                        S[x * q + y] = True

    N = ~S

    # Validity check and repair
    # We want to minimize |N|, i.e., maximize |S|, subject to:
    # For every point x, there exists a line L through x such that L \ {x} subset N.
    # Equivalently: For every point x, there exists a line L through x such that L \ {x} is disjoint from S.
    
    # Let's define "bad" points in S.
    # A point s in S is "bad" if for every line L through s, L \ {s} intersects S.
    # If we remove a point s from S (add to N), it might fix other points.
    
    # Greedy strategy:
    # 1. Start with the Unital S.
    # 2. Check validity.
    # 3. If invalid, find points that violate the condition.
    #    A point x violates if for all lines L through x, (L \ {x}) intersects S.
    #    This is equivalent to: x has no "clean" line.
    #    Note: if x in S, x is a "center" of S.
    #    If x not in S, x is in N.
    
    # Let's refine the repair loop to be more efficient.
    # Instead of recomputing everything, we can maintain a count of "clean lines" for each point.
    # clean_count[x] = number of lines L through x such that (L \ {x}) is disjoint from S.
    
    # Initialize clean_count
    clean_count = np.zeros(q * q, dtype=np.int32)
    
    # For each point x, for each line L through x, check if (L \ {x}) is disjoint from S.
    # This is O(N * D * (D-1)) = 2401 * 50 * 48 ~ 5.7M.
    # We can vectorize this check.
    
    # line_other is (2401, 50, 48).
    # S is (2401,).
    # S[line_other] gives (2401, 50, 48) boolean array.
    # np.all(~S[line_other], axis=2) gives (2401, 50) boolean array: clean_lines[x, d]
    # clean_count = np.sum(clean_lines, axis=1)
    
    def compute_clean_counts(N, S):
        # N is not needed directly if we use S
        # Disjoint from S means all others are NOT in S.
        others_in_S = S[line_other]  # (2401, 50, 48)
        clean_lines = ~np.any(others_in_S, axis=2)  # (2401, 50)
        return np.sum(clean_lines, axis=1)

    clean_count = compute_clean_counts(N, S)
    
    # Repair loop
    # If clean_count[x] == 0, x is unsatisfied.
    # We need to remove some points from S to make clean_count[x] > 0.
    # Removing a point s from S (adding to N) can only increase clean_count for other points.
    # Specifically, if s is on a line L through x, and L was not clean because of s,
    # then removing s makes L clean (if no other S points on L).
    
    # Heuristic:
    # While there are unsatisfied points:
    #   Find an unsatisfied point x.
    #   We need to remove a point s from S such that s lies on a line through x.
    #   Ideally, s is a point that "blocks" many lines.
    #   Let's just pick the first unsatisfied point x.
    #   Look at all lines L through x.
    #   For each L, find S points on L.
    #   Pick one S point s on L to remove.
    #   Which one? Maybe the one that appears in the most "blocking" lines?
    #   Simple: pick the first one found.
    
    # To be faster, we can batch remove.
    # But let's try simple iteration first.
    
    deadline = time.time() + 590
    max_iter = 1000
    it = 0
    
    while it < max_iter:
        if time.time() > deadline:
            break
        
        unsatisfied = np.where(clean_count == 0)[0]
        if len(unsatisfied) == 0:
            break
        
        x = unsatisfied[0]
        
        # Find a point s in S to remove
        # Look at lines through x
        found = False
        for d in range(50):
            others = line_other[x, d, :]
            # Find indices in others that are in S
            s_indices = np.where(S[others])[0]
            if len(s_indices) > 0:
                # Pick the first one
                s_idx = others[s_indices[0]]
                S[s_idx] = False
                N[s_idx] = True
                found = True
                # Update clean_count
                # Removing s_idx from S affects clean_count of points y such that s_idx is on a line through y.
                # i.e., y is on a line through s_idx.
                # Which points are on lines through s_idx?
                # All points y such that there exists a direction d' and t' such that y = s_idx + t' * v_{d'}.
                # This is just all points except s_idx itself? No.
                # The line through s_idx with direction d' contains s_idx and 48 others.
                # So s_idx is on 50 lines.
                # The points affected are s_idx and the 48*50 = 2400 other points?
                # Actually, removing s from S makes lines through s potentially cleaner.
                # Specifically, for any y, if s is on a line L through y, then L might become clean.
                # So we need to update clean_count for all y such that s is on a line through y.
                # This is equivalent to: y is on a line through s.
                # The set of points on lines through s is the whole space?
                # No. A line through s is defined by a direction.
                # The union of all lines through s is the whole space F_49^2.
                # Wait. For any y != s, there is a unique line through s and y.
                # So s lies on a line through y for ALL y != s.
                # And s lies on 50 lines through itself.
                # So removing s from S potentially affects clean_count for ALL points.
                # Thus, we must recompute clean_count from scratch.
                # This is O(5.7M) per iteration.
                # If we have ~50 iterations, that's ~300M ops.
                # In NumPy, 5.7M ops is fast (~10ms). 50 * 10ms = 500ms.
                # So full recompute is fine.
                
                clean_count = compute_clean_counts(N, S)
                break
        if not found:
            # Should not happen if S is not empty and x is unsatisfied.
            # If x is unsatisfied, it means for all d, there is an S point on line(x,d).
            # So we should always find one.
            break
        
        it += 1

    points = []
    for x in range(q):
        for y in range(q):
            if N[x * q + y]:
                xa, xb = divmod(x, p)
                ya, yb = divmod(y, p)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END

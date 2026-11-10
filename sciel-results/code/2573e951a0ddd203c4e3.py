# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy:
    1. Start with the known valid complement S_0 (points in the Hermitian unital 
       excluding those with y-coordinates in a specific set Y_0).
    2. Use precomputed line structures to efficiently check if adding a point 
       to S (removing from N) preserves the Nikodym property.
    3. Greedily add points to S to maximize its size (minimize N).
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin before 630s kill

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, non-residue mod 7

    # Precompute field operations tables
    # Index 0..48 represents a + b*alpha
    def idx(a, b):
        return a + b * p

    def a_of(i):
        return i % p

    def b_of(i):
        return i // p

    # Multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            res_a = (ai * aj + bi * bj * k) % p
            res_b = (ai * bj + bi * aj) % p
            mul_table[i, j] = idx(res_a, res_b)

    # Addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            res_a = (ai + aj) % p
            res_b = (bi + bj) % p
            add_table[i, j] = idx(res_a, res_b)

    total_points = q * q  # 2401
    num_dirs = 50  # 49 directions (1, m) + 1 direction (0, 1)

    # Precompute line_other[d, x, :] = the 48 other points on the line through x in direction d
    # d in 0..48: direction (1, m) with m = d
    # d = 49: direction (0, 1)
    line_other = np.zeros((num_dirs, total_points, q - 1), dtype=np.int32)
    
    for d in range(num_dirs):
        for x_idx in range(total_points):
            x = x_idx // q
            y = x_idx % q
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    # Point: (x + t, y + t*m)
                    nx = add_table[x, t]
                    ny = add_table[y, mul_table[t, m]]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1
            else:
                # Direction (0, 1)
                for t in range(1, q):
                    # Point: (x, y + t)
                    nx = x
                    ny = add_table[y, t]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1

    # Precompute affected[s, d, :] = points x such that s is on line_d(x) and s != x
    # This is the set of points x for which adding s to S might invalidate direction d.
    affected = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for s_idx in range(total_points):
        sx = s_idx // q
        sy = s_idx % q
        for d in range(num_dirs):
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    # x = s - t*(1, m) = (sx - t, sy - t*m)
                    # -t in F_49 is (q-t) % q? No, field negation.
                    # Negation of t (index t) is (q-t) % q ONLY if we were in Z_q.
                    # In F_49, negation of a+b*alpha is (-a) + (-b)*alpha.
                    # Index of -t: if t = a+b*alpha, -t = (7-a)%(7) + (7-b)%(7)*alpha.
                    # Let's precompute neg_table.
                    pass # Will fill below after neg_table

    # Precompute negation table
    neg_table = np.zeros(q, dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        neg_a = (-ai) % p
        neg_b = (-bi) % p
        neg_table[i] = idx(neg_a, neg_b)

    # Fill affected table
    for s_idx in range(total_points):
        sx = s_idx // q
        sy = s_idx % q
        for d in range(num_dirs):
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    # x = s + (-t)*(1, m)
                    neg_t = neg_table[t]
                    nx = add_table[sx, neg_t]
                    ny = add_table[sy, mul_table[neg_t, m]]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1
            else:
                for t in range(1, q):
                    neg_t = neg_table[t]
                    nx = sx
                    ny = add_table[sy, neg_t]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1

    # Initialize S with the base valid complement
    # Base construction: Points in Hermitian Unital U = {(x,y) : Tr(x) + Norm(y) = 0}
    # Tr(x) = 2*xa, Norm(y) = ya^2 - k*yb^2
    # Excluded from S (i.e., kept in N) if y is in Y_base.
    # Y_base = {y in F_49 : y_a is a quadratic residue mod 7}
    # Quadratic residues mod 7: 0, 1, 2, 4
    y_residues = {0, 1, 2, 4}
    
    S = set()
    for xa in range(p):
        for xb in range(p):
            x_idx = idx(xa, xb)
            for ya in range(p):
                for yb in range(p):
                    y_idx = idx(ya, yb)
                    # Check if (x, y) is in Unital
                    tr_x = 2 * xa
                    norm_y = (ya * ya - k * yb * yb) % p
                    if (tr_x + norm_y) % p == 0:
                        # In unital. Check if y is in Y_base (to be kept in N)
                        if ya in y_residues:
                            pass # Keep in N, do not add to S
                        else:
                            S.add(x_idx * q + y_idx)

    # N_mask: True if point is in N
    N_mask = np.ones(total_points, dtype=bool)
    for s in S:
        N_mask[s] = False

    # Calculate valid directions for each point
    # valid_count[x] = number of directions d such that all 48 other points on line_d(x) are in N
    # valid_dir_mask[x, d] = True if direction d is valid for x
    valid_count = np.zeros(total_points, dtype=np.int32)
    valid_dir_mask = np.zeros((total_points, num_dirs), dtype=bool)

    for x_idx in range(total_points):
        for d in range(num_dirs):
            others = line_other[d, x_idx, :]
            if np.all(N_mask[others]):
                valid_count[x_idx] += 1
                valid_dir_mask[x_idx, d] = True

    # Verify initial validity
    if not np.all(valid_count > 0):
        # Fallback: return just the unital complement if initial is broken (shouldn't happen)
        # But we assume the base construction is valid.
        pass

    # Greedy expansion
    # Candidates: all points not in S
    candidates = [i for i in range(total_points) if i not in S]
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    # To speed up, we can process in batches or just iterate.
    # Since 2401 is small, we can afford multiple passes if needed, but one pass is usually enough for greedy.
    # We will do a single pass, trying to add each candidate.
    
    for s in candidates:
        if time.time() - start_time > deadline - 10:
            break
            
        if s in S:
            continue
            
        # Tentatively add s to S
        # This removes s from N.
        # This affects all points x such that s is on line_d(x) for some d.
        # For each such (x, d), if valid_dir_mask[x, d] was True, it becomes False.
        # We need to check if any x has valid_count[x] drop to 0.
        
        # Collect affected (x, d) pairs
        # affected[s, d, :] gives x's.
        # We need to count how many valid directions are lost for each x.
        
        # Optimization: 
        # lost_count[x] = number of d such that s is on line_d(x) AND valid_dir_mask[x, d] is True.
        # If valid_count[x] - lost_count[x] <= 0, then x is broken.
        
        # We can iterate over d and x in affected[s, d, :].
        # Since affected[s, d, :] has 48 entries, and there are 50 directions, total ~2400 checks.
        # This is fast.
        
        lost_counts = np.zeros(total_points, dtype=np.int32)
        
        for d in range(num_dirs):
            xs = affected[s, d, :]
            for x in xs:
                if x == -1: # Should not happen with our fill logic, but safe
                    continue
                if valid_dir_mask[x, d]:
                    lost_counts[x] += 1
        
        # Check if any point is broken
        # We only need to check points where lost_counts[x] > 0
        affected_points = np.nonzero(lost_counts)[0]
        valid_addition = True
        for x in affected_points:
            if valid_count[x] - lost_counts[x] <= 0:
                valid_addition = False
                break
        
        if valid_addition:
            # Commit the change
            S.add(s)
            N_mask[s] = False
            for x in affected_points:
                # We need to update valid_dir_mask and valid_count for these x's
                # Actually, we need to iterate over d again to update the mask
                # But we only invalidated directions d where s is on the line.
                # Let's just re-calculate valid_dir_mask for these specific x's?
                # No, we can just decrement.
                # We know which (x,d) pairs were lost.
                # Let's loop d and x again to update the mask, it's cheap.
                pass
            
            # Update masks and counts
            for d in range(num_dirs):
                xs = affected[s, d, :]
                for x in xs:
                    if x == -1:
                        continue
                    if valid_dir_mask[x, d]:
                        valid_dir_mask[x, d] = False
                        valid_count[x] -= 1

    # Build result
    points = []
    for i in range(total_points):
        if i not in S:
            x = i // q
            y = i % q
            xa, xb = a_of(x), b_of(x)
            ya, yb = a_of(y), b_of(y)
            points.append([[xa, xb], [ya, yb]])

    result = np.array(points, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END

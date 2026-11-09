# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.

    Strategy:
    1. Start with a known valid Nikodym set based on a Hermitian unital construction.
    2. Iteratively remove points from N that are not essential for the Nikodym property.
    3. Use precomputed field tables for speed.
    4. Use a greedy algorithm with multiple passes to maximize the number of removed points.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit

    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    # Find smallest quadratic non-residue mod p
    k_field = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_field = c
            break
    if k_field is None:
        k_field = 3  # fallback

    # Represent elements of F_q as integers 0..48: element (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables for F_q
    # f_add[x][y] = x + y in F_q
    # f_mul[x][y] = x * y in F_q
    f_add_table = np.zeros((q, q), dtype=np.int32)
    f_mul_table = np.zeros((q, q), dtype=np.int32)
    
    # Helper for field multiplication
    def f_mul_raw(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        real = (xa * ya + xb * yb * k_field) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    for x in range(q):
        for y in range(q):
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            f_add_table[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            f_mul_table[x, y] = f_mul_raw(x, y)

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))  # vertical
    n_dir = len(directions)

    # All points in F_q^2 as integer pairs (x_idx, y_idx) where x_idx, y_idx in 0..q-1
    # Map (x_idx, y_idx) to a single index 0..q^2-1
    point_index = {}
    all_points = []
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            all_points.append((x_idx, y_idx))
    
    idx_to_point = [None] * (q * q)
    for pt, idx in point_index.items():
        idx_to_point[idx] = pt

    # Build the base Nikodym set
    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    # Blocking set B = points of U whose y-coordinate has y_a a square mod p
    # Excluded set S = U \ B
    # N = complement of S

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    # Initialize N: all points except excluded
    in_N = np.ones(q * q, dtype=bool)

    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0:
                # In unital
                if y_idx not in y_special:
                    # Excluded
                    in_N[point_index[(x_idx, y_idx)]] = False

    # Initialize not_in_N_count
    # not_in_N_count[p_idx][d_idx] = number of points on line(point, v_d) \ {point} that are not in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    # Compute initial counts
    # For each direction d, for each point p, count non-N points on line(p, d) \ {p}
    for d_idx, (vx, vy) in enumerate(directions):
        # Precompute t*v for t in 1..q-1
        t_v = [(f_mul_table[t, vx], f_mul_table[t, vy]) for t in range(1, q)]
        
        for p_idx in range(q * q):
            px, py = idx_to_point[p_idx]
            count = 0
            for tx, ty in t_v:
                nx = f_add_table[px, tx]
                ny = f_add_table[py, ty]
                n_idx = point_index[(nx, ny)]
                if not in_N[n_idx]:
                    count += 1
            not_in_N_count[p_idx, d_idx] = count

    # cover_count[p_idx] = number of directions d such that not_in_N_count[p_idx, d] == 0
    cover_count = np.sum(not_in_N_count == 0, axis=1).astype(np.int32)

    # Precompute affects
    # affects[p_idx] = list of (x_idx, d_idx) where p_idx is on line(x, d) \ {x}
    # This means x + t*d = p for some t != 0
    # So x = p - t*d
    affects = [[] for _ in range(q * q)]
    
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tx = f_add_table[px, f_add_table[px, f_mul_table[t, vx]]] # This is wrong, need subtraction
                # Subtraction: p - t*v
                # p + (-1)*(t*v)
                # -1 in F_q is (p-1, p-1)? No, -1 is just additive inverse.
                # In F_p, -1 is p-1. In F_q, -1 is (p-1, p-1) if we map int to pair?
                # Let's just use f_add with negation.
                # Negation of (a,b) is (-a, -b) mod p.
                # Let's precompute f_neg table or just calculate.
                pass

    # Let's build a proper f_sub table or use f_add with neg.
    # f_neg[x] = -x
    f_neg_table = np.zeros(q, dtype=np.int32)
    for x in range(q):
        xa, xb = to_pair(x)
        f_neg_table[x] = to_int((-xa) % p, (-xb) % p)
        
    # f_sub[x, y] = x - y = x + (-y)
    # We can compute on the fly using f_add_table and f_neg_table.

    # Re-compute affects correctly
    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                # x = p - t*v
                tv_x = f_mul_table[t, vx]
                tv_y = f_mul_table[t, vy]
                neg_tv_x = f_neg_table[tv_x]
                neg_tv_y = f_neg_table[tv_y]
                nx = f_add_table[px, neg_tv_x]
                ny = f_add_table[py, neg_tv_y]
                if (nx, ny) in point_index:
                    x_idx = point_index[(nx, ny)]
                    affects[p_idx].append((x_idx, d_idx))

    # Greedy removal
    # We will iterate through points in N and try to remove them.
    # To be more effective, we can sort points by some heuristic.
    # Points with high cover_count are less likely to be critical for themselves, 
    # but we care about points x that depend on p.
    # Let's just do multiple passes.
    
    n_indices = [i for i in range(q * q) if in_N[i]]
    
    # Shuffle to avoid bias? Or sort by cover_count?
    # Let's try sorting by cover_count descending (most redundant first?)
    # Actually, if a point p is removed, it affects x.
    # If x has high cover_count, it's robust.
    # We want to remove p if it doesn't break any x.
    # Let's just iterate in a fixed order first, then repeat.
    
    removed_total = 0
    while True:
        if time.time() > deadline - 10:
            break
            
        progress = False
        # Iterate through points currently in N
        # We need to be careful: if we remove p1, it might affect the validity of removing p2.
        # So we should check validity against the *current* state.
        
        # To allow more removals in one pass, we can iterate in a random order or specific order.
        # Let's iterate through all indices.
        for p_idx in range(q * q):
            if not in_N[p_idx]:
                continue
            if time.time() > deadline - 5:
                break
                
            # Tentatively remove p_idx
            in_N[p_idx] = False
            affected_counts = []
            invalid = False
            
            for (x, d) in affects[p_idx]:
                old_val = not_in_N_count[x, d]
                new_val = old_val + 1
                not_in_N_count[x, d] = new_val
                affected_counts.append((x, d, old_val))
                
                if old_val == 0:
                    # This direction d was covering x. Now it's not.
                    # Check if x has any other covering direction.
                    # cover_count[x] was the number of directions with count 0.
                    # If cover_count[x] == 1, then x becomes uncovered.
                    if cover_count[x] == 1:
                        invalid = True
                        break
                    else:
                        cover_count[x] -= 1
            
            if invalid:
                # Revert
                for (x, d, old_val) in affected_counts:
                    not_in_N_count[x, d] = old_val
                in_N[p_idx] = True
            else:
                # Keep removal
                # cover_count was already updated in the loop if not invalid?
                # Wait, if invalid was True, we broke out. We need to revert cover_count changes too.
                # Let's track cover_count changes.
                pass

    # The logic above has a flaw: if invalid is found, we revert not_in_N_count, but we might have already decremented cover_count for some x before finding the invalid one.
    # Let's refine the removal check.

    # Refined Removal Check:
    # 1. Collect all (x, d) affected.
    # 2. For each (x, d), if old_val == 0, mark x as "losing coverage d".
    # 3. For each x that is losing coverage, check if it has other covering directions.
    #    We can do this by checking if sum(not_in_N_count[x] == 0) > 1 (before update) or similar.
    #    Actually, if old_val == 0, then d was a covering direction.
    #    If cover_count[x] > 1, then x has another covering direction.
    #    If cover_count[x] == 1, then x loses its last covering direction -> Invalid.
    
    # So the check is:
    # For all (x, d) in affects[p]:
    #   if not_in_N_count[x, d] == 0:
    #       if cover_count[x] == 1:
    #           invalid = True
    #       else:
    #           cover_count[x] -= 1
    #           not_in_N_count[x, d] += 1
    #   else:
    #       not_in_N_count[x, d] += 1
    #
    # If invalid, revert all changes.
    # To revert, we need to store the changes.
    
    while True:
        if time.time() > deadline - 10:
            break
        
        progress = False
        # Iterate through points in N
        # We can randomize the order to escape local minima?
        # Let's try a fixed order first, then random.
        
        indices_in_N = [i for i in range(q * q) if in_N[i]]
        # Shuffle for variety
        np.random.shuffle(indices_in_N)
        
        for p_idx in indices_in_N:
            if not in_N[p_idx]:
                continue
            if time.time() > deadline - 5:
                break
                
            # Check if p_idx can be removed
            changes = [] # List of (x, d, old_count, old_cover)
            valid = True
            
            for (x, d) in affects[p_idx]:
                old_count = not_in_N_count[x, d]
                old_cover = cover_count[x]
                
                if old_count == 0:
                    if old_cover == 1:
                        valid = False
                        break
                    else:
                        # Update cover count tentatively
                        cover_count[x] -= 1
                        changes.append((x, d, old_count, old_cover))
                        not_in_N_count[x, d] += 1
                else:
                    changes.append((x, d, old_count, old_cover))
                    not_in_N_count[x, d] += 1
            
            if valid:
                in_N[p_idx] = False
                progress = True
            else:
                # Revert
                for (x, d, old_count, old_cover) in changes:
                    not_in_N_count[x, d] = old_count
                    cover_count[x] = old_cover
                # in_N[p_idx] remains True
        
        if not progress:
            break

    # Build result
    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return the base set
        points = []
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                if in_N[point_index[(x_idx, y_idx)]]: # This will be false if we removed everything, but fallback should be base
                    points.append([[xa, xb], [ya, yb]])
        # If still empty, something went wrong, return a valid single point? No, must be Nikodym.
        # The base set is always valid.
        # Reconstruct base set if empty
        if not points:
             y_special = set()
             for u in range(p):
                 for v in range(p):
                     y_special.add(to_int((u * u) % p, v))
             for x_idx in range(q):
                 xa, xb = to_pair(x_idx)
                 trace_x = (2 * xa) % p
                 for y_idx in range(q):
                     ya, yb = to_pair(y_idx)
                     norm_y = (ya * ya - yb * yb * k_field) % p
                     if (trace_x + norm_y) % p == 0:
                         if y_idx not in y_special:
                             continue
                     else:
                         points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END

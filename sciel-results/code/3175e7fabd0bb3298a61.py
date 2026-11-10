import time
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set N in F_q^2 with q = p^2, p = 7.
    Maximize |N^c| = |S|.
    """
    p = 7
    q = p * p
    NPTS = q * q  # 2401

    start_time = time.time()
    deadline = start_time + 580.0

    # --- Field F_p[alpha], alpha^2 = 3 ---
    k_const = 3

    # Precompute field tables
    # Element represented as index 0..48
    # idx = a + b*7 where a,b in 0..6
    # val = a + b*alpha
    # Let's just use the integer 0..48 as the field element representation.
    # We need add and mul tables.
    
    def idx_to_ab(idx):
        return idx // 7, idx % 7

    def ab_to_idx(a, b):
        return a * 7 + b

    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        ai, bi = idx_to_ab(i)
        for j in range(q):
            aj, bj = idx_to_ab(j)
            # Add
            res_a = (ai + aj) % p
            res_b = (bi + bj) % p
            add_table[i, j] = ab_to_idx(res_a, res_b)
            
            # Mul: (ai + bi*alpha) * (aj + bj*alpha)
            # = ai*aj + ai*bj*alpha + bi*aj*alpha + bi*bj*alpha^2
            # = ai*aj + bi*bj*3 + (ai*bj + bi*aj)*alpha
            res_a = (ai * aj + bi * bj * k_const) % p
            res_b = (ai * bj + bi * aj) % p
            mul_table[i, j] = ab_to_idx(res_a, res_b)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        # Point is (x, y) in F_q^2
        # x = xa + xb*alpha -> index in F_q is xa*7+xb?
        # Wait, the task says: "pair (a, b) ... stands for a + b*alpha"
        # "point ... [[x_a, x_b], [y_a, y_b]]"
        # So x = x_a + x_b*alpha.
        # My field element index `i` corresponds to `a + b*7`?
        # In my table, `i` is `a*7+b`?
        # Let's check `idx_to_ab`: `a = i//7`, `b = i%7`.
        # So `i` represents `a + b*alpha`?
        # No, usually `i` is the integer value.
        # Let's align:
        # Field element `f` is represented by integer `i` in 0..48.
        # `i` corresponds to `a + b*alpha` where `a = i//7` and `b = i%7`?
        # If so, `i = a*7 + b`.
        # Then `pt_to_idx` should map `[[xa, xb], [ya, yb]]` to a global index.
        # Global index for point `(x, y)` where `x, y \in F_q` (represented as 0..48).
        # Global index = `x * q + y`?
        # `x` is `xa*7+xb`. `y` is `ya*7+yb`.
        # Global index = `(xa*7+xb)*49 + (ya*7+yb)`.
        
        x_idx = xa * 7 + xb
        y_idx = ya * 7 + yb
        return x_idx * q + y_idx

    def idx_to_pt(idx):
        y_idx = idx % q
        x_idx = idx // q
        xa = x_idx // 7
        xb = x_idx % 7
        ya = y_idx // 7
        yb = y_idx % 7
        return [[xa, xb], [ya, yb]]

    # --- Directions ---
    # (1, m) for m in F_q, and (0, 1)
    # v = (vx, vy)
    # m is field element 0..48
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    num_dirs = len(directions)

    # --- Precompute affected ---
    # affected[p_idx, d_idx] is array of q-1 indices
    affected = np.zeros((NPTS, num_dirs, q - 1), dtype=np.int32)
    
    for p_idx in range(NPTS):
        # Get x, y field indices
        y_idx = p_idx % q
        x_idx = p_idx // q
        
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for t in range(1, q): # t != 0
                # t is field element index 0..48? No, t is 1..48?
                # t iterates over F_q \ {0}.
                # Field elements are 0..48. 0 is index 0.
                # So t ranges from 1 to 48.
                tvx = mul_table[t, vx]
                tvy = mul_table[t, vy]
                nx_idx = add_table[x_idx, tvx]
                ny_idx = add_table[y_idx, tvy]
                n_idx = nx_idx * q + ny_idx
                affected[p_idx, d_idx, count] = n_idx
                count += 1

    # --- State ---
    line_count = np.zeros((NPTS, num_dirs), dtype=np.int32)
    zero_counts = np.zeros(NPTS, dtype=np.int32)
    in_S = np.zeros(NPTS, dtype=bool)

    def recompute(S_list):
        line_count[:] = 0
        in_S[:] = False
        for s_idx in S_list:
            in_S[s_idx] = True
            for d_idx in range(num_dirs):
                line_count[affected[s_idx, d_idx], d_idx] += 1
        zero_counts[:] = np.sum(line_count == 0, axis=1)

    def is_valid():
        return bool(np.all(zero_counts > 0))

    def try_add(p_idx):
        # Returns list of (n_idx, d_idx) where line_count changes from 0 to 1
        # and None if it breaks validity
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                if line_count[n_idx, d_idx] == 0:
                    if zero_counts[n_idx] == 1:
                        return None
                    changes.append((n_idx, d_idx))
        return changes

    def apply_add(p_idx, changes):
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1
        in_S[p_idx] = True

    def try_remove(q_idx):
        # Returns list of (n_idx, d_idx) where line_count changes from 1 to 0
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[q_idx, d_idx]:
                if line_count[n_idx, d_idx] == 1:
                    changes.append((n_idx, d_idx))
        return changes

    def apply_remove(q_idx, changes):
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] -= 1
            zero_counts[n_idx] += 1
        in_S[q_idx] = False

    def try_swap(p_add, q_rem):
        # Check if S \ {q} U {p} is valid
        # 1. Simulate remove q
        # 2. Simulate add p
        # We need to check final zero_counts > 0
        
        # Get changes for remove
        rem_changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[q_rem, d_idx]:
                if line_count[n_idx, d_idx] == 1:
                    rem_changes.append((n_idx, d_idx))
        
        # Get changes for add
        add_changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_add, d_idx]:
                # Note: if n_idx is affected by both add and remove, we need to be careful
                # But we can just compute the net change in line_count for each (n, d)
                pass
        
        # Better approach:
        # Calculate delta for each (n, d)
        # delta(n, d) = (1 if n in line(p, d) else 0) - (1 if n in line(q, d) else 0)
        # But line(p, d) excludes p. line(q, d) excludes q.
        # If n == p, it's not in line(p, d).
        # If n == q, it's not in line(q, d).
        
        # Let's just apply to a temporary copy? No, too slow.
        # Let's collect all affected (n, d) pairs.
        affected_pairs = set()
        for d_idx in range(num_dirs):
            for n_idx in affected[p_add, d_idx]:
                affected_pairs.add((n_idx, d_idx))
            for n_idx in affected[q_rem, d_idx]:
                affected_pairs.add((n_idx, d_idx))
        
        # Check validity for these pairs
        # We need to know the new line_count for these pairs.
        # new_count = old_count + delta
        # delta = (1 if p_add contributes to n on d) - (1 if q_rem contributes to n on d)
        
        # p_add contributes to n on d if n in affected[p_add, d]
        # q_rem contributes to n on d if n in affected[q_rem, d]
        
        # To avoid set overhead, we can use a dictionary or just iterate and track.
        # Given small size, a dict is fine.
        
        # Actually, we can just check if any zero_count becomes 0.
        # For each n in affected_pairs:
        #   new_zero_count = zero_counts[n]
        #   For each d in dirs affecting n:
        #      old_c = line_count[n, d]
        #      add_c = 1 if n in affected[p_add, d] else 0
        #      rem_c = 1 if n in affected[q_rem, d] else 0
        #      new_c = old_c + add_c - rem_c
        #      if old_c == 0 and new_c > 0: new_zero_count -= 1
        #      if old_c > 0 and new_c == 0: new_zero_count += 1
        #   if new_zero_count == 0: return None
        
        # This is O(affected_pairs * num_dirs).
        # affected_pairs size <= 2 * (q-1) * num_dirs? No, union.
        # Max 2 * 48 * 50 = 4800 pairs.
        # 4800 * 50 = 240k ops per swap check.
        # Too slow for many swaps.
        
        # Optimization:
        # Only check n where zero_counts[n] is small (1 or 2).
        # If zero_counts[n] > 2, it can't become 0 unless multiple counts drop.
        # But a single swap changes at most 1 count per direction for a given n?
        # No, p and q are distinct.
        # For a fixed n, d:
        #   add_c is 0 or 1.
        #   rem_c is 0 or 1.
        #   So new_c is old_c - 1, old_c, or old_c + 1.
        #   zero_count changes by at most 1 for each direction.
        #   Total change in zero_counts[n] is sum of changes over d.
        
        # We can just do the full check but optimize the inner loop.
        
        # Let's try a simpler check:
        # Apply remove to a copy of line_count? No.
        
        # I will implement a direct check for swap validity.
        
        valid = True
        # Create a set of affected n's to avoid duplicate checks
        # But we need to check per direction.
        
        # Let's just iterate through all n in the union of affected points.
        # To get the union efficiently:
        # We can mark them.
        
        # Actually, let's just try the swap on a copy of the state?
        # State size: line_count (2401*50) + zero_counts (2401).
        # Copying is ~120k ints. ~500KB.
        # If we do 1000 swaps/sec, that's 500MB/s. Might be okay.
        # But 600s budget.
        
        # Let's stick to the incremental check but optimize.
        
        # For each n in affected_union:
        #   Calculate new_zero_count.
        
        # To get affected_union:
        #   It's the set of n such that n in line(p, d) or n in line(q, d) for some d.
        
        # I will use a dictionary to count deltas per (n, d).
        deltas = {}
        for d_idx in range(num_dirs):
            for n_idx in affected[p_add, d_idx]:
                key = (n_idx, d_idx)
                deltas[key] = deltas.get(key, 0) + 1
            for n_idx in affected[q_rem, d_idx]:
                key = (n_idx, d_idx)
                deltas[key] = deltas.get(key, 0) - 1
        
        # Now check validity
        # For each n, we need to know if zero_counts[n] becomes 0.
        # We can aggregate deltas per n.
        n_deltas = {}
        for (n_idx, d_idx), delta in deltas.items():
            if delta != 0:
                # We need to check if line_count[n, d] + delta == 0 or line_count[n, d] == 0 and delta != 0
                # Actually, we just need to know if the number of zeros for n changes.
                # Let's compute new_zero_count for each n involved.
                pass
        
        # This is getting complicated.
        # Given the time, I will implement a simpler local search:
        # 1. Greedy Add.
        # 2. If stuck, try Remove-Add sequence (2-opt).
        
        return None # Placeholder

    # --- Main Loop ---
    np.random.seed(42)
    best_S_size = 0
    best_S = set()

    for _ in range(20): # More restarts
        if time.time() > deadline - 10:
            break
            
        # Greedy Init
        seed_idx = np.random.randint(NPTS)
        S = {seed_idx}
        recompute(S)
        
        if not is_valid():
            continue

        outside = [i for i in range(NPTS) if i not in S]
        
        # Greedy Phase
        while time.time() < deadline - 10:
            np.random.shuffle(outside)
            added_any = False
            for p_idx in outside:
                if time.time() > deadline - 10:
                    break
                if p_idx in S:
                    continue
                changes = try_add(p_idx)
                if changes is not None:
                    apply_add(p_idx, changes)
                    S.add(p_idx)
                    outside.remove(p_idx)
                    added_any = True
            if not added_any:
                break
        
        # Local Search Phase (Swaps)
        # Try to improve by swapping
        # We'll do a simple hill climbing with random swaps
        stuck_count = 0
        while time.time() < deadline - 10 and stuck_count < 100:
            # Pick random q in S, p not in S
            q_idx = np.random.choice(list(S))
            p_idx = np.random.randint(NPTS)
            if p_idx in S:
                continue
                
            # Check swap validity
            # I'll implement a fast check for swap
            # If valid, accept.
            
            # For now, skip swaps if too slow to implement correctly.
            # Just try to add more if possible?
            # The greedy phase should have exhausted adds.
            
            # Let's try a perturbation: remove a random point, then greedy add.
            # This might escape local optima.
            if stuck_count % 10 == 0:
                # Perturb
                q_idx = np.random.choice(list(S))
                rem_changes = try_remove(q_idx)
                apply_remove(q_idx, rem_changes)
                S.remove(q_idx)
                
                # Greedy re-add
                outside = [i for i in range(NPTS) if i not in S]
                np.random.shuffle(outside)
                for p_idx in outside:
                    if time.time() > deadline - 10:
                        break
                    changes = try_add(p_idx)
                    if changes is not None:
                        apply_add(p_idx, changes)
                        S.add(p_idx)
                
            stuck_count += 1

        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)
            # Update state to match best_S for next iteration?
            # No, we restart from scratch each time in the outer loop.

    if not best_S:
        best_S = {0}
        recompute(best_S)

    S_final = best_S
    arr = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = pt_to_idx([[xa, xb], [ya, yb]])
                    if idx not in S_final:
                        arr.append([[xa % p, xb % p], [ya % p, yb % p]])
    
    return np.array(arr, dtype=np.int64)

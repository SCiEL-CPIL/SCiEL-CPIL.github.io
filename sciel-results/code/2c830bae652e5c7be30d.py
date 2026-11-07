"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    n = 128

    # ---- Fast validity check (bitset-based) ----
    def is_valid_mask(mask):
        """Check if set (bitmask) violates the no-triple condition."""
        for i in range(n):
            if not (mask >> i) & 1:
                continue
            for j in range(i + 1, n):
                if not (mask >> j) & 1:
                    continue
                s_idx = i + j + 1
                if s_idx >= n:
                    continue
                if not (mask >> s_idx) & 1:
                    continue
                # Need c > b (k > j) with a+c, b+c in S
                candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
                if candidates & ~((1 << (j + 1)) - 1):
                    return False
        return True

    # ---- Incremental validity: can we add x to mask? ----
    def is_valid_add(mask, x):
        """Check if adding x (1..128) to mask keeps validity."""
        bit = 1 << (x - 1)
        if mask & bit:
            return True
        new_mask = mask | bit
        xi = x - 1
        # New violations must include x.
        # Case A: x = a (smallest of triple a<b<c): need b,c in S with x+b, x+c, b+c in S
        # Case B: x = b: need a<x, c>x in S with a+x, a+c, x+c in S
        # Case C: x = c (largest): need a<b<x in S with a+b, a+x, b+x in S
        # Also x could be one of the SUMS: a+b=x, a+c, b+c in S
        # We just check the full mask for speed; n=128 is small.
        return is_valid_mask(new_mask)

    # ---- Precompute pair sums for speed ----
    # For each value v (1..128), precompute pairs (a,b) with a<b, a+b=v
    pair_sums = [[] for _ in range(2 * n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            s = a + b
            if s <= 2 * n:
                pair_sums[s].append((a, b))

    # ---- Set/mask conversions ----
    def set_to_mask(S):
        mask = 0
        for v in S:
            mask |= (1 << (v - 1))
        return mask

    def mask_to_set(mask):
        result = []
        for i in range(n):
            if (mask >> i) & 1:
                result.append(i + 1)
        return result

    # ---- Phase 1: Structured initialization ----
    best_mask = 0
    best_size = 0

    # Known good baseline: {16..31} | {64..128}, size 81
    baseline = set(range(16, 32)) | set(range(64, 129))
    baseline_mask = set_to_mask(baseline)
    if is_valid_mask(baseline_mask):
        best_mask = baseline_mask
        best_size = baseline_mask.bit_count()

    # Try different K values for H = {K..128}
    for K in range(50, 80):
        if time.time() > deadline - 20:
            break
        H = set(range(K, 129))
        max_L = (K - 1) // 2
        if max_L < 1:
            continue
        start_L = max_L // 2 + 1
        L = set(range(start_L, max_L + 1))
        S = L | H
        mask = set_to_mask(S)
        if is_valid_mask(mask):
            if mask.bit_count() > best_size:
                best_size = mask.bit_count()
                best_mask = mask

    # Try interval {63..128}
    interval_mask = set_to_mask(set(range(63, 129)))
    if is_valid_mask(interval_mask):
        if interval_mask.bit_count() > best_size:
            best_size = interval_mask.bit_count()
            best_mask = interval_mask

    # ---- Phase 2: Greedy add (multiple passes) ----
    for _ in range(8):
        if time.time() > deadline - 25:
            break
        improved = False
        for v in range(1, 129):
            if time.time() > deadline - 25:
                break
            bit = 1 << (v - 1)
            if best_mask & bit:
                continue
            trial_mask = best_mask | bit
            if is_valid_mask(trial_mask):
                best_mask = trial_mask
                best_size = best_mask.bit_count()
                improved = True
        if not improved:
            break

    # ---- Phase 3: Single-swap local search ----
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for rem in range(1, 129):
            if time.time() > deadline - 15:
                break
            rem_bit = 1 << (rem - 1)
            if not (best_mask & rem_bit):
                continue
            trial_base = best_mask & ~rem_bit
            for add in range(1, 129):
                if add == rem:
                    continue
                add_bit = 1 << (add - 1)
                if trial_base & add_bit:
                    continue
                trial = trial_base | add_bit
                if is_valid_mask(trial):
                    if trial.bit_count() > best_size:
                        best_mask = trial
                        best_size = trial.bit_count()
                        improved = True
                        break

    # ---- Phase 4: Two-swap local search (remove 2, add 2) ----
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            in_set = [x for x in range(1, 129) if (best_mask >> (x - 1)) & 1]
            not_in_set = [x for x in range(1, 129) if not (best_mask >> (x - 1)) & 1]
            if not in_set or not not_in_set:
                break
            for idx1 in range(len(in_set)):
                if time.time() > deadline - 10:
                    break
                rem1 = in_set[idx1]
                rem1_bit = 1 << (rem1 - 1)
                for idx2 in range(idx1 + 1, len(in_set)):
                    if time.time() > deadline - 10:
                        break
                    rem2 = in_set[idx2]
                    rem2_bit = 1 << (rem2 - 1)
                    trial_base = best_mask & ~rem1_bit & ~rem2_bit
                    for add1 in not_in_set:
                        if time.time() > deadline - 10:
                            break
                        add1_bit = 1 << (add1 - 1)
                        if trial_base & add1_bit:
                            continue
                        trial1 = trial_base | add1_bit
                        # Quick check: does adding add1 to base work?
                        if not is_valid_mask(trial1):
                            continue
                        for add2 in not_in_set:
                            if add2 == add1:
                                continue
                            add2_bit = 1 << (add2 - 1)
                            if trial_base & add2_bit:
                                continue
                            trial = trial1 | add2_bit
                            if is_valid_mask(trial):
                                if trial.bit_count() > best_size:
                                    best_mask = trial
                                    best_size = trial.bit_count()
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # ---- Phase 5: Randomized local search with swaps and additions ----
    random.seed(12345)
    current_mask = best_mask
    current_size = best_size

    if time.time() < deadline - 5:
        for iteration in range(50000):
            if time.time() > deadline - 5:
                break

            action = random.random()

            if action < 0.35:
                # Try adding an element
                not_in_set = [x for x in range(1, 129) if not (current_mask >> (x - 1)) & 1]
                if not not_in_set:
                    continue
                to_add = random.choice(not_in_set)
                new_mask = current_mask | (1 << (to_add - 1))
                if is_valid_mask(new_mask):
                    current_mask = new_mask
                    current_size = current_mask.bit_count()
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size

            elif action < 0.75:
                # Random swap
                in_set = [x for x in range(1, 129) if (current_mask >> (x - 1)) & 1]
                not_in_set = [x for x in range(1, 129) if not (current_mask >> (x - 1)) & 1]
                if not in_set or not not_in_set:
                    continue
                to_remove = random.choice(in_set)
                to_add = random.choice(not_in_set)
                new_mask = (current_mask & ~(1 << (to_remove - 1))) | (1 << (to_add - 1))
                if is_valid_mask(new_mask):
                    current_mask = new_mask
                    current_size = current_mask.bit_count()
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size

            else:
                # Two-swap: remove 2, add 2
                in_set = [x for x in range(1, 129) if (current_mask >> (x - 1)) & 1]
                not_in_set = [x for x in range(1, 129) if not (current_mask >> (x - 1)) & 1]
                if len(in_set) < 2 or len(not_in_set) < 2:
                    continue
                r1, r2 = random.sample(in_set, 2)
                a1, a2 = random.sample(not_in_set, 2)
                new_mask = current_mask
                new_mask &= ~(1 << (r1 - 1))
                new_mask &= ~(1 << (r2 - 1))
                new_mask |= (1 << (a1 - 1))
                new_mask |= (1 << (a2 - 1))
                if is_valid_mask(new_mask):
                    current_mask = new_mask
                    current_size = current_mask.bit_count()
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size

    # ---- Phase 6: Branch-and-bound on small candidate pool ----
    # Try to find a better solution by exploring a restricted search space
    if time.time() < deadline - 2:
        # Use the current best as a starting point
        # Identify "free" elements that could potentially be added
        # with minimal removals
        not_in_best = [x for x in range(1, 129) if not (best_mask >> (x - 1)) & 1]
        
        # For each candidate, compute how many removals needed
        # and try the most promising ones with B&B
        candidates_by_cost = []
        for x in not_in_best:
            # Count how many violations x creates
            new_mask = best_mask | (1 << (x - 1))
            # Count violating triples involving x
            xi = x - 1
            violations = 0
            for i in range(n):
                if not (new_mask >> i) & 1:
                    continue
                for j in range(i + 1, n):
                    if not (new_mask >> j) & 1:
                        continue
                    s_idx = i + j + 1
                    if s_idx >= n:
                        continue
                    if not (new_mask >> s_idx) & 1:
                        continue
                    candidates = new_mask & (new_mask >> (i + 1)) & (new_mask >> (j + 1))
                    cands = candidates & ~((1 << (j + 1)) - 1)
                    while cands:
                        lsb = cands & -cands
                        k = lsb.bit_length() - 1
                        cands ^= lsb
                        # This violation involves elements i+1, j+1, k+1
                        # and sums (i+j+2), (i+k+2), (j+k+2)
                        # x must be one of these 6 values for it to be "new"
                        vals = {i+1, j+1, k+1, i+j+2, i+k+2, j+k+2}
                        if x in vals:
                            violations += 1
            candidates_by_cost.append((violations, x))
        
        candidates_by_cost.sort()
        
        # Try the top candidates with a simple B&B: remove the minimum set to fix
        for cost, x in candidates_by_cost[:20]:
            if time.time() > deadline - 2:
                break
            if cost <= 1:
                # Can fix by removing at most 1 element
                new_mask = best_mask | (1 << (x - 1))
                # Find which single removal fixes it
                for r in range(1, 129):
                    if r == x:
                        continue
                    trial = new_mask & ~(1 << (r - 1))
                    if is_valid_mask(trial):
                        if trial.bit_count() > best_size:
                            best_mask = trial
                            best_size = best_mask.bit_count()
                        break
            elif cost == 2:
                # Try removing pairs (limited search)
                new_mask = best_mask | (1 << (x - 1))
                # Find elements involved in violations
                involved = set()
                for i in range(n):
                    if not (new_mask >> i) & 1:
                        continue
                    for j in range(i + 1, n):
                        if not (new_mask >> j) & 1:
                            continue
                        s_idx = i + j + 1
                        if s_idx >= n:
                            continue
                        if not (new_mask >> s_idx) & 1:
                            continue
                        candidates = new_mask & (new_mask >> (i + 1)) & (new_mask >> (j + 1))
                        cands = candidates & ~((1 << (j + 1)) - 1)
                        while cands:
                            lsb = cands & -cands
                            k = lsb.bit_length() - 1
                            cands ^= lsb
                            vals = {i+1, j+1, k+1, i+j+2, i+k+2, j+k+2}
                            if x in vals:
                                for v in vals:
                                    if v != x:
                                        involved.add(v)
                # Try removing pairs from involved
                inv_list = list(involved)
                for idx1 in range(min(len(inv_list), 15)):
                    if time.time() > deadline - 2:
                        break
                    r1 = inv_list[idx1]
                    for idx2 in range(idx1 + 1, min(len(inv_list), 15)):
                        r2 = inv_list[idx2]
                        trial = new_mask & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))
                        if is_valid_mask(trial):
                            if trial.bit_count() > best_size:
                                best_mask = trial
                                best_size = best_mask.bit_count()
                            break

    # Final verification
    if not is_valid_mask(best_mask):
        best_mask = set_to_mask(set(range(63, 129)))
        best_size = 66

    result = mask_to_set(best_mask)
    result = [int(x) for x in result]
    return result

# EVOLVE-BLOCK-END

import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    conflict_mask = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute reciprocal weights
    weights = [Fraction(1, values[i]) for i in range(size)]
    weight_list = [1.0 / values[i] for i in range(size)]

    def score_mask(mask):
        total = Fraction(0, 1)
        i = 0
        m = mask
        while m:
            if m & 1:
                total += weights[i]
            i += 1
            m >>= 1
        return total

    def is_valid_mask(mask):
        i = 0
        m = mask
        while m:
            if m & 1:
                if conflict_mask[i] & mask:
                    return False
            i += 1
            m >>= 1
        return True

    def greedy_construct(order):
        S_mask = 0
        for i in order:
            if not (conflict_mask[i] & S_mask):
                S_mask |= (1 << i)
        return S_mask

    # Conflict counts
    conflict_counts = [bin(cm).count('1') for cm in conflict_mask]

    # Identify "safe" numbers: those with no conflicts among themselves
    # Numbers >= 129 are all mutually compatible
    safe_threshold = 129
    safe_indices = [i for i in range(size) if values[i] >= safe_threshold]
    unsafe_indices = [i for i in range(size) if values[i] < safe_threshold]

    # Verify: are all safe_indices mutually compatible?
    for i in range(len(safe_indices)):
        for j in range(i+1, len(safe_indices)):
            if conflict_mask[safe_indices[i]] & (1 << safe_indices[j]):
                safe_threshold = values[i] + 1  # This shouldn't happen, but guard

    # Build the base set: all safe indices
    base_mask = 0
    for i in safe_indices:
        base_mask |= (1 << i)
    base_score = score_mask(base_mask)

    # For each unsafe index, compute which safe indices it conflicts with
    # and whether it can be added to the base set
    compatible_with_base = []
    for i in unsafe_indices:
        if not (conflict_mask[i] & base_mask):
            compatible_with_base.append(i)

    # Sort compatible unsafe indices by weight descending
    compatible_with_base.sort(key=lambda x: weight_list[x], reverse=True)

    # Now we need to select a subset of compatible_with_base that is mutually compatible
    # This is a smaller MWIS problem
    n_compat = len(compatible_with_base)

    # Precompute conflicts among compatible_with_base
    compat_conflict = [0] * n_compat
    for a in range(n_compat):
        ia = compatible_with_base[a]
        for b in range(a+1, n_compat):
            ib = compatible_with_base[b]
            if conflict_mask[ia] & (1 << ib):
                compat_conflict[a] |= (1 << b)
                compat_conflict[b] |= (1 << a)

    # Branch and bound on the compatible unsafe subset
    # Upper bound: sum of weights of all remaining candidates (relaxed)
    # Better: greedy upper bound

    best_subset_mask = 0  # bitmask over n_compat
    best_subset_score = Fraction(0, 1)

    # Sort candidates by weight descending for B&B
    order_bb = list(range(n_compat))
    order_bb.sort(key=lambda x: weight_list[compatible_with_base[x]], reverse=True)

    # Precompute for B&B
    def upper_bound(candidates_mask, pos):
        """Upper bound: sum of weights of all candidates in candidates_mask that are at or after pos in order_bb"""
        total = 0.0
        for idx in range(pos, n_compat):
            if candidates_mask & (1 << idx):
                total += weight_list[compatible_with_base[idx]]
        return total

    # Recursive B&B
    def branch_and_bound(candidates_mask, current_mask, current_score, pos, deadline):
        nonlocal best_subset_mask, best_subset_score

        if time.time() > deadline:
            return

        if pos >= n_compat:
            if current_score > best_subset_score:
                best_subset_score = current_score
                best_subset_mask = current_mask
            return

        # Upper bound check
        ub = current_score
        for idx in range(pos, n_compat):
            if candidates_mask & (1 << idx):
                ub += weights[compatible_with_base[idx]]
        if ub <= best_subset_score:
            return

        # Try including each candidate from pos onwards
        for idx in range(pos, n_compat):
            if time.time() > deadline:
                return
            if not (candidates_mask & (1 << idx)):
                continue

            ia = compatible_with_base[idx]
            # Check if ia is compatible with all in current_mask
            if compat_conflict[idx] & current_mask:
                continue

            new_mask = current_mask | (1 << idx)
            new_score = current_score + weights[ia]
            new_candidates = candidates_mask & ~(1 << idx)
            # Remove candidates that conflict with ia
            new_candidates &= ~compat_conflict[idx]

            branch_and_bound(new_candidates, new_mask, new_score, idx + 1, deadline)

            # Try excluding idx: just continue with same candidates but skip idx
            # This is handled by the loop continuing

    # Run B&B
    all_candidates = (1 << n_compat) - 1
    branch_and_bound(all_candidates, 0, Fraction(0, 1), 0, start_time + timeout - 10)

    # Build final solution
    final_mask = base_mask
    for i in range(n_compat):
        if best_subset_mask & (1 << i):
            final_mask |= (1 << compatible_with_base[i])

    # Local search polish
    def local_search(S_mask, deadline):
        current_score = score_mask(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        weight_sorted = sorted(range(size), key=lambda x: weight_list[x], reverse=True)

        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break

            # Try swaps
            for u in range(size):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                best_v = -1
                best_gain = Fraction(0, 1)
                for v in weight_sorted:
                    if in_set[v]:
                        continue
                    if weight_list[v] <= weight_list[u]:
                        break
                    if conflict_mask[v] & S_without_u:
                        continue
                    best_v = v
                    best_gain = weights[v] - weights[u]
                    break
                if best_v >= 0 and best_gain > 0:
                    S_mask = S_without_u | (1 << best_v)
                    in_set[u] = False
                    in_set[best_v] = True
                    current_score += best_gain
                    improved = True
                    if time.time() > deadline:
                        return S_mask, current_score

            # Try additions
            for v in weight_sorted:
                if in_set[v]:
                    continue
                if not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    if time.time() > deadline:
                        return S_mask, current_score

        return S_mask, current_score

    # Polish the B&B result
    if time.time() < timeout - 5:
        final_mask, final_score = local_search(final_mask, timeout - 3)

    # Validate
    if not is_valid_mask(final_mask) or final_mask == 0:
        # Fallback to greedy
        order_asc = list(range(size))
        fallback = greedy_construct(order_asc)
        fallback, _ = local_search(fallback, timeout - 3)
        if is_valid_mask(fallback):
            final_mask = fallback

    result = [values[i] for i in range(size) if final_mask & (1 << i)]
    return result if result else [2]

# EVOLVE-BLOCK-END

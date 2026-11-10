import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 82.0  # Safety margin for 90s limit

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

    # Conflict counts for diversity
    conflict_counts = [bin(cm).count('1') for cm in conflict_mask]

    # Pre-sorted orders
    order_asc = list(range(size))
    order_desc = list(range(size - 1, -1, -1))
    order_by_conflict = sorted(range(size), key=lambda x: (conflict_counts[x], x))
    order_by_ratio = sorted(range(size), key=lambda x: (weight_list[x] / (conflict_counts[x] + 1)), reverse=True)
    # Additional ordering: by value descending (favor larger numbers which have fewer conflicts)
    order_by_value_desc = sorted(range(size), key=lambda x: values[x], reverse=True)
    # By weight/conflict² ratio
    order_by_ratio2 = sorted(range(size), key=lambda x: (weight_list[x] / ((conflict_counts[x] + 1) ** 2)), reverse=True)

    # Weight-sorted indices for fast best-v lookup
    weight_sorted = sorted(range(size), key=lambda x: weight_list[x], reverse=True)

    # Local search using swaps, additions, and optimized 2-exchanges
    def local_search(S_mask, deadline):
        current_score = score_mask(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break

            # Try swaps: remove u, add v (only best gain per u)
            for u in range(size):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                best_v = -1
                best_gain = Fraction(0, 1)
                # Iterate in weight-descending order; first compatible v with positive gain is best
                for v in weight_sorted:
                    if in_set[v]:
                        continue
                    if weight_list[v] <= weight_list[u]:
                        break  # no further v can have positive gain
                    if conflict_mask[v] & S_without_u:
                        continue
                    best_v = v
                    best_gain = weights[v] - weights[u]
                    break  # first compatible in weight order is optimal
                if best_v >= 0 and best_gain > 0:
                    S_mask = S_without_u | (1 << best_v)
                    in_set[u] = False
                    in_set[best_v] = True
                    current_score += best_gain
                    improved = True
                    if time.time() - start_time > deadline:
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
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

            # Try 2-exchanges: remove u1, u2; add v1, v2
            # Optimized: only try if we have time remaining
            if time.time() - start_time < deadline - 5:
                in_set_indices = [i for i in range(size) if in_set[i]]
                n_in = len(in_set_indices)
                if n_in >= 2:
                    for idx1 in range(n_in):
                        if time.time() - start_time > deadline:
                            break
                        u1 = in_set_indices[idx1]
                        for idx2 in range(idx1 + 1, n_in):
                            u2 = in_set_indices[idx2]
                            S_without_pair = S_mask & ~(1 << u1) & ~(1 << u2)
                            base_loss = weights[u1] + weights[u2]
                            base_loss_float = weight_list[u1] + weight_list[u2]
                            
                            # Find best pair (v1, v2) not in S, compatible with S_without_pair
                            # and compatible with each other, maximizing weight[v1]+weight[v2]-weight[u1]-weight[u2]
                            best_pair = (-1, -1)
                            best_pair_gain = Fraction(0, 1)
                            best_pair_gain_float = 0.0
                            
                            # Build list of candidates compatible with S_without_pair
                            candidates = []
                            for v in weight_sorted:
                                if in_set[v] or v == u1 or v == u2:
                                    continue
                                if conflict_mask[v] & S_without_pair:
                                    continue
                                candidates.append(v)
                                if weight_list[v] * 2 <= base_loss_float:
                                    break  # Even best pair with this v can't beat loss
                            
                            # Try pairs from candidates
                            n_cand = len(candidates)
                            for i1 in range(n_cand):
                                v1 = candidates[i1]
                                if weight_list[v1] * 2 <= base_loss_float:
                                    break
                                for i2 in range(i1 + 1, n_cand):
                                    v2 = candidates[i2]
                                    if weight_list[v1] + weight_list[v2] <= base_loss_float:
                                        break
                                    if conflict_mask[v1] & (1 << v2):
                                        continue
                                    gain_float = weight_list[v1] + weight_list[v2] - base_loss_float
                                    if gain_float > best_pair_gain_float:
                                        best_pair_gain = weights[v1] + weights[v2] - base_loss
                                        best_pair_gain_float = gain_float
                                        best_pair = (v1, v2)
                            
                            if best_pair_gain > 0:
                                v1, v2 = best_pair
                                S_mask = S_without_pair | (1 << v1) | (1 << v2)
                                in_set[u1] = False
                                in_set[u2] = False
                                in_set[v1] = True
                                in_set[v2] = True
                                current_score += best_pair_gain
                                improved = True
                                if time.time() - start_time > deadline:
                                    return S_mask, current_score
                                break  # restart u1 loop after a successful 2-exchange

        return S_mask, current_score

    # Generate diverse initial solutions
    initial_solutions = [
        greedy_construct(order_asc),
        greedy_construct(order_desc),
        greedy_construct(order_by_conflict),
        greedy_construct(order_by_ratio),
        greedy_construct(order_by_value_desc),
        greedy_construct(order_by_ratio2),
    ]

    # Add randomized starts
    import random
    random.seed(42)
    num_random = 10
    for _ in range(num_random):
        shuffled = list(range(size))
        random.shuffle(shuffled)
        initial_solutions.append(greedy_construct(shuffled))

    # Local search on each, keeping track of best
    best_mask = initial_solutions[0]
    best_score = score_mask(best_mask)

    # Process in order of initial score (best first) to maximize time on promising solutions
    scored_solutions = [(score_mask(S), S) for S in initial_solutions]
    scored_solutions.sort(key=lambda x: x[0], reverse=True)

    for score_init, S in scored_solutions:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
        # Give more time to better initial solutions
        time_budget = min(15.0, timeout - elapsed - 3)
        if time_budget < 2:
            break
        S_opt, val = local_search(S, start_time + time_budget)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Final polish on the best solution with remaining time
    elapsed = time.time() - start_time
    if elapsed < timeout - 5:
        final_deadline = timeout - 2
        S_final, val_final = local_search(best_mask, final_deadline)
        if val_final > best_score:
            best_score = val_final
            best_mask = S_final

    # Final check
    if not is_valid_mask(best_mask) or best_mask == 0:
        fallback = greedy_construct(order_asc)
        fallback, val = local_search(fallback, timeout - 3)
        if val > best_score:
            best_score = val
            best_mask = fallback

    result = [values[i] for i in range(size) if best_mask & (1 << i)]
    return result if result else [2]

# EVOLVE-BLOCK-END

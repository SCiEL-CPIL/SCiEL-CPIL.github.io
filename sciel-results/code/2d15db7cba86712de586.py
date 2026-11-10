import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 86.0  # Safety margin for 90s limit

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
    weight_float = [1.0 / values[i] for i in range(size)]

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

    def local_search(S_mask, deadline):
        current_score = score_mask(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        # Build list of elements in set and not in set for faster iteration
        def get_in_set_list():
            return [i for i in range(size) if in_set[i]]

        def get_not_in_set_list():
            return [i for i in range(size) if not in_set[i]]

        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break

            in_list = get_in_set_list()
            not_in_list = get_not_in_set_list()

            # Try swaps: remove u, add v (use float for speed, verify with Fraction)
            best_swap = None
            best_swap_gain_float = 0.0
            for u in in_list:
                S_without_u = S_mask & ~(1 << u)
                for v in not_in_list:
                    if conflict_mask[v] & S_without_u:
                        continue
                    gain_float = weight_float[v] - weight_float[u]
                    if gain_float > best_swap_gain_float:
                        best_swap_gain_float = gain_float
                        best_swap = (u, v)

            if best_swap and best_swap_gain_float > 1e-12:
                u, v = best_swap
                S_without_u = S_mask & ~(1 << u)
                # Verify with Fraction
                gain_exact = weights[v] - weights[u]
                if gain_exact > 0:
                    S_mask = S_without_u | (1 << v)
                    in_set[u] = False
                    in_set[v] = True
                    current_score += gain_exact
                    improved = True
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

            # Try additions
            for v in not_in_list:
                if not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

            # Try 2-for-1 exchanges: remove u1, u2, add v
            # This can improve when removing two small-weight elements allows adding one large-weight element
            # Only do this if we have time
            if time.time() - start_time < deadline - 5:
                in_list = get_in_set_list()
                not_in_list = get_not_in_set_list()
                best_2for1 = None
                best_2for1_gain_float = 0.0
                for i in range(len(in_list)):
                    for j in range(i + 1, len(in_list)):
                        u1 = in_list[i]
                        u2 = in_list[j]
                        S_without_both = S_mask & ~(1 << u1) & ~(1 << u2)
                        for v in not_in_list:
                            if conflict_mask[v] & S_without_both:
                                continue
                            gain_float = weight_float[v] - weight_float[u1] - weight_float[u2]
                            if gain_float > best_2for1_gain_float:
                                best_2for1_gain_float = gain_float
                                best_2for1 = (u1, u2, v)
                if best_2for1 and best_2for1_gain_float > 1e-12:
                    u1, u2, v = best_2for1
                    S_without_both = S_mask & ~(1 << u1) & ~(1 << u2)
                    gain_exact = weights[v] - weights[u1] - weights[u2]
                    if gain_exact > 0:
                        S_mask = S_without_both | (1 << v)
                        in_set[u1] = False
                        in_set[u2] = False
                        in_set[v] = True
                        current_score += gain_exact
                        improved = True
                        if time.time() - start_time > deadline:
                            return S_mask, current_score

        return S_mask, current_score

    # Generate diverse initial solutions
    initial_solutions = []

    # 1. Greedy by ascending index (smallest values first - highest weight)
    order_asc = list(range(size))
    initial_solutions.append(greedy_construct(order_asc))

    # 2. Greedy by descending index (largest values first)
    order_desc = list(range(size - 1, -1, -1))
    initial_solutions.append(greedy_construct(order_desc))

    # 3. Greedy by least conflicts
    order_by_conflict = sorted(range(size), key=lambda x: (conflict_counts[x], x))
    initial_solutions.append(greedy_construct(order_by_conflict))

    # 4. Greedy by weight/conflict ratio
    order_by_ratio = sorted(range(size), key=lambda x: (weight_float[x] / (conflict_counts[x] + 1)), reverse=True)
    initial_solutions.append(greedy_construct(order_by_ratio))

    # 5. Greedy by weight only (descending weight = ascending value)
    order_by_weight = sorted(range(size), key=lambda x: weight_float[x], reverse=True)
    initial_solutions.append(greedy_construct(order_by_weight))

    # 6-15. Random restarts with different seeds
    import random
    for seed in range(10):
        random.seed(seed * 1000 + 42)
        shuffled = list(range(size))
        random.shuffle(shuffled)
        initial_solutions.append(greedy_construct(shuffled))

    # Local search on each, keep best
    best_mask = initial_solutions[0]
    best_score = score_mask(best_mask)

    for S in initial_solutions:
        elapsed = time.time() - start_time
        if elapsed > timeout - 5:
            break
        S_opt, val = local_search(S, timeout - 3)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Final polish: one more local search pass on the best solution
    if time.time() - start_time < timeout - 2:
        S_opt, val = local_search(best_mask, timeout - 1)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Final check
    if not is_valid_mask(best_mask) or best_mask == 0:
        fallback = greedy_construct(order_asc)
        if time.time() - start_time < timeout - 2:
            fallback, val = local_search(fallback, timeout - 1)
            if val > best_score:
                best_score = val
                best_mask = fallback
        else:
            best_mask = fallback

    result = [values[i] for i in range(size) if best_mask & (1 << i)]
    return result if result else [2]

# EVOLVE-BLOCK-END

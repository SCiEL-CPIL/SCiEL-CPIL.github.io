import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 88.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    # We use a Python int bitmask for conflicts; bit j is set in conflict_mask[i] if (i,j) conflict
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
    # weight/conflict ratio
    order_by_ratio = sorted(range(size), key=lambda x: (weight_list[x] / (conflict_counts[x] + 1)), reverse=True)

    # Local search using swaps and additions
    def local_search(S_mask, deadline):
        current_score = score_mask(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        # Precompute sorted list of elements by weight (descending) for fast best-v search
        # For each u in S, we want to find best v not in S, compatible with S\{u}, with weight[v] > weight[u]
        # We'll iterate u and for each v check

        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break

            # Try swaps: remove u, add v
            for u in range(size):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                best_v = -1
                best_gain = Fraction(0, 1)
                for v in range(size):
                    if in_set[v]:
                        continue
                    if conflict_mask[v] & S_without_u:
                        continue
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v >= 0 and best_gain > 0:
                    S_mask = S_without_u | (1 << best_v)
                    in_set[u] = False
                    in_set[best_v] = True
                    current_score += best_gain
                    improved = True
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

            # Try additions
            for v in range(size):
                if in_set[v]:
                    continue
                if not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

        return S_mask, current_score

    # Generate diverse initial solutions
    initial_solutions = [
        greedy_construct(order_asc),
        greedy_construct(order_desc),
        greedy_construct(order_by_conflict),
        greedy_construct(order_by_ratio),
    ]

    # Add randomized starts
    import random
    random.seed(42)
    num_random = 15
    for _ in range(num_random):
        shuffled = list(range(size))
        random.shuffle(shuffled)
        initial_solutions.append(greedy_construct(shuffled))

    # Local search on each
    best_mask = initial_solutions[0]
    best_score = score_mask(best_mask)

    for S in initial_solutions:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
        S_opt, val = local_search(S, timeout - 2)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Final check
    if not is_valid_mask(best_mask) or best_mask == 0:
        # Fallback
        fallback = greedy_construct(order_asc)
        fallback, val = local_search(fallback, timeout - 2)
        if val > best_score:
            best_score = val
            best_mask = fallback

    result = [values[i] for i in range(size) if best_mask & (1 << i)]
    return result if result else [2]

# EVOLVE-BLOCK-END

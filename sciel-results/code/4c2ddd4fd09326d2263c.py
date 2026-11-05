import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 148  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        if not subset:
            return True
        s_set = set(subset)
        n = len(subset)
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                c = a + subset[j]
                if c in s_set:
                    return False
        return True

    def greedy_add(S, order, C_init=None):
        """Greedy: start from C_init (or empty), add elements from S in 'order' if valid."""
        C = list(C_init) if C_init is not None else []
        C_set = set(C)
        for x in order:
            if x in C_set:
                continue
            valid = True
            for a in C:
                if (x - a) in C_set:
                    valid = False
                    break
                if (a + x) in C_set:
                    valid = False
                    break
            if valid:
                if (2 * x) in C_set:
                    valid = False
            if valid:
                C.append(x)
                C_set.add(x)
        return C

    def local_improve(S, C_init, max_rounds=30, time_limit=None):
        """Try to add elements from S\C to C in multiple orders, keep best."""
        C = list(C_init)
        C_set = set(C)
        excluded = [x for x in S if x not in C_set]
        best_C = C
        best_len = len(C)
        
        for rnd in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            orders = [
                sorted(excluded, key=lambda x: abs(x), reverse=True),
                sorted(excluded, key=lambda x: x, reverse=True),
                sorted(excluded, key=lambda x: x),
                sorted(excluded, key=lambda x: abs(x)),
            ]
            rng_local = random.Random(rnd * 1000 + len(C) * 7 + hash(frozenset(C[:10])) % 10000)
            shuffled = list(excluded)
            rng_local.shuffle(shuffled)
            orders.append(shuffled)
            
            for order in orders:
                if time_limit is not None and time.time() > time_limit:
                    break
                new_C = greedy_add(S, order, C_init=C)
                if len(new_C) > best_len:
                    best_C = new_C
                    best_len = len(new_C)
                    improved = True
                    C = new_C
                    C_set = set(C)
                    excluded = [x for x in S if x not in C_set]
            if not improved:
                break
        return best_C

    def swap_improve(S, C_init, max_rounds=20, time_limit=None):
        """Try swapping one element out and two in, or one out and one in."""
        C = list(C_init)
        C_set = set(C)
        best_C = C
        best_len = len(C)
        excluded = [x for x in S if x not in C_set]
        
        for rnd in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            # Try removing each element and adding two excluded ones
            for i in range(len(C)):
                if time_limit is not None and time.time() > time_limit:
                    break
                C_no_i = [x for j, x in enumerate(C) if j != i]
                C_no_i_set = set(C_no_i)
                # Try adding pairs from excluded
                added_any = False
                for a in excluded:
                    if a in C_no_i_set:
                        continue
                    # Check if a can be added
                    valid_a = True
                    for b in C_no_i:
                        if (a - b) in C_no_i_set or (a + b) in C_no_i_set:
                            valid_a = False
                            break
                    if valid_a and (2 * a) in C_no_i_set:
                        valid_a = False
                    if not valid_a:
                        continue
                    new_C = C_no_i + [a]
                    new_C_set = C_no_i_set | {a}
                    # Try adding another
                    for b in excluded:
                        if b in new_C_set or b == a:
                            continue
                        valid_b = True
                        for c in new_C:
                            if (b - c) in new_C_set or (b + c) in new_C_set:
                                valid_b = False
                                break
                        if valid_b and (2 * b) in new_C_set:
                            valid_b = False
                        if valid_b:
                            candidate = new_C + [b]
                            if len(candidate) > best_len:
                                best_C = candidate
                                best_len = len(candidate)
                                improved = True
                                C = candidate
                                C_set = set(C)
                                excluded = [x for x in S if x not in C_set]
                                added_any = True
                                break
                    if added_any:
                        break
            if not improved:
                break
        return best_C

    def solve_instance(S, time_budget):
        """Find best sum-free subset of S within time_budget seconds."""
        n = len(S)
        candidates = []
        
        # --- Structural candidates (fast) ---
        S_odd = [x for x in S if x % 2 != 0]
        candidates.append(S_odd)
        
        M = max(S)
        m = min(S)
        S_upper = [x for x in S if x > M / 2.0]
        candidates.append(S_upper)
        S_lower = [x for x in S if x < m / 2.0]
        candidates.append(S_lower)
        
        abs_vals = sorted(set(abs(x) for x in S))
        for K in abs_vals[:15]:
            S_K = [x for x in S if abs(x) > K]
            candidates.append(S_K)
        
        S_pos = [x for x in S if x > 0]
        candidates.append(S_pos)
        S_neg = [x for x in S if x < 0]
        candidates.append(S_neg)
        
        S_large_pos = [x for x in S if x > 0 and x > M / 3.0]
        candidates.append(S_large_pos)
        S_large_neg = [x for x in S if x < 0 and x < m / 3.0]
        candidates.append(S_large_neg)
        
        # --- Greedy from scratch with different orderings ---
        S_sorted_abs_desc = sorted(S, key=lambda x: abs(x), reverse=True)
        S_sorted_desc = sorted(S, key=lambda x: x, reverse=True)
        S_sorted_asc = sorted(S, key=lambda x: x)
        S_sorted_abs_asc = sorted(S, key=lambda x: abs(x))
        
        rng_greedy = random.Random(42)
        S_shuffled = list(S)
        rng_greedy.shuffle(S_shuffled)
        
        greedy_orders = [S_sorted_abs_desc, S_sorted_desc, S_sorted_asc, S_sorted_abs_asc, S_shuffled]
        
        for order in greedy_orders:
            if time.time() > start_time + time_budget * 0.2:
                break
            C = greedy_add(S, order)
            candidates.append(C)
        
        # --- Validate and pick best structural/greedy candidate ---
        best_C = []
        best_len = 0
        for C in candidates:
            if len(C) <= best_len:
                continue
            if is_sum_free(C):
                best_C = C
                best_len = len(C)
        
        # --- Local improvement on best candidate ---
        t_limit = start_time + time_budget * 0.6
        improved = local_improve(S, best_C, max_rounds=25, time_limit=t_limit)
        if len(improved) > best_len:
            best_C = improved
            best_len = len(best_C)
        
        # --- Swap improvement ---
        t_limit2 = start_time + time_budget * 0.85
        improved2 = swap_improve(S, best_C, max_rounds=15, time_limit=t_limit2)
        if len(improved2) > best_len:
            best_C = improved2
            best_len = len(best_C)
        
        # --- More local improvement if time remains ---
        remaining = start_time + time_budget - time.time()
        if remaining > 3:
            improved3 = local_improve(S, best_C, max_rounds=20, time_limit=start_time + time_budget - 1)
            if len(improved3) > best_len:
                best_C = improved3
        
        return best_C

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    time_alloc = [25, 40, 75]  # total 140s, leaving 10s buffer
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        C = solve_instance(S, time_alloc[i])
        
        if not is_sum_free(C):
            C = [x for x in S if x % 2 != 0]
        
        results.append(C)
    
    return results

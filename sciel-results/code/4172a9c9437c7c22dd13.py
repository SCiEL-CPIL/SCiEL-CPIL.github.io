import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

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

    def best_k_subset(S):
        candidates = set([0])
        for x in S:
            candidates.add(abs(x))
        best_subset = []
        for K in sorted(candidates):
            if time.time() - start_time > DEADLINE:
                break
            S_K = [x for x in S if abs(x) > K]
            if is_sum_free(S_K):
                return S_K
        return []

    def greedy_max_abs(S):
        S_sorted = sorted(S, key=lambda x: abs(x), reverse=True)
        C = []
        C_set = set()
        for x in S_sorted:
            valid = True
            for a in C:
                if (x - a) in C_set:
                    valid = False
                    break
            if valid:
                for a in C:
                    if (x + a) in C_set:
                        valid = False
                        break
            if valid:
                if (2 * x) in C_set:
                    valid = False
            if valid:
                C.append(x)
                C_set.add(x)
        return C

    def greedy_min_abs(S):
        S_sorted = sorted(S, key=lambda x: abs(x))
        C = []
        C_set = set()
        for x in S_sorted:
            valid = True
            for a in C:
                if (x - a) in C_set:
                    valid = False
                    break
            if valid:
                for a in C:
                    if (x + a) in C_set:
                        valid = False
                        break
            if valid:
                if (2 * x) in C_set:
                    valid = False
            if valid:
                C.append(x)
                C_set.add(x)
        return C

    def greedy_positive(S):
        S_sorted = sorted([x for x in S if x > 0], key=lambda x: x, reverse=True)
        C = []
        C_set = set()
        for x in S_sorted:
            valid = True
            for a in C:
                if (x - a) in C_set:
                    valid = False
                    break
            if valid:
                for a in C:
                    if (x + a) in C_set:
                        valid = False
                        break
            if valid:
                if (2 * x) in C_set:
                    valid = False
            if valid:
                C.append(x)
                C_set.add(x)
        return C

    def greedy_negative(S):
        S_sorted = sorted([x for x in S if x < 0], key=lambda x: abs(x), reverse=True)
        C = []
        C_set = set()
        for x in S_sorted:
            valid = True
            for a in C:
                if (x - a) in C_set:
                    valid = False
                    break
            if valid:
                for a in C:
                    if (x + a) in C_set:
                        valid = False
                        break
            if valid:
                if (2 * x) in C_set:
                    valid = False
            if valid:
                C.append(x)
                C_set.add(x)
        return C

    def local_search(S, initial_subset, time_limit):
        """Improve a sum-free subset via swap/insert/delete operations."""
        C = list(initial_subset)
        C_set = set(C)
        remaining = [x for x in S if x not in C_set]
        remaining_set = set(remaining)
        
        if not is_sum_free(C):
            return C
        
        best_size = len(C)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            improved = False
            
            # Try to insert elements from remaining
            for x in list(remaining):
                if time.time() - start_time > time_limit:
                    break
                # Check if x can be added
                valid = True
                for a in C:
                    if (x - a) in C_set:
                        valid = False
                        break
                if valid:
                    for a in C:
                        if (x + a) in C_set:
                            valid = False
                            break
                if valid:
                    if (2 * x) in C_set:
                        valid = False
                if valid:
                    C.append(x)
                    C_set.add(x)
                    remaining.remove(x)
                    remaining_set.discard(x)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # Try swap: remove one element from C, add one from remaining
            found_swap = False
            for i, a in enumerate(C):
                if time.time() - start_time > time_limit:
                    break
                if found_swap:
                    break
                # Temporarily remove a
                C_set.discard(a)
                C_no_a = C[:i] + C[i+1:]
                
                for x in list(remaining):
                    if time.time() - start_time > time_limit:
                        break
                    # Check if x can be added to C_no_a
                    valid = True
                    for b in C_no_a:
                        if (x - b) in C_set:
                            valid = False
                            break
                    if valid:
                        for b in C_no_a:
                            if (x + b) in C_set:
                                valid = False
                                break
                    if valid:
                        if (2 * x) in C_set:
                            valid = False
                    if valid:
                        # Do the swap
                        C = C_no_a + [x]
                        C_set.add(x)
                        remaining.remove(x)
                        remaining_set.discard(x)
                        found_swap = True
                        improved = True
                        break
                
                # Restore a if no swap was made
                if not found_swap:
                    C_set.add(a)
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return C

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Generate structural candidates
        S_odd = [x for x in S if x % 2 != 0]
        S_best_k = best_k_subset(S)
        
        M = max(S)
        m = min(S)
        S_upper = [x for x in S if x > M / 2.0]
        S_lower = [x for x in S if x < m / 2.0]
        
        S_greedy_max = greedy_max_abs(S)
        S_greedy_min = greedy_min_abs(S)
        S_greedy_pos = greedy_positive(S)
        S_greedy_neg = greedy_negative(S)
        
        # Additional structural candidates
        # Positive numbers only
        S_pos = [x for x in S if x > 0]
        # Negative numbers only
        S_neg = [x for x in S if x < 0]
        
        # Top half by value
        S_sorted_val = sorted(S)
        half = len(S_sorted_val) // 2
        S_top_half = S_sorted_val[half:]
        S_bottom_half = S_sorted_val[:half]
        
        # Magnitude threshold at various levels
        abs_vals = sorted(set(abs(x) for x in S))
        S_mag_75 = [x for x in S if abs(x) > abs_vals[len(abs_vals)*3//4] if len(abs_vals) > 1]
        S_mag_50 = [x for x in S if abs(x) > abs_vals[len(abs_vals)//2] if len(abs_vals) > 1]
        
        candidates = [S_odd, S_best_k, S_upper, S_lower, S_greedy_max, 
                      S_greedy_min, S_greedy_pos, S_greedy_neg,
                      S_pos, S_neg, S_top_half, S_bottom_half, S_mag_75, S_mag_50]
        
        # Filter to valid sum-free sets
        valid_candidates = []
        for cand in candidates:
            if is_sum_free(cand):
                valid_candidates.append(cand)
        
        if not valid_candidates:
            valid_candidates = [S_odd]
        
        best = max(valid_candidates, key=len)
        
        # Local search improvement
        if time.time() - start_time < DEADLINE - 5:
            improved = local_search(S, best, DEADLINE - 2)
            if len(improved) > len(best):
                best = improved
        
        # Final safety check
        if not is_sum_free(best):
            best = S_odd
        
        results.append(best)
    
    return results

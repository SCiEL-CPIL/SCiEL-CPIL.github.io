import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

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
                b = subset[j]
                c = a + b
                if c in s_set:
                    return False
        return True

    def can_add(C_set, x):
        """Check if x can be added to C (represented as set) to maintain sum-free property."""
        for y in C_set:
            if (x + y) in C_set:
                return False
            if (x - y) in C_set:
                return False
        if (2 * x) in C_set:
            return False
        return True

    def greedy_add(S, C_init):
        """Greedily add elements from S to C_init until no more can be added."""
        C = list(C_init)
        C_set = set(C)
        excluded = [x for x in S if x not in C_set]
        
        changed = True
        while changed:
            changed = False
            new_excluded = []
            for x in excluded:
                if can_add(C_set, x):
                    C.append(x)
                    C_set.add(x)
                    changed = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
        return C

    def local_search(S, C_init, max_iters=100, seed=42):
        """
        Local search to maximize sum-free subset size.
        Uses efficient 1-1 swaps and greedy saturation.
        """
        C = list(C_init)
        C_set = set(C)
        
        # First, saturate with greedy add
        C = greedy_add(S, C)
        C_set = set(C)
        
        best_C = C
        best_len = len(C)
        
        rng = random.Random(seed)
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            improved = False
            excluded = [x for x in S if x not in C_set]
            
            if not C or not excluded:
                break
                
            # Try 1-1 swaps: remove u from C, add v from excluded
            # This is more efficient than 1-2 swaps
            # We try random pairs until we find an improvement
            
            # Shuffle to get different order each iteration
            C_indices = list(range(len(C)))
            rng.shuffle(C_indices)
            excluded_shuffled = list(excluded)
            rng.shuffle(excluded_shuffled)
            
            found_improvement = False
            for u_idx in C_indices[:min(20, len(C_indices))]:  # Try up to 20 removals
                if time.time() > DEADLINE:
                    break
                    
                u = C[u_idx]
                C_temp = C[:u_idx] + C[u_idx+1:]
                C_temp_set = set(C_temp)
                
                # Try to add elements from excluded to C_temp greedily
                # but stop early if we can't improve
                added_count = 0
                C_temp_new = list(C_temp)
                C_temp_new_set = set(C_temp_new)
                
                for v in excluded_shuffled:
                    if can_add(C_temp_new_set, v):
                        C_temp_new.append(v)
                        C_temp_new_set.add(v)
                        added_count += 1
                        if added_count > 1:  # We already have net gain
                            break
                
                # Net gain: removed 1, added added_count
                if added_count >= 2:
                    C = C_temp_new
                    C_set = C_temp_new_set
                    improved = True
                    found_improvement = True
                    if len(C) > best_len:
                        best_C = C
                        best_len = len(C)
                    break
                
            if not found_improvement:
                # No improvement found, try perturbation
                if len(C) > 3:
                    # Remove 2 random elements to escape local optimum
                    indices_to_remove = rng.sample(range(len(C)), 2)
                    indices_to_remove.sort(reverse=True)
                    for idx in indices_to_remove:
                        C.pop(idx)
                    C_set = set(C)
                    
                    # Re-saturate
                    C = greedy_add(S, C)
                    C_set = set(C)
                    
                    if len(C) > best_len:
                        best_C = C
                        best_len = len(C)
                else:
                    break
            
        return best_C

    def solve_instance(S, time_budget):
        n = len(S)
        
        # Generate initial candidates
        candidates = []
        
        # 1. Odds
        S_odd = [x for x in S if x % 2 != 0]
        candidates.append(S_odd)
        
        # 2. Large Positive
        M = max(S)
        S_large_pos = [x for x in S if x > M / 2.0]
        candidates.append(S_large_pos)
        
        # 3. Large Negative
        m = min(S)
        S_large_neg = [x for x in S if x < m / 2.0]
        candidates.append(S_large_neg)
        
        # 4. Mod 3 = 1
        S_mod1 = [x for x in S if x % 3 == 1]
        candidates.append(S_mod1)
        
        # 5. Mod 3 = 2
        S_mod2 = [x for x in S if x % 3 == 2]
        candidates.append(S_mod2)
        
        # 6. Mod 4 = 1
        S_mod4_1 = [x for x in S if x % 4 == 1]
        candidates.append(S_mod4_1)
        
        # 7. Mod 4 = 3
        S_mod4_3 = [x for x in S if x % 4 == 3]
        candidates.append(S_mod4_3)
        
        # 8. Mod 5 = 1
        S_mod5_1 = [x for x in S if x % 5 == 1]
        candidates.append(S_mod5_1)
        
        # 9. Mod 5 = 3
        S_mod5_3 = [x for x in S if x % 5 == 3]
        candidates.append(S_mod5_3)
        
        # 10. Greedy by abs desc
        S_sorted_abs = sorted(S, key=lambda x: abs(x), reverse=True)
        C_greedy_abs = []
        C_set = set()
        for x in S_sorted_abs:
            if can_add(C_set, x):
                C_greedy_abs.append(x)
                C_set.add(x)
        candidates.append(C_greedy_abs)
        
        # 11. Greedy by abs asc
        S_sorted_abs_asc = sorted(S, key=lambda x: abs(x))
        C_greedy_abs_asc = []
        C_set = set()
        for x in S_sorted_abs_asc:
            if can_add(C_set, x):
                C_greedy_abs_asc.append(x)
                C_set.add(x)
        candidates.append(C_greedy_abs_asc)

        # 12-14. Random Greedy with different seeds
        for rs in [123, 456, 789]:
            rng_init = random.Random(rs)
            S_shuffled = list(S)
            rng_init.shuffle(S_shuffled)
            C_greedy_rand = []
            C_set = set()
            for x in S_shuffled:
                if can_add(C_set, x):
                    C_greedy_rand.append(x)
                    C_set.add(x)
            candidates.append(C_greedy_rand)

        # Evaluate candidates and pick the best starting point for local search
        best_start = []
        best_start_len = 0
        for C in candidates:
            if len(C) > best_start_len:
                if is_sum_free(C):
                    best_start = C
                    best_start_len = len(C)
        
        # Run local search from multiple good starts to diversify
        final_best = best_start
        final_best_len = best_start_len
        
        # Time management: split among restarts
        # Use more restarts for larger instances
        if n <= 60:
            num_restarts = 5
        elif n <= 120:
            num_restarts = 8
        else:
            num_restarts = 12
        
        time_per_restart = time_budget / num_restarts
        
        # Sort candidates by size descending, take top ones for restarts
        candidates_sorted = sorted(candidates, key=len, reverse=True)
        # Remove duplicates by checking content
        unique_candidates = []
        seen = set()
        for C in candidates_sorted:
            C_tuple = tuple(sorted(C))
            if C_tuple not in seen:
                seen.add(C_tuple)
                unique_candidates.append(C)
        
        for i in range(num_restarts):
            if time.time() > start_time + time_budget:
                break
            
            # Pick a different structural candidate for each restart
            if i < len(unique_candidates):
                start_C = unique_candidates[i]
            else:
                start_C = best_start
            
            # Use different random seed for each restart
            improved_C = local_search(S, start_C, max_iters=50, seed=1000 + i)
            
            if len(improved_C) > final_best_len:
                final_best = improved_C
                final_best_len = len(improved_C)
                
        return final_best

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time: give more to larger instances
    time_alloc = [20, 40, 80]
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        C = solve_instance(S, time_alloc[i])
        
        # Final safety check
        if not is_sum_free(C):
            C = [x for x in S if x % 2 != 0]
        
        results.append(C)
    
    return results

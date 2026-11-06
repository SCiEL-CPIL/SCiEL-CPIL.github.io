import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 4096
    best_set = []
    best_len = 0
    
    def compute_pair_sums(S):
        """Compute all sums S[i] + S[j] for i < j."""
        ps = set()
        k = len(S)
        for i in range(k):
            si = S[i]
            for j in range(i + 1, k):
                ps.add(si + S[j])
        return ps

    def compute_all_triple_sums(S):
        """Compute all sums S[i] + S[j] + S[l] for i <= j <= l."""
        sums = set()
        k = len(S)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                for l in range(j, k):
                    sums.add(si + sj + S[l])
        return sums

    def can_add(x, S_list, S_set, existing_sums, pair_sums):
        """
        Check if x can be added to S_list.
        Returns (valid, new_sums_set)
        """
        # New sums involving x:
        # 1. 3*x
        # 2. 2*x + y for y in S
        # 3. x + 2*y for y in S
        # 4. x + y + z for y < z in S  => x + (y+z) where y+z in pair_sums
        
        new_sums = set()
        
        # 1
        new_sums.add(3 * x)
        
        # 2 & 3
        for y in S_list:
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
            
        # 4
        for p in pair_sums:
            new_sums.add(x + p)
            
        # Check collision with existing
        if new_sums & existing_sums:
            return False, None
            
        # Check internal uniqueness
        # Expected count: 1 + k + k + k(k-1)/2
        k = len(S_list)
        expected = 1 + k + k + k * (k - 1) // 2
        if len(new_sums) != expected:
            return False, None
            
        return True, new_sums

    def greedy_build(candidates, S_init):
        """
        Greedily add elements from candidates to S_init.
        Returns (S_list, existing_sums, pair_sums)
        """
        S_list = list(S_init)
        S_set = set(S_list)
        existing_sums = compute_all_triple_sums(S_list)
        pair_sums = compute_pair_sums(S_list)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            valid, new_sums = can_add(x, S_list, S_set, existing_sums, pair_sums)
            if valid:
                S_list.append(x)
                S_set.add(x)
                existing_sums.update(new_sums)
                
                # Update pair_sums: add x + y for all y in old S_list
                # Note: S_list has been updated, so we need to be careful.
                # The new pairs are (x, y) for y in S_list[:-1]
                for y in S_list[:-1]:
                    pair_sums.add(x + y)
                    
        return S_list, existing_sums, pair_sums

    # Strategy 1: Power of 3 baseline
    power3_set = [3**i for i in range(8) if 3**i <= n]
    if power3_set:
        best_set = list(power3_set)
        best_len = len(power3_set)

    # Strategy 2: Randomized Greedy
    num_runs = 100
    for run in range(num_runs):
        if time.time() > deadline:
            break
        
        random.seed(run * 100 + 42)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S_list, existing_sums, pair_sums = greedy_build(candidates, [])
        
        if len(S_list) > best_len:
            best_set = list(S_list)
            best_len = len(S_list)
            
    # Strategy 3: Local Search on the global best
    if best_set:
        S_list = list(best_set)
        S_set = set(S_list)
        existing_sums = compute_all_triple_sums(S_list)
        pair_sums = compute_pair_sums(S_list)
        
        # Phase 3a: Greedy Fill
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            valid, new_sums = can_add(x, S_list, S_set, existing_sums, pair_sums)
            if valid:
                S_list.append(x)
                S_set.add(x)
                existing_sums.update(new_sums)
                for y in S_list[:-1]:
                    pair_sums.add(x + y)
        
        if len(S_list) > best_len:
            best_set = list(S_list)
            best_len = len(S_list)
            S_set = set(S_list)
            existing_sums = compute_all_triple_sums(S_list)
            pair_sums = compute_pair_sums(S_list)

        # Phase 3b: Local Search (Swaps)
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                    
                if not S_list:
                    break
                
                idx_to_remove = random.randint(0, len(S_list) - 1)
                removed_val = S_list[idx_to_remove]
                
                S_temp = S_list.copy()
                S_temp.pop(idx_to_remove)
                S_temp_set = set(S_temp)
                
                sums_temp = compute_all_triple_sums(S_temp)
                pair_sums_temp = compute_pair_sums(S_temp)
                
                found_better = False
                candidates_to_try = random.sample([x for x in range(1, n+1) if x not in S_temp_set], min(50, n - len(S_temp)))
                
                for x in candidates_to_try:
                    valid, new_sums = can_add(x, S_temp, S_temp_set, sums_temp, pair_sums_temp)
                    if valid:
                        S_temp.append(x)
                        S_temp_set.add(x)
                        sums_temp.update(new_sums)
                        for y in S_temp[:-1]:
                            pair_sums_temp.add(x + y)
                        
                        for x2 in range(1, n + 1):
                            if time.time() > deadline:
                                break
                            if x2 in S_temp_set:
                                continue
                            valid2, new_sums2 = can_add(x2, S_temp, S_temp_set, sums_temp, pair_sums_temp)
                            if valid2:
                                S_temp.append(x2)
                                S_temp_set.add(x2)
                                sums_temp.update(new_sums2)
                                for y in S_temp[:-1]:
                                    pair_sums_temp.add(x2 + y)
                                
                        if len(S_temp) > best_len:
                            best_set = list(S_temp)
                            best_len = len(S_temp)
                            S_list = list(S_temp)
                            S_set = set(S_list)
                            existing_sums = sums_temp
                            pair_sums = pair_sums_temp
                            improved = True
                            found_better = True
                            break
                        
                if found_better:
                    break

            if not improved:
                break

    if best_set and not all(1 <= x <= n for x in best_set):
        best_set = [3**i for i in range(8) if 3**i <= n]
        
    return best_set

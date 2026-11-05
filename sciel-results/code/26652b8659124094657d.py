# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n_val = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = [d for d in range(1, n_val + 1) if n_val % d == 0]
    divisors.sort()
    n_divs = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute prefix sums
    prefix = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    total_sum = prefix[n_divs]
    
    # k=1
    if target in divisor_set:
        return [target]
        
    # k=2: O(N) using set
    for d in divisors:
        complement = target - d
        if complement in divisor_set and complement > d:
            return [d, complement]
            
    # k=3: O(N^2) using set
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            c = target - a - b
            if c in divisor_set and c > b:
                return [a, b, c]
    
    # For k >= 4: Use DP to find minimum k
    # dp[k][s] = True if we can form sum s using exactly k distinct divisors
    # We track the actual selection via parent pointers
    
    # For each k from 4 to n_divs, check feasibility
    # Use bitset DP for efficiency
    
    for k in range(4, n_divs + 1):
        if time.time() > deadline:
            break
        
        # Quick feasibility: min sum of k smallest, max sum of k largest
        min_sum_k = prefix[k]  # sum of first k divisors
        max_sum_k = total_sum - prefix[n_divs - k]  # sum of last k divisors
        if target < min_sum_k or target > max_sum_k:
            continue
        
        # DP: reachable sums using exactly k divisors
        # Use a set of reachable sums, tracking which divisors were used
        # To save memory, use: dp[i][j] = set of sums achievable with j divisors from first i divisors
        # But this is O(n_divs * k * target) which might be too slow
        
        # Alternative: use a 2D array dp[j][s] = bitmask or just boolean
        # dp[j][s] = True if sum s is achievable with exactly j divisors
        # We need to track the actual divisors, so use parent pointers
        
        # For small k, use iterative deepening with pruning
        # For larger k, the answer is likely close to n_divs
        
        # Let's use a different approach: since k is what we're minimizing,
        # and we've checked k=1,2,3, try k=4,5,6 with meet-in-the-middle
        
        if k == 4:
            # Meet in the middle: pairs
            pair_sums = {}
            for i in range(n_divs):
                a = divisors[i]
                if a >= target:
                    break
                for j in range(i + 1, n_divs):
                    b = divisors[j]
                    s = a + b
                    if s >= target:
                        break
                    if s not in pair_sums:
                        pair_sums[s] = []
                    pair_sums[s].append((i, j))
            
            for s1, pairs1 in pair_sums.items():
                s2 = target - s1
                if s2 not in pair_sums:
                    continue
                if s2 < s1:
                    continue
                pairs2 = pair_sums[s2]
                
                if s1 == s2:
                    for p1_idx in range(len(pairs1)):
                        i1, j1 = pairs1[p1_idx]
                        for p2_idx in range(p1_idx + 1, len(pairs1)):
                            i2, j2 = pairs1[p2_idx]
                            if len({i1, j1, i2, j2}) == 4:
                                return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
                else:
                    for i1, j1 in pairs1:
                        for i2, j2 in pairs2:
                            if len({i1, j1, i2, j2}) == 4:
                                return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
                        
        elif k == 5:
            # Pairs + triples
            pair_sums = {}
            for i in range(n_divs):
                a = divisors[i]
                if a >= target:
                    break
                for j in range(i + 1, n_divs):
                    b = divisors[j]
                    s = a + b
                    if s >= target:
                        break
                    if s not in pair_sums:
                        pair_sums[s] = []
                    pair_sums[s].append((i, j))
            
            triple_sums = {}
            for i in range(n_divs):
                a = divisors[i]
                if a >= target:
                    break
                for j in range(i + 1, n_divs):
                    b = divisors[j]
                    if a + b >= target:
                        break
                    for kk in range(j + 1, n_divs):
                        c = divisors[kk]
                        s = a + b + c
                        if s >= target:
                            break
                        if s not in triple_sums:
                            triple_sums[s] = []
                        triple_sums[s].append((i, j, kk))
            
            for s_t, triples in triple_sums.items():
                s_p = target - s_t
                if s_p not in pair_sums:
                    continue
                pairs = pair_sums[s_p]
                for i1, j1, k1 in triples:
                    used = {i1, j1, k1}
                    for i2, j2 in pairs:
                        if i2 not in used and j2 not in used:
                            return [divisors[i1], divisors[j1], divisors[k1], divisors[i2], divisors[j2]]
                            
        elif k == 6:
            # Triples + triples
            triple_sums = {}
            for i in range(n_divs):
                a = divisors[i]
                if a >= target:
                    break
                for j in range(i + 1, n_divs):
                    b = divisors[j]
                    if a + b >= target:
                        break
                    for kk in range(j + 1, n_divs):
                        c = divisors[kk]
                        s = a + b + c
                        if s >= target:
                            break
                        if s not in triple_sums:
                            triple_sums[s] = []
                        triple_sums[s].append((i, j, kk))
            
            for s1, triples1 in triple_sums.items():
                s2 = target - s1
                if s2 not in triple_sums:
                    continue
                if s2 < s1:
                    continue
                triples2 = triple_sums[s2]
                
                if s1 == s2:
                    for t1_idx in range(len(triples1)):
                        i1, j1, k1 = triples1[t1_idx]
                        used1 = {i1, j1, k1}
                        for t2_idx in range(t1_idx + 1, len(triples1)):
                            i2, j2, k2 = triples1[t2_idx]
                            if i2 not in used1 and j2 not in used1 and k2 not in used1:
                                return [divisors[i1], divisors[j1], divisors[k1], divisors[i2], divisors[j2], divisors[k2]]
                else:
                    for i1, j1, k1 in triples1:
                        used1 = {i1, j1, k1}
                        for i2, j2, k2 in triples2:
                            if i2 not in used1 and j2 not in used1 and k2 not in used1:
                                return [divisors[i1], divisors[j1], divisors[k1], divisors[i2], divisors[j2], divisors[k2]]
        else:
            # k >= 7: Backtracking with strong pruning
            def find_k_divisors(k, start_idx, remaining_sum):
                if k == 0:
                    return [] if remaining_sum == 0 else None
                
                if start_idx >= n_divs:
                    return None
                    
                if n_divs - start_idx < k:
                    return None
                    
                min_possible = prefix[start_idx + k] - prefix[start_idx]
                if remaining_sum < min_possible:
                    return None
                    
                max_possible = total_sum - prefix[n_divs - k]
                if remaining_sum > max_possible:
                    return None
                    
                for i in range(start_idx, n_divs):
                    d = divisors[i]
                    
                    new_remaining = remaining_sum - d
                    if new_remaining < 0:
                        break
                    
                    if i + 1 + (k - 1) <= n_divs:
                        min_next = prefix[i + 1 + (k - 1)] - prefix[i + 1]
                        if new_remaining < min_next:
                            break
                    else:
                        continue

                    if i + 1 <= n_divs - (k - 1):
                        max_next = total_sum - prefix[n_divs - (k - 1)]
                    else:
                        max_next = total_sum - prefix[i + 1]
                        
                    if new_remaining > max_next:
                        pass

                    res = find_k_divisors(k - 1, i + 1, new_remaining)
                    if res is not None:
                        return [d] + res
                            
                return None
            
            res = find_k_divisors(k, 0, target)
            if res is not None:
                return res
            
    # Fallback: Greedy
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder and d not in chosen:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
                
    return chosen
# EVOLVE-BLOCK-END

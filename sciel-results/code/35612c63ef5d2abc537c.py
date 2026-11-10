"""Rewritten program: systematic search for minimal maximum factor in 12! decomposition."""
import time
from math import factorial
from functools import lru_cache
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = factorial(12)  # 479001600
    
    # We need to find a set of 1..12 distinct integers, all > 12,
    # whose product is exactly N, minimizing the largest factor.
    
    # Strategy: 
    # 1. Find all divisors of N that are > 12
    # 2. Among subsets of these divisors (size 1..12), find those whose product is N
    # 3. Minimize the maximum element in such a subset
    
    # First, get all divisors of N
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            other = N // i
            if other != i and other > 12:
                divisors.append(other)
    divisors.sort()
    
    # We want to minimize the largest factor. So we can try increasing upper bounds.
    # For a given max factor M, check if there exists a subset of divisors in (12, M] 
    # whose product is N.
    
    # But the search space of subsets is huge. Let's think differently.
    
    # Key insight: The number of factors is at most 12. Since all factors > 12,
    # and product is 12! ≈ 4.79e8, the factors can't be too small.
    
    # Let's use a different approach: 
    # We know the prime factorization of 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # We need to partition these prime powers into 1..12 groups, each group's product > 12,
    # minimizing the maximum group product.
    
    # This is a bin-packing-like problem. Let's use DFS with pruning.
    
    # Prime factors with exponents:
    # 2^10, 3^5, 5^2, 7^1, 11^1
    
    # Total number of divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    # So there are 792 divisors of 12!. Those > 12 are candidates.
    
    # Let's enumerate all divisors > 12 and their prime exponent vectors.
    divs_info = []  # list of (value, e2, e3, e5, e7, e11)
    for i in range(13, int(N**0.5) + 1):
        if N % i == 0:
            # Factor i
            val = i
            e2 = e3 = e5 = e7 = e11 = 0
            while val % 2 == 0:
                e2 += 1
                val //= 2
            while val % 3 == 0:
                e3 += 1
                val //= 3
            while val % 5 == 0:
                e5 += 1
                val //= 5
            while val % 7 == 0:
                e7 += 1
                val //= 7
            while val % 11 == 0:
                e11 += 1
                val //= 11
            if val != 1:
                continue  # shouldn't happen for divisors of 12!
            divs_info.append((i, e2, e3, e5, e7, e11))
    
    # Add the large divisors
    for i in range(int(N**0.5), 1, -1):
        if N % i == 0:
            other = N // i
            if other < i:
                break
            if i > 12:
                val = i
                e2 = e3 = e5 = e7 = e11 = 0
                while val % 2 == 0:
                    e2 += 1
                    val //= 2
                while val % 3 == 0:
                    e3 += 1
                    val //= 3
                while val % 5 == 0:
                    e5 += 1
                    val //= 5
                while val % 7 == 0:
                    e7 += 1
                    val //= 7
                while val % 11 == 0:
                    e11 += 1
                    val //= 11
                if val != 1:
                    continue
                divs_info.append((i, e2, e3, e5, e7, e11))
    
    # Remove duplicates and sort
    seen = set()
    unique_divs = []
    for item in divs_info:
        if item not in seen:
            seen.add(item)
            unique_divs.append(item)
    unique_divs.sort(key=lambda x: x[0])
    
    # Now, we want to find a subset S of unique_divs (size 1..12) such that:
    # - All distinct (guaranteed since we pick from a set)
    # - All > 12 (guaranteed)
    # - Product = N
    # - Minimize max(S)
    
    # The sum of exponents in the subset must equal (10, 5, 2, 1, 1).
    
    # This is a multi-dimensional subset sum problem. With 792 items and 5 dimensions,
    # it's challenging but we can use DFS with strong pruning.
    
    # Alternative: Since we want to minimize the max, let's try binary search on the answer.
    # For a given bound B, check if there's a valid subset with all elements <= B.
    
    # But first, let's try a direct approach: enumerate subsets by size, starting from
    # smaller max values.
    
    # Actually, let's think about what's feasible. The product is ~4.8e8.
    # If we use k factors, the geometric mean is (4.8e8)^(1/k).
    # k=1: 4.8e8
    # k=2: ~21886
    # k=3: ~7830
    # k=4: ~2635
    # k=5: ~1363
    # k=6: ~856
    # k=7: ~612
    # k=8: ~475
    # k=9: ~388
    # k=10: ~328
    # k=11: ~285
    # k=12: ~253
    
    # So the max factor should be at least around 253 for k=12.
    
    # Let's use a DFS that builds a subset, tracking the product (as exponent vector)
    # and the current max. We prune if product exceeds N in any prime dimension,
    # or if the number of factors exceeds 12.
    
    # To minimize the max, we can do a best-first search or iterative deepening on the max.
    
    # Let's try: for max_bound from 253 upward, check if a valid subset exists.
    
    # But checking existence is NP-hard in general. However, the constraint structure
    # (5-dimensional, small exponents) makes it tractable with good pruning.
    
    # Let's use a recursive DFS with memoization on the remaining exponent vector.
    
    # State: (idx, e2_rem, e3_rem, e5_rem, e7_rem, e11_rem, count, current_max)
    # But this state space might be too large.
    
    # Better: Since we want to minimize the max, let's sort divisors and do a DFS
    # that tries to pick divisors in increasing order, ensuring the last picked
    # (largest) is minimized.
    
    # Approach: 
    # 1. Sort all candidate divisors.
    # 2. Use DFS to find subsets whose exponent sum is (10,5,2,1,1).
    # 3. Among all valid subsets, track the one with minimum max element.
    
    # To make this efficient, we can use the fact that the exponents are small.
    # The number of distinct exponent vectors is at most (11*6*3*2*2) = 792.
    
    # Let's use a DP approach:
    # dp[e2][e3][e5][e7][e11] = minimum possible max factor for a subset that achieves
    # this exponent sum, or infinity if not achievable.
    
    # But we also need to ensure distinctness, which complicates things.
    
    # Alternative: Since the number of factors is at most 12, and we need distinct factors,
    # let's just do a careful DFS with pruning.
    
    # Let's try a different tactic: enumerate all subsets of size up to 12 from the
    # candidate divisors, but with heavy pruning based on the exponent sums.
    
    # Actually, the number of candidate divisors > 12 is at most 792 - (number <= 12).
    # Divisors of 12! that are <= 12: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.
    # So 12 divisors are <= 12, leaving 780 candidates.
    
    # C(780, 12) is astronomical. We need much better pruning.
    
    # Let's use the following: 
    # - Sort candidates by value.
    # - Use DFS, picking candidates in order.
    # - Prune if the remaining candidates can't possibly complete the exponent sum.
    # - Track the best (minimum max) found.
    
    # For the pruning, we can precompute for each suffix of the sorted list,
    # the maximum possible contribution to each exponent.
    
    # Let's implement this.
    
    candidates = unique_divs  # list of (val, e2, e3, e5, e7, e11)
    n_cands = len(candidates)
    
    # Precompute suffix max exponents
    suffix_max = [[0]*5 for _ in range(n_cands + 1)]
    for i in range(n_cands - 1, -1, -1):
        for j in range(5):
            suffix_max[i][j] = max(candidates[i][1 + j], suffix_max[i + 1][j])
    
    # Actually, that's not quite right. We need the sum of exponents, not max.
    # Let's compute suffix sums.
    suffix_sum = [[0]*5 for _ in range(n_cands + 1)]
    for i in range(n_cands - 1, -1, -1):
        for j in range(5):
            suffix_sum[i][j] = candidates[i][1 + j] + suffix_sum[i + 1][j]
    
    target = (10, 5, 2, 1, 1)
    
    best_max = float('inf')
    best_subset = None
    
    def dfs(idx, count, e2, e3, e5, e7, e11, current_max, chosen):
        nonlocal best_max, best_subset
        
        if time.time() > deadline:
            return
        
        # Check if we've achieved the target
        if e2 == target[0] and e3 == target[1] and e5 == target[2] and e7 == target[3] and e11 == target[4]:
            if count >= 1 and count <= 12:
                if current_max < best_max:
                    best_max = current_max
                    best_subset = chosen[:]
            return
        
        # If we've exceeded any exponent, prune
        if e2 > target[0] or e3 > target[1] or e5 > target[2] or e7 > target[3] or e11 > target[4]:
            return
        
        # If we've used 12 factors, prune
        if count >= 12:
            return
        
        # Pruning: check if remaining candidates can fill the gap
        rem_e2 = target[0] - e2
        rem_e3 = target[1] - e3
        rem_e5 = target[2] - e5
        rem_e7 = target[3] - e7
        rem_e11 = target[4] - e11
        
        if idx >= n_cands:
            return
        
        # Check if suffix can provide the remaining exponents
        if suffix_sum[idx][0] < rem_e2 or suffix_sum[idx][1] < rem_e3 or \
           suffix_sum[idx][2] < rem_e5 or suffix_sum[idx][3] < rem_e7 or \
           suffix_sum[idx][4] < rem_e11:
            return
        
        # Try skipping candidates[idx]
        dfs(idx + 1, count, e2, e3, e5, e7, e11, current_max, chosen)
        
        # Try including candidates[idx]
        val, ce2, ce3, ce5, ce7, ce11 = candidates[idx]
        new_e2 = e2 + ce2
        new_e3 = e3 + ce3
        new_e5 = e5 + ce5
        new_e7 = e7 + ce7
        new_e11 = e11 + ce11
        
        if new_e2 <= target[0] and new_e3 <= target[1] and new_e5 <= target[2] and \
           new_e7 <= target[3] and new_e11 <= target[4]:
            new_max = max(current_max, val)
            if new_max < best_max:  # Only explore if this could improve
                chosen.append(val)
                dfs(idx + 1, count + 1, new_e2, new_e3, new_e5, new_e7, new_e11, new_max, chosen)
                chosen.pop()
    
    dfs(0, 0, 0, 0, 0, 0, 0, 0, [])
    
    if best_subset is not None:
        return best_subset
    else:
        # Fallback: return [12!]
        return [N]

BET: {mechanism: "Systematic DFS over divisor exponent vectors with suffix-sum pruning and best-first max tracking finds the true optimal subset, beating the trivial [12!] baseline by orders of magnitude", q: 0.95, delta: [1000000, 479001599]}

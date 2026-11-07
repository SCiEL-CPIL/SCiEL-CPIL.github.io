import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    n = 256
    
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # For each prime, the standalone value is the largest power p^k <= n
    standalone_val = {}
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        standalone_val[p] = val
    
    # Split primes into small (p^2 <= 256) and large (p^2 > 256)
    # Small primes: 2, 3, 5, 7, 11, 13 (since 13^2 = 169 <= 256, 17^2 = 289 > 256)
    small_primes = [p for p in primes if p * p <= n]
    large_primes = [p for p in primes if p * p > n]
    
    small_set = set(small_primes)
    num_small = len(small_primes)
    small_idx = {p: i for i, p in enumerate(small_primes)}
    
    # Precompute best value for each subset of small primes
    # best_small[mask] = max product <= 256 using only primes in mask (each prime used at most once in the product's prime factorization, but powers allowed)
    # Actually, for a subset S of small primes, we want max n <= 256 such that prime factors of n are subset of S
    # But we also need to track which primes are "used" - actually for the DP we need:
    # For a given set of small primes S, what's the max value we can get?
    # This is: max over all n <= 256 where all prime factors of n are in S
    
    # Precompute prime factors for each number 1..256
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not prime_factors[i]:
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    # For each mask of small primes, compute best value
    all_small_mask = (1 << num_small) - 1
    best_small = [0] * (1 << num_small)
    
    for mask in range(1, 1 << num_small):
        # Get the set of primes in this mask
        mask_primes = set()
        for i in range(num_small):
            if mask & (1 << i):
                mask_primes.add(small_primes[i])
        
        # Find max n <= 256 where all prime factors are in mask_primes
        best = 0
        for val in range(n, 0, -1):
            if val == 1:
                best = 1
                break
            pf = prime_factors[val]
            if all(p in mask_primes for p in pf):
                best = val
                break
        best_small[mask] = best
    
    # DP for large primes
    # dp[mask] = max sum contribution from processing some large primes, with small primes in mask used
    # Initialize
    dp = [-1] * (1 << num_small)
    dp[0] = 0
    
    for p in large_primes:
        new_dp = [-1] * (1 << num_small)
        for mask in range(1 << num_small):
            if dp[mask] < 0:
                continue
            # Option 1: don't use p at all (standalone)
            # p is large, so standalone_val[p] = p (since p^2 > 256)
            # But wait, we need to add p to the result. The DP tracks the sum.
            # If we don't merge p with any small primes, we just add standalone_val[p] = p
            new_dp[mask] = max(new_dp[mask], dp[mask] + p)
            
            # Option 2: merge p with some subset of available small primes
            # Available small primes: those not in mask
            avail_mask = all_small_mask ^ mask
            # Try all submasks of avail_mask
            sub = avail_mask
            while sub:
                # If we use small primes in sub with p, the value is best_small[sub | (1 << small_idx[p])] if p were small
                # But p is large, so we can't use best_small directly.
                # For large prime p and subset S of small primes, the best merged value is:
                # max n <= 256 where prime factors of n are in S ∪ {p}
                # This equals: max over k >= 1 of p^k * m where m's prime factors are in S and p^k * m <= 256
                # Since p is large (p >= 17), p^2 > 256, so k can only be 1.
                # So merged value = p * max m <= 256/p where m's prime factors are in S
                # max m <= 256/p with prime factors in S = best value from small primes S up to 256//p
                
                # Compute max m <= 256//p with prime factors in sub
                limit = n // p
                if limit >= 1:
                    # Find max m <= limit with all prime factors in sub
                    sub_primes = set()
                    for i in range(num_small):
                        if sub & (1 << i):
                            sub_primes.add(small_primes[i])
                    
                    max_m = 1  # at least 1
                    for m in range(limit, 0, -1):
                        if m == 1:
                            max_m = 1
                            break
                        pf = prime_factors[m]
                        if all(q in sub_primes for q in pf):
                            max_m = m
                            break
                    
                    merged_val = p * max_m
                    new_mask = mask | sub
                    gain = merged_val - p  # gain over standalone
                    # Actually, we're replacing standalone p with merged_val
                    # So the contribution is merged_val instead of p
                    if new_dp[new_mask] < dp[mask] + merged_val:
                        new_dp[new_mask] = dp[mask] + merged_val
                sub = (sub - 1) & avail_mask
        dp = new_dp
    
    # Final answer: max over all masks of dp[mask] + best_small[all_small_mask ^ mask] + 1
    # Wait, dp already includes the contributions from large primes.
    # The remaining small primes (not used by any large prime) can be combined.
    # best_small[all_small_mask ^ mask] gives the best value using those remaining small primes.
    # But we need to be careful: dp[mask] already accounts for the large primes and which small primes they used.
    # The remaining small primes can form one or more numbers.
    # Actually, best_small[remaining_mask] is the max single number using those primes.
    # But we could also split them into multiple numbers.
    # For small primes, the optimal is to find the max sum of pairwise coprime numbers using those primes.
    # Since small primes are few (6), we can precompute the best sum for each subset.
    
    # Precompute best sum for each subset of small primes (using DP over subsets)
    # best_sum_small[mask] = max sum of pairwise coprime numbers > 1 using only primes in mask
    best_sum_small = [0] * (1 << num_small)
    
    # For each subset, try all ways to partition it into groups, each group forming one number
    # Since the number of small primes is small, we can use a DP where:
    # best_sum_small[mask] = max over all submasks sub of mask (containing the lowest set bit):
    #   best_small[sub] + best_sum_small[mask ^ sub]
    
    for mask in range(1, 1 << num_small):
        # Find lowest set bit
        low_bit = mask & (-mask)
        # Iterate over all submasks of mask that contain low_bit
        sub = mask
        while sub:
            if sub & low_bit:
                val = best_small[sub] + best_sum_small[mask ^ sub]
                if val > best_sum_small[mask]:
                    best_sum_small[mask] = val
            sub = (sub - 1) & mask
    
    # Now compute final answer
    best_total = 1  # include 1
    for mask in range(1 << num_small):
        if dp[mask] < 0:
            continue
        remaining = all_small_mask ^ mask
        total = dp[mask] + best_sum_small[remaining] + 1
        if total > best_total:
            best_total = total
    
    # Now we need to reconstruct the actual list of numbers
    # This is more complex. Let's use a different approach:
    # Use branch and bound on the full problem with the insights from above.
    
    # Actually, let's just do a clean branch and bound with good pruning.
    
    # For each prime, standalone value
    # For pairs/triples of primes, merged values
    
    # Let's use a simpler approach: enumerate all valid sets using backtracking
    # with the constraint that each number's prime factors are distinct across numbers.
    
    # Generate all candidate numbers 1..256 with their prime factor sets
    candidates = []
    for val in range(256, 0, -1):
        pf = set(prime_factors[val])
        candidates.append((val, pf))
    
    # Sort by value descending for better pruning
    candidates.sort(key=lambda x: -x[0])
    
    # Branch and bound
    best_sum = 0
    best_set = []
    
    def backtrack(idx, used_primes, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Pruning: even if we take all remaining, can we beat best?
        remaining_max = sum(c[0] for c in candidates[idx:])
        if current_sum + remaining_max <= best_sum:
            return
        
        if idx == len(candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
        
        val, pf = candidates[idx]
        
        # Option 1: skip this candidate
        backtrack(idx + 1, used_primes, current_sum, current_set)
        
        # Option 2: take this candidate if no prime conflict
        if not used_primes.intersection(pf):
            new_used = used_primes | pf
            current_set.append(val)
            backtrack(idx + 1, new_used, current_sum + val, current_set)
            current_set.pop()
    
    backtrack(0, set(), 0, [])
    
    if best_set:
        result = [int(x) for x in best_set]
    else:
        # Fallback
        result = [1] + [standalone_val[p] for p in primes]
    
    # Verify
    def is_valid(lst):
        if not lst:
            return False
        if len(lst) != len(set(lst)):
            return False
        for x in lst:
            if not (1 <= x <= 256):
                return False
            if not isinstance(x, int) or isinstance(x, bool):
                return False
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    if not is_valid(result):
        result = [1] + [standalone_val[p] for p in primes]
    
    return result

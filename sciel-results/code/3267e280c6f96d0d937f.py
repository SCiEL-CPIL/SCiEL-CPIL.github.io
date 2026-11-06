"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, gcd
from functools import lru_cache
from itertools import combinations

def construct():
    """
    Find 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    
    Strategy:
    1. Compute 12! = 479001600.
    2. We want to partition the prime factorization of 12! into k groups (1 <= k <= 12),
       such that each group's product is > 12 and all products are distinct.
    3. We want to minimize the maximum product.
    
    Approach:
    - Generate all divisors of 12! that are > 12.
    - Use backtracking to find a set of distinct divisors > 12 whose product is 12!,
      minimizing the maximum element.
    
    Better approach for efficiency:
    - Since 12! = 2^10 * 3^5 * 5^2 * 7 * 11, we can work with exponent vectors.
    - Use DFS with pruning: at each step, pick the next factor from remaining divisors,
      ensuring we can still complete the product.
    
    Even better:
    - We can enumerate possible factor counts k from 1 to 12.
    - For each k, try to find k distinct factors > 12 with product 12! and min max.
    - Start with small k (fewer factors means larger average factor, so likely larger max).
    - Actually, more factors means smaller average, so we should try larger k first
      to get smaller max factors.
    
    Let's think about this more carefully:
    - 12! = 479001600
    - If we use 1 factor: [479001600], max = 479001600
    - If we use 2 factors: we need two distinct factors > 12 with product 479001600
      To minimize the max, we want them as close as possible to sqrt(479001600) ≈ 21886
    - If we use more factors, we can get smaller max values.
    
    Strategy:
    - For k from 12 down to 1 (try more factors first for smaller max):
      - Try to find k distinct divisors > 12 of 12! whose product is 12!
      - Among all valid solutions for this k, find the one with minimum max
      - Keep track of the best (minimum max) across all k
    
    For efficiency, use backtracking with:
    - Precomputed list of all divisors > 12 of 12!
    - Sort divisors
    - At each step, pick a divisor that divides the remaining product
    - Prune if current max already exceeds best found
    
    Let's implement this.
    """
    target = factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    # Generate all divisors of target
    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return sorted(divs)
    
    all_divs = get_divisors(target)
    # Filter: must be > 12
    valid_divs = [d for d in all_divs if d > 12]
    
    # For backtracking, we need to efficiently check if a divisor divides remaining
    # and find candidates.
    
    best_result = None
    best_max = target + 1  # Initial best is the single factor case
    
    # Try k from 12 down to 1
    # But we can also be smarter: for a given k, the minimum possible max is at least
    # (target)^(1/k) * some factor. Let's just try all k.
    
    for k in range(12, 0, -1):
        if time.time() > deadline:
            break
        
        # For k factors, each must be > 12, so product of k factors > 12^k
        # We need 12^k <= target, i.e., 12^k <= 479001600
        # 12^1=12, 12^2=144, 12^3=1728, 12^4=20736, 12^5=248832, 12^6=2985984,
        # 12^7=35831808, 12^8=429981696, 12^9=5159780352 > 479001600
        # So k can be at most 8 (since 12^9 > target)
        # Actually wait: the factors just need to be > 12, not >= 12.
        # But the product of k factors each > 12 is > 12^k.
        # So we need 12^k < target (strictly, since factors are integers > 12, min factor is 13)
        # 13^k <= target
        # 13^1=13, 13^2=169, 13^3=2197, 13^4=28561, 13^5=371293, 13^6=4826809,
        # 13^7=62748517, 13^8=815730721 > 479001600
        # So k can be at most 7.
        
        if k > 7:
            continue
        
        # Backtracking search for k distinct factors > 12 with product = target
        # Minimize the max
        
        # We'll use DFS. State: (remaining_product, min_allowed, used_set, current_factors, current_max)
        # To ensure distinctness and that we pick factors in a way that avoids duplicates,
        # we can enforce that factors are picked in increasing order.
        
        # Precompute: for each divisor d > 12, check if d divides target (they all do by construction)
        
        def dfs(rem, min_factor, used, factors, cur_max, k_target):
            nonlocal best_result, best_max
            
            if time.time() > deadline:
                return
            
            if len(factors) == k_target:
                if rem == 1:
                    if cur_max < best_max:
                        best_max = cur_max
                        best_result = list(factors)
                    return
                else:
                    return
            
            # We need to pick (k_target - len(factors)) more factors
            # Each must be >= min_factor, distinct, and divide rem
            # The product of the remaining factors must be rem
            
            # Pruning: if cur_max >= best_max, we can still continue if we might find
            # a better solution, but actually if cur_max >= best_max, any completion
            # will have max >= cur_max >= best_max, so we can prune.
            if cur_max >= best_max:
                return
            
            # Another pruning: the smallest possible product of (k_target - len(factors))
            # distinct factors each >= min_factor is min_factor * (min_factor+1) * ... 
            # But this is complex. Simpler: the remaining factors must multiply to rem,
            # and each is >= min_factor. So rem must be >= min_factor^(k_target - len(factors))
            # Actually, since they're distinct and >= min_factor, the minimum product is
            # min_factor * (min_factor+1) * ... but that's not quite right either.
            # Let's use: each remaining factor is >= min_factor, so rem >= min_factor^(remaining_count)
            remaining_count = k_target - len(factors)
            if rem < min_factor ** remaining_count:
                return
            
            # Find all divisors of rem that are >= min_factor and not in used
            # and are > 12
            candidates = []
            for d in valid_divs:
                if d < min_factor:
                    continue
                if d in used:
                    continue
                if rem % d != 0:
                    continue
                # Additional pruning: if d > rem, skip (but d divides rem, so d <= rem)
                candidates.append(d)
            
            # Sort candidates to explore smaller ones first (to keep max small)
            candidates.sort()
            
            for d in candidates:
                if time.time() > deadline:
                    return
                used.add(d)
                factors.append(d)
                new_max = max(cur_max, d)
                dfs(rem // d, d + 1, used, factors, new_max, k_target)
                factors.pop()
                used.remove(d)
        
        # Initial call
        used = set()
        factors = []
        dfs(target, 13, used, factors, 0, k)
    
    # If no solution found with k >= 2, fall back to k=1
    if best_result is None:
        best_result = [target]
    
    return best_result
# EVOLVE-BLOCK-END

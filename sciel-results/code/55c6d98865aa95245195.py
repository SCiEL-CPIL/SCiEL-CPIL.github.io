import sys
from math import factorial
from functools import lru_cache
import time
import bisect

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 90s kill

    target = factorial(12)  # 479001600

    # Get all divisors of target
    all_divs = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
        i += 1

    # Only divisors > 12, sorted
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)

    if n_divs == 0:
        return [target]

    MAX_FACTORS = 12

    def check_solution(max_idx):
        """
        Checks if target can be decomposed into distinct integers from valid_divs[0..max_idx],
        all <= valid_divs[max_idx], with product exactly target.
        Returns list of factors if found, else None.
        """
        candidates = valid_divs[:max_idx + 1]
        n = len(candidates)

        if n == 0:
            return None

        # Precompute suffix maximum products for pruning
        # suffix_max_prod[i][k] = max product of at most k distinct factors from candidates[i:]
        # Since factors are > 1, max product of at most k factors is product of largest min(k, n-i) factors.
        # We only need k up to MAX_FACTORS.
        
        # Optimization: 
        # If k >= n - i, max_prod is product(candidates[i:])
        # Else, max_prod is product(candidates[n-k:n])
        
        # Precompute prefix products of reversed candidates to get product of largest k quickly?
        # Let rev_cands = candidates[::-1]
        # prefix_prod[j] = product(rev_cands[0:j]) = product of largest j elements in candidates.
        # Then product of largest k elements in candidates is prefix_prod[k].
        # But we need product of largest k elements in candidates[i:].
        # The largest elements in candidates[i:] are still the largest elements in candidates, 
        # UNLESS i > n-k.
        
        # Let's just compute a 2D array or use a helper function.
        # Given small MAX_FACTORS (12), we can compute on the fly or precompute a small table.
        
        # Table: max_prod_bound[i][k]
        # Size: n x 13. n is ~6000. 6000 * 13 * 8 bytes ~ 600KB. Acceptable.
        
        max_prod_bound = [[1] * (MAX_FACTORS + 1) for _ in range(n + 1)]
        
        # suffix_product[i] = product(candidates[i:])
        suffix_product = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_product[i] = suffix_product[i + 1] * candidates[i]
            
        # prefix_product of reversed candidates (largest elements)
        # rev_prefix[j] = product of largest j elements in candidates
        rev_prefix = [1] * (MAX_FACTORS + 1)
        # Largest elements are candidates[n-1], candidates[n-2], ...
        for j in range(1, MAX_FACTORS + 1):
            if j > n:
                rev_prefix[j] = 0 # Should not happen if we check bounds
            else:
                rev_prefix[j] = rev_prefix[j - 1] * candidates[n - j]
                
        # Fill max_prod_bound
        for i in range(n, -1, -1):
            for k in range(1, MAX_FACTORS + 1):
                remaining_count = n - i
                if k >= remaining_count:
                    # Can take all remaining
                    max_prod_bound[i][k] = suffix_product[i]
                else:
                    # Can take largest k from the whole set, provided they are in candidates[i:]
                    # The largest k elements are candidates[n-k ... n-1]
                    # They are in candidates[i:] if and only if i <= n-k
                    if i <= n - k:
                        max_prod_bound[i][k] = rev_prefix[k]
                    else:
                        # This case is covered by k >= remaining_count? 
                        # If i > n-k, then n-i < k, so remaining_count < k.
                        # So it falls into the first case.
                        max_prod_bound[i][k] = suffix_product[i]

        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True
            if i >= n or count >= MAX_FACTORS:
                return False
            
            # Pruning 1: If rem is greater than the max product achievable with remaining slots
            slots_left = MAX_FACTORS - count
            if rem > max_prod_bound[i][slots_left]:
                return False
                
            # Pruning 2: If rem is prime and not in candidates[i:], fail?
            # Hard to check primality quickly for large numbers, but divisors of 12! are small.
            # However, rem is a divisor of 12!, so its factors are small.
            # If rem is not divisible by any candidate in candidates[i:], fail.
            # This is effectively handled by the recursion, but we can add a quick check.
            # Since candidates are sorted, we can check if rem % c == 0 for any c.
            # But we have to pick distinct factors.
            
            c = candidates[i]
            
            # Option 1: Skip candidates[i]
            if dfs(i + 1, rem, count):
                return True
            
            # Option 2: Take candidates[i] if it divides rem
            if rem % c == 0:
                if dfs(i + 1, rem // c, count + 1):
                    return True
            
            return False

        if dfs(0, target, 0):
            # Reconstruct solution
            result = []
            i, rem, count = 0, target, 0
            while rem != 1:
                if i >= n or count >= MAX_FACTORS:
                    break
                c = candidates[i]
                # Check if skipping leads to solution
                if dfs(i + 1, rem, count):
                    i += 1
                else:
                    # Must have taken c
                    if rem % c != 0:
                        break # Should not happen
                    result.append(c)
                    rem //= c
                    count += 1
                    i += 1
            
            return result
        else:
            return None

    # Binary search for the minimal maximum factor
    low = 0
    high = n_divs - 1
    best_result = None
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
        
        mid = (low + high) // 2
        max_val = valid_divs[mid]
        
        sol = check_solution(mid)
        
        if sol is not None:
            best_result = sol
            high = mid - 1
        else:
            low = mid + 1

    if best_result is not None:
        return best_result
    
    # Fallback: trivial solution
    return [target]

# EVOLVE-BLOCK-END

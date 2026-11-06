import sys
from math import factorial
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)
    
    # Get all divisors of target
    all_divs = []
    for i in range(1, int(target ** 0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    
    # Only divisors > 12
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)
    
    if n_divs == 0:
        return [target]

    # We know max factors is 7
    MAX_FACTORS = 7

    # We will iterate M over valid_divs
    # To speed up, we can start from a reasonable lower bound?
    # No, just iterate.
    
    # Precompute suffix max products for each candidate list?
    # Since the candidate list changes with M, we can't precompute globally.
    # But we can compute the bound on the fly.
    
    def check_solution(max_idx):
        """
        Checks if a solution exists using valid_divs[0..max_idx].
        Returns the list of factors if found, else None.
        """
        candidates = valid_divs[:max_idx + 1]
        n = len(candidates)
        
        # If the product of the largest 7 candidates is less than target, impossible
        # Largest 7 are candidates[n-7:n] if n>=7, else all
        k_max = min(MAX_FACTORS, n)
        max_prod = 1
        for i in range(n - k_max, n):
            max_prod *= candidates[i]
            
        if max_prod < target:
            return None

        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True
            if i >= n or count >= MAX_FACTORS:
                return False
            
            # Pruning: Check if we can reach rem with remaining candidates
            # Max product from candidates[i:] with (MAX_FACTORS - count) factors
            avail = n - i
            k_left = MAX_FACTORS - count
            if k_left > avail:
                k_left = avail
            
            max_possible = 1
            # Take largest k_left from candidates[i:]
            # They are at the end of the slice candidates[i:n]
            # i.e., candidates[n-k_left : n]
            for j in range(n - k_left, n):
                max_possible *= candidates[j]
            
            if rem > max_possible:
                return False

            c = candidates[i]
            
            # Option 1: Skip candidates[i]
            if dfs(i + 1, rem, count):
                return True
            
            # Option 2: Take candidates[i]
            if rem % c == 0:
                if dfs(i + 1, rem // c, count + 1):
                    return True
            
            return False

        if dfs(0, target, 0):
            # Reconstruct solution
            result = []
            i, rem, count = 0, target, 0
            while rem != 1:
                if i >= n:
                    break
                c = candidates[i]
                
                # Check if we can solve by skipping i
                if dfs(i + 1, rem, count):
                    i += 1
                elif rem % c == 0 and dfs(i + 1, rem // c, count + 1):
                    result.append(c)
                    rem //= c
                    count += 1
                    i += 1
                else:
                    break
            
            return result
        else:
            return None

    best_result = None
    
    for idx in range(n_divs):
        if time.time() - start_time > timeout:
            break
        
        sol = check_solution(idx)
        
        if sol is not None:
            best_result = sol
            break # Since we iterate M in increasing order, first found is optimal
            
    if best_result is not None:
        return best_result
    
    return [target]

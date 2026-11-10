import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    max_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12!
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        for d in divisors:
            for e in range(max_exponents[i] + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    
    # Filter divisors > 12 and sort
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    # Map divisor to its prime exponents tuple for fast lookup
    divisor_to_pf = {}
    for d in valid_divisors:
        n = d
        pf = [0, 0, 0, 0, 0]
        for i, p in enumerate(primes):
            while n % p == 0:
                pf[i] += 1
                n //= p
        divisor_to_pf[d] = tuple(pf)
        
    # Precompute "next smallest divisors" product?
    # Given a last_factor and count m, what is the min product of m distinct divisors > last_factor?
    # We can compute this on the fly using a helper function.
    
    def min_product_of_next_m(last_factor, m):
        """
        Returns the product of the m smallest distinct divisors from valid_divisors 
        that are strictly greater than last_factor.
        Returns None if fewer than m such divisors exist.
        """
        if m == 0:
            return 1
        prod = 1
        count = 0
        # valid_divisors is sorted
        # Find start index
        # Linear scan is fine since list is small (~700)
        for d in valid_divisors:
            if d > last_factor:
                prod *= d
                count += 1
                if count == m:
                    return prod
        return None

    best_max_val = float('inf')
    best_factors = []

    # Target prime exponents
    target_pf = tuple(max_exponents)

    def dfs(last_factor, rem_pf, count_left, current_path, current_max):
        nonlocal best_max_val, best_factors
        
        if time.time() - start_time > 85:
            return

        # Check time limit
        if time.time() - start_time > 85:
            return

        if count_left == 0:
            if rem_pf == (0, 0, 0, 0, 0):
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_factors = list(current_path)
            return

        # Pruning 1: If current_max is already >= best, no point continuing
        # (Actually, we pick factors in increasing order, so current_max is the last factor)
        if current_max >= best_max_val:
            return

        # Pruning 2: Minimum possible product for remaining factors
        min_prod = min_product_of_next_m(last_factor, count_left)
        if min_prod is None:
            return
        
        # Calculate current remaining target value from prime factors
        # We need to check if rem_target < min_prod
        # rem_target = product of primes^exponents
        rem_target = 1
        for i in range(5):
            rem_target *= primes[i] ** rem_pf[i]
            
        if rem_target < min_prod:
            return
            
        # Pruning 3: Maximum possible product? 
        # The largest factor we can pick is bounded by best_max_val.
        # Also, the remaining product must be at least the product of the largest possible factors?
        # Not strictly necessary if we iterate candidates.

        # Determine candidates for the next factor x
        # x must be > last_factor
        # x must be < best_max_val (strictly, to improve)
        # x must be a divisor of rem_target (i.e., pf[x] <= rem_pf)
        # x must be in valid_divisors
        
        # Iterate through valid_divisors
        for x in valid_divisors:
            if x <= last_factor:
                continue
            if x >= best_max_val:
                break # Since sorted, no further x can improve
            
            # Check if x divides rem_target
            pf_x = divisor_to_pf[x]
            can_divide = True
            for i in range(5):
                if pf_x[i] > rem_pf[i]:
                    can_divide = False
                    break
            if not can_divide:
                continue
                
            # Pruning 4: After picking x, can the rest be formed?
            new_rem_pf = tuple(rem_pf[i] - pf_x[i] for i in range(5))
            new_count_left = count_left - 1
            
            # The next factors must be > x
            # Min product for new_count_left factors > x
            if new_count_left > 0:
                min_prod_next = min_product_of_next_m(x, new_count_left)
                if min_prod_next is None:
                    continue
                # Calculate new remaining target
                new_rem_target = rem_target // x
                if new_rem_target < min_prod_next:
                    continue
            else:
                if new_rem_pf != (0, 0, 0, 0, 0):
                    continue

            current_path.append(x)
            dfs(x, new_rem_pf, new_count_left, current_path, x)
            current_path.pop()

    # Search for k factors
    # We can search k from 1 to 12.
    # However, more factors usually allow smaller max.
    # Let's try all k.
    for k in range(1, 13):
        # Optional: lower bound on k?
        # If we find a solution with max M, we don't need to find another with same M.
        # We update best_max_val globally.
        dfs(12, target_pf, k, [], 12)
        
        # Early exit if we found a very good solution?
        # Theoretical min max is around 13-20.
        if best_max_val <= 20:
             # Can't get much better than 20 if factors are distinct > 12
             # 13, 14, 15... 
             # If best is 13, it must be a single factor 13? No, product is 12!.
             # So 13 is impossible as max if product is 12!.
             # If max is 20, factors are <= 20.
             # We can stop if we are confident.
             pass

    if not best_factors:
        # Fallback: return a valid solution, even if not optimal
        # Just return [12!] if nothing found? No, must be distinct > 12.
        # 12! is > 12. So [12!] is a valid solution with k=1.
        best_factors = [N]
        best_max_val = N

    return best_factors

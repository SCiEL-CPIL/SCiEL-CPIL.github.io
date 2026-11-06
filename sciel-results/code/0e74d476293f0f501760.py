"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial
    import itertools
    import math

    target = factorial(12)
    
    # Prime factorization of 12!
    # 12! = 479001600
    # Primes <= 12: 2, 3, 5, 7, 11
    # Exponents in 12!:
    # 2: floor(12/2) + floor(12/4) + floor(12/8) = 6 + 3 + 1 = 10
    # 3: floor(12/3) + floor(12/9) = 4 + 1 = 5
    # 5: floor(12/5) = 2
    # 7: floor(12/7) = 1
    # 11: floor(12/11) = 1
    
    # We need to find a set of distinct integers > 12, product = 12!, minimizing the max.
    # Approach: Backtracking search over divisors.
    # State: remaining product, set of used factors, current max factor.
    # We want to minimize the maximum factor.
    
    # To make this efficient, we can iterate through possible divisors in a smart way.
    # Since we want to minimize the max, we should try to keep factors as small as possible?
    # No, minimizing the max means we want all factors to be "balanced" or at least the largest one to be small.
    # Actually, if we fix the number of factors k, the optimal is when they are roughly equal to (12!)^(1/k).
    # But k is variable (1..12).
    
    # Let's generate all divisors of 12! greater than 12.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Total divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
    # This is a small number. We can generate all divisors, filter those > 12, and sort them.
    
    divisors = []
    def gen_divisors(exp_2, exp_3, exp_5, exp_7, exp_11, current):
        if exp_2 == 10 and exp_3 == 5 and exp_5 == 2 and exp_7 == 1 and exp_11 == 1:
            if current > 12:
                divisors.append(current)
            return
        # Next prime is 2
        if exp_2 < 10:
            gen_divisors(exp_2 + 1, exp_3, exp_5, exp_7, exp_11, current * 2)
        # Next prime is 3
        if exp_3 < 5:
            gen_divisors(exp_2, exp_3 + 1, exp_5, exp_7, exp_11, current * 3)
        # Next prime is 5
        if exp_5 < 2:
            gen_divisors(exp_2, exp_3, exp_5 + 1, exp_7, exp_11, current * 5)
        # Next prime is 7
        if exp_7 < 1:
            gen_divisors(exp_2, exp_3, exp_5, exp_7 + 1, exp_11, current * 7)
        # Next prime is 11
        if exp_11 < 1:
            gen_divisors(exp_2, exp_3, exp_5, exp_7, exp_11 + 1, current * 11)
            
    gen_divisors(0, 0, 0, 0, 0, 1)
    divisors.sort()
    
    best_max = float('inf')
    best_factors = []
    
    # Backtracking
    # We need to pick distinct divisors from `divisors` such that their product is `target`.
    # We want to minimize the maximum factor chosen.
    
    # Optimization: 
    # 1. If the product of remaining factors must be 1, we are done.
    # 2. The next factor must be <= remaining_product.
    # 3. The next factor must be distinct from previously chosen.
    # 4. To minimize the max, we can prune if the current candidate max is already >= best_max.
    # 5. Also, if we pick a factor `d`, the remaining product is `rem // d`.
    #    The next factor must be >= `d` to avoid duplicates? No, they just need to be distinct.
    #    However, to avoid permutations, we can enforce an ordering: f1 < f2 < ... < fk.
    #    If we enforce strict increasing order, we cover each set exactly once.
    
    def backtrack(rem, start_idx, current_factors, current_max):
        nonlocal best_max, best_factors
        
        if rem == 1:
            if len(current_factors) > 0:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
            
        # Pruning: if current_max >= best_max, we might still improve if we haven't added the max yet?
        # No, current_max is the max of factors chosen so far. The final max will be >= current_max.
        # So if current_max >= best_max, we can prune? 
        # Wait, if we add more factors, the max could stay the same or increase.
        # So if current_max >= best_max, any solution found will have max >= current_max >= best_max.
        # So we can prune.
        if current_max >= best_max:
            return
            
        # Another pruning: 
        # The minimum possible value for the next factor is divisors[start_idx].
        # If divisors[start_idx] > rem, we can't pick any more factors (since all subsequent are larger).
        # But we need rem to become 1. If rem > 1, we must pick at least one more factor.
        # So if divisors[start_idx] > rem, we cannot complete the product (unless rem==1, handled above).
        # Actually, if rem > 1, we need a divisor d of rem such that d >= divisors[start_idx].
        # If no such d exists in the list from start_idx onwards, we fail.
        
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max:
                # Since divisors are sorted, any subsequent d' >= d will also have new_max' >= new_max >= best_max.
                # So we can break?
                # Yes, because d is increasing, so new_max is non-decreasing.
                break
                
            backtrack(rem // d, i + 1, current_factors + [d], new_max)
            
    backtrack(target, 0, [], 0)
    
    if best_factors:
        return best_factors
    else:
        # Fallback, should not happen
        return [target]

# EVOLVE-BLOCK-END

import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    target = 479001600  # 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Meet-in-the-middle approach:
    # Split into two groups of primes
    # Group A: 2, 3 (exponents 0-10, 0-5)
    # Group B: 5, 7, 11 (exponents 0-2, 0-1, 0-1)
    
    # Generate all possible "half-factors" for each group
    # A half-factor is a tuple (value_a, value_b, exps) where:
    #   value_a = product of 2^e2 * 3^e3 for some e2, e3
    #   value_b = product of 5^e5 * 7^e7 * 11^e11 for some e5, e7, e11
    #   The full factor is value_a * value_b
    
    # Actually, let me think differently.
    # We need to partition the prime exponents into k distinct factors, each > 12.
    # Each factor is a product of some primes with some exponents.
    
    # Better approach: 
    # 1. Generate all divisors of 12! that are > 12
    # 2. Use DFS/backtracking to find a set of distinct divisors whose product is 12!
    # 3. Minimize the largest factor
    
    # Let me use a smarter DFS with strong pruning.
    
    # First, generate all divisors of 12! that are > 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12:
                            divisors.append(d)
    
    # Remove duplicates and sort
    divisors = sorted(set(divisors))
    n_divs = len(divisors)
    
    # Map value to exponents
    val_to_exps = {}
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        val_to_exps[d] = (e2, e3, e5, e7, e11)
    
    target_exps = (10, 5, 2, 1, 1)
    
    # DFS with pruning
    # State: remaining exponents, current max factor, number of factors used
    # We want to minimize the max factor
    
    best_solution = [target]  # Fallback: single factor
    best_max = target
    
    # Sort divisors in ascending order for better pruning
    # We'll build the solution greedily, trying to keep factors small
    
    # Use a set to track used divisors
    used = set()
    
    def dfs(rem_exps, current_factors, current_max, idx):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return
        
        # Check if we've used the target
        if rem_exps == (0, 0, 0, 0, 0):
            if current_factors:
                max_factor = max(current_factors)
                if max_factor < best_max:
                    best_max = max_factor
                    best_solution = list(current_factors)
            return
        
        # Pruning: if current_max >= best_max, no point continuing
        if current_max >= best_max:
            return
        
        # Pruning: if we can't reach zero with remaining factors
        # The remaining product must be >= 1
        rem_product = (2 ** rem_exps[0]) * (3 ** rem_exps[1]) * (5 ** rem_exps[2]) * (7 ** rem_exps[3]) * (11 ** rem_exps[4])
        
        if rem_product == 1:
            return  # Already handled above
        
        # We need to pick a divisor that divides the remaining product
        # and is > 12 and not already used
        
        # Optimization: try divisors in ascending order
        # But also, we should try to use larger divisors first to reduce the number of factors?
        # Actually, to minimize the max, we want to balance the factors.
        
        # Let's try divisors from smallest to largest, but with a twist:
        # If we pick a small divisor, we might need more factors, which could increase the max.
        # If we pick a larger divisor, we might need fewer factors.
        
        # Heuristic: try divisors that are close to the "ideal" size
        # Ideal size = rem_product^(1/n_remaining)
        # But we don't know n_remaining.
        
        # Simpler: just try all valid divisors, but prune aggressively.
        
        for i in range(idx, n_divs):
            if time.time() > deadline:
                return
            
            d = divisors[i]
            
            # Skip if already used
            if d in used:
                continue
            
            # Check if d divides the remaining product
            d_exps = val_to_exps[d]
            if (d_exps[0] > rem_exps[0] or d_exps[1] > rem_exps[1] or 
                d_exps[2] > rem_exps[2] or d_exps[3] > rem_exps[3] or 
                d_exps[4] > rem_exps[4]):
                continue
            
            # Pruning: if d > best_max, skip (since we're trying to minimize max)
            if d >= best_max:
                continue
            
            # New remaining exponents
            new_rem = (rem_exps[0] - d_exps[0], rem_exps[1] - d_exps[1],
                       rem_exps[2] - d_exps[2], rem_exps[3] - d_exps[3],
                       rem_exps[4] - d_exps[4])
            
            # New current max
            new_max = max(current_max, d)
            
            # Pruning: if new_max >= best_max, skip
            if new_max >= best_max:
                continue
            
            # Additional pruning: the remaining product must be achievable with distinct factors > 12
            # The minimum product of k distinct factors > 12 is at least 13*14*15*...*(12+k)
            # But this is complex. Let's use a simpler bound.
            
            # If the remaining product is > 1, we need at least one more factor.
            # The smallest possible factor is 13, so if rem_product < 13, we can't form it.
            rem_prod = (2 ** new_rem[0]) * (3 ** new_rem[1]) * (5 ** new_rem[2]) * (7 ** new_rem[3]) * (11 ** new_rem[4])
            if rem_prod > 1 and rem_prod < 13:
                continue
            
            # Also, if we have too many factors, skip
            if len(current_factors) >= 12:
                continue
            
            # Recurse
            used.add(d)
            dfs(new_rem, current_factors + [d], new_max, i + 1)
            used.remove(d)
            
            # If we found a solution, we can return early
            if best_max < d:
                return
    
    # Start DFS
    dfs(target_exps, [], 0, 0)
    
    return best_solution

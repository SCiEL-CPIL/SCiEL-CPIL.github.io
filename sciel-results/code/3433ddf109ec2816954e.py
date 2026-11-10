import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors
    divisors = [1]
    for p, e in zip(primes, target_exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    
    # Pre-compute exponent vectors for each valid divisor
    def get_exp_vector(n):
        exps = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return tuple(exps)
    
    div_exps = [get_exp_vector(d) for d in valid_divs]
    
    # Helper to check if rem_exp is <= suffix_sum
    def le_exp(a, b):
        return all(x <= y for x, y in zip(a, b))
    
    # We will iterate through the sorted divisors.
    # For each candidate maximum M = valid_divs[i], we check if a solution exists
    # using only divisors from valid_divs[0..i].
    
    # To make the check fast, we pre-compute suffix sums of exponents for the full list?
    # No, the list changes as i increases.
    # But we can just pass the sublist or indices.
    # Given N ~ 780, and depth <= 12, and strong pruning, it should be fine.
    
    # Optimization: 
    # The search space for "subset sum" in exponent space is small.
    # We can use a recursive function that takes:
    # - rem_exp: tuple of remaining exponents needed
    # - start_idx: index in valid_divs to start looking from
    # - current_factors: list of chosen factors
    # - max_allowed_idx: i (the current candidate max index)
    
    # To avoid deep recursion overhead, we can use a stack or just simple recursion.
    # Python recursion limit is usually 1000, depth is <= 12, so it's fine.
    
    def can_solve(rem_exp, start_idx, max_idx, current_factors):
        # rem_exp: tuple
        # start_idx: int
        # max_idx: int (inclusive)
        
        if time.time() > deadline:
            return False, []
            
        if all(x == 0 for x in rem_exp):
            return True, list(current_factors)
        
        if start_idx > max_idx:
            return False, []
            
        # Pruning: 
        # Calculate sum of exponents available from start_idx to max_idx
        # We can precompute this? 
        # Since max_idx changes in the outer loop, precomputing a 2D table is possible.
        # suffix_sums[i][j] = sum of exps from i to j? Too much memory/time to precompute all.
        # But we can just check if the "theoretical max" sum of exponents in the range is sufficient.
        # A simpler prune: 
        # If any component of rem_exp is greater than the sum of that component in all divs[start_idx..max_idx], return False.
        
        # Let's compute the sum of exponents in the range quickly.
        # Since the range is a suffix of the prefix [0..max_idx], 
        # and we are iterating start_idx upwards, 
        # we can just sum. N is small.
        
        # Actually, we can precompute prefix sums of exponents.
        # prefix_exp[i] = sum of exps[0..i-1]
        # Then sum(start..end) = prefix[end+1] - prefix[start]
        
        pass # Placeholder for logic below

    # Precompute prefix sums of exponent vectors
    # prefix_exp[i] is the sum of exps of valid_divs[0..i-1]
    prefix_exp = [tuple([0]*len(primes))]
    for exp in div_exps:
        last = prefix_exp[-1]
        new_sum = tuple(last[k] + exp[k] for k in range(len(primes)))
        prefix_exp.append(new_sum)
        
    # Now sum of exps from i to j is prefix_exp[j+1] - prefix_exp[i]
    
    def can_solve(rem_exp, start_idx, max_idx, current_factors):
        if time.time() > deadline:
            return False, []
            
        if all(x == 0 for x in rem_exp):
            return True, list(current_factors)
        
        if start_idx > max_idx:
            return False, []
            
        # Prune: check if available exponents in [start_idx, max_idx] are sufficient
        # available = prefix_exp[max_idx+1] - prefix_exp[start_idx]
        for k in range(len(primes)):
            available = prefix_exp[max_idx+1][k] - prefix_exp[start_idx][k]
            if rem_exp[k] > available:
                return False, []
                
        # Iterate candidates
        for i in range(start_idx, max_idx + 1):
            exp_i = div_exps[i]
            # Check if exp_i can be subtracted from rem_exp
            if all(rem_exp[k] >= exp_i[k] for k in range(len(primes))):
                new_rem = tuple(rem_exp[k] - exp_i[k] for k in range(len(primes)))
                current_factors.append(valid_divs[i])
                res, sol = can_solve(new_rem, i + 1, max_idx, current_factors)
                current_factors.pop()
                if res:
                    return True, sol
        return False, []

    # Iterate M
    # We start from the smallest possible M.
    # Note: The solution must have at least 1 factor.
    # The smallest factor is valid_divs[0].
    
    for i in range(len(valid_divs)):
        # M is valid_divs[i]
        # We look for a solution using divs[0..i]
        # Note: The last factor in the solution will be <= valid_divs[i].
        # Is it possible that the solution doesn't use valid_divs[i]?
        # Yes. But if a solution exists using {0..i-1}, it would have been found in previous iterations.
        # So if we find a solution now, its max is <= valid_divs[i].
        # Since we iterate i increasing, the first solution found has the minimal max.
        
        # Optimization: 
        # If valid_divs[i] is very large, we might find a solution earlier.
        
        # One more prune: 
        # If the product of the largest possible factors is too small?
        # No, we are matching exponents exactly.
        
        # Start search
        res, sol = can_solve(tuple(target_exps), 0, i, [])
        
        if res:
            # We found a solution with max factor <= valid_divs[i]
            # Since we iterate i in increasing order, valid_divs[i] is the minimal possible max?
            # Wait. The solution `sol` consists of factors from `valid_divs[0..i]`.
            # The max factor in `sol` is `max(sol)`.
            # It is possible that `max(sol) < valid_divs[i]`.
            # If so, we would have found it in a previous iteration?
            # Yes, if `max(sol) = valid_divs[j]` with `j < i`, then in iteration `j`,
            # the search space was `valid_divs[0..j]`, which contains `sol`.
            # So `can_solve` would have returned True at `j`.
            # Thus, the first `i` for which `can_solve` returns True corresponds to a solution
            # where `max(sol) == valid_divs[i]`? 
            # Not necessarily. It could be that `sol` uses `valid_divs[i]` or not.
            # But if it doesn't use `valid_divs[i]`, it would have been found earlier.
            # So effectively, the first success is optimal.
            
            return sol

    # If no solution found with distinct factors > 12?
    # The problem guarantees a solution exists? 
    # "Return 1..12 DISTINCT integers...". 
    # 12! itself is a valid solution (1 factor).
    # valid_divs includes 12! (since 12! > 12).
    # So the loop will eventually reach i where valid_divs[i] == 12!.
    # And can_solve will find [12!].
    
    return [target]

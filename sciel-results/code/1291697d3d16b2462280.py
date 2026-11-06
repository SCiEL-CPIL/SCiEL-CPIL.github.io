"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    target = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Let's verify:
    # 2s: 6+4+2+1 (from 2,4,6,8,10,12) -> 1+2+1+3+1+2 = 10
    # 3s: 4+2+1 (from 3,6,9,12) -> 1+1+2+1 = 5
    # 5s: 2+1 (from 5,10) -> 1+1 = 2
    # 7s: 1 (from 7) -> 1
    # 11s: 1 (from 11) -> 1
    target_exponents = [10, 5, 2, 1, 1]
    primes = [2, 3, 5, 7, 11]
    
    # We need to find a set of distinct integers > 12 whose product is 12!
    # Minimize the largest factor.
    # 
    # Strategy:
    # 1. Generate all divisors of 12! that are > 12.
    # 2. Sort them.
    # 3. Use backtracking to find a subset of these divisors such that:
    #    - Their product is 12!
    #    - They are distinct
    #    - All are > 12
    #    - The number of factors is between 1 and 12
    #    - We want to minimize the maximum element in the subset.
    #
    # Since we want to minimize the max factor, we can try to build solutions
    # using only factors from a small range [13, L] and increase L if no solution is found.
    # However, the number of divisors of 12! is manageable.
    # Number of divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
    # So there are at most 792 divisors.
    
    # Generate all divisors of 12!
    divisors = []
    for e0 in range(11):
        for e1 in range(6):
            for e2 in range(3):
                for e3 in range(2):
                    for e4 in range(2):
                        d = (2**e0) * (3**e1) * (5**e2) * (7**e3) * (11**e4)
                        if d > 12:
                            divisors.append(d)
    
    # Sort divisors
    divisors.sort()
    
    # Map divisor to exponent vector for quick lookup
    # Actually, we can just store the exponent vector along with the value.
    # Let's create a list of (value, exponents_tuple)
    div_with_exp = []
    for d in divisors:
        # Recover exponents
        rem = d
        exps = []
        for p in primes:
            e = 0
            while rem % p == 0:
                rem //= p
                e += 1
            exps.append(e)
        # rem should be 1
        div_with_exp.append((d, tuple(exps)))
    
    # We want to find a subset of these divisors that multiplies to 12!
    # This is equivalent to: sum of exponent vectors = target_exponents
    # And the values are distinct (which they are since divisors are unique).
    # And count is between 1 and 12.
    
    # To minimize the max factor, we can iterate through possible max values L
    # and check if there's a valid subset using only divisors <= L.
    # Since divisors are sorted, we can try L = divisors[i] for i in range(len(divisors)).
    # For each L, we restrict our search space to divisors[0..i].
    
    # Backtracking search with forward checking
    # State: current index in divisors list (up to limit), remaining exponent vector, count of factors used
    # We want to find ANY valid subset.
    
    # To make it faster, we can use memoization on (remaining_exponents, max_index_allowed, count_used)
    # But count_used is not critical for the search itself, only for the constraint <= 12.
    # Since 12! has 792 divisors, and we're looking for a small number of factors,
    # the depth of the search tree will be small.
    
    # Let's define a recursive function with memoization.
    # Key: (remaining_exps_tuple, max_idx)
    # Value: True if a solution exists, and we can also store the solution path.
    
    # Actually, we just need to find ONE solution for a given L.
    # We'll try increasing L until we find a solution.
    
    # Let's precompute for each divisor its index in the sorted list.
    # We'll use a recursive backtracking search.
    
    def solve(remaining, max_idx, current_factors):
        """
        remaining: tuple of exponents remaining
        max_idx: maximum index in div_with_exp we can use (inclusive)
        current_factors: list of factors chosen so far
        Returns: list of factors if a solution is found, else None
        """
        if all(x == 0 for x in remaining):
            return current_factors[:]
        
        if len(current_factors) >= 12:
            return None
        
        # Try to pick a divisor from div_with_exp[0..max_idx]
        # To minimize the largest factor, we should try larger factors first?
        # No, we are fixing the max factor L. So we just need to find any combination.
        # But to find a solution quickly, maybe we should try divisors that are "large"
        # to reduce the remaining product quickly? Or small?
        # Let's just iterate through all possible divisors up to max_idx.
        
        # Optimization: if the product of remaining exponents is 1, we need one more factor.
        # The factor must be exactly the number formed by remaining exponents.
        # Check if that number is in div_with_exp[0..max_idx] and not already used.
        
        # Let's compute the target value for the last factor if we pick just one more.
        target_val = 1
        for i, e in enumerate(remaining):
            target_val *= primes[i] ** e
        
        # If target_val is in our allowed set and not used, we can pick it.
        # But we need to check if it's <= div_with_exp[max_idx][0] and distinct.
        
        # General case: try each divisor
        for i in range(max_idx + 1):
            val, exps = div_with_exp[i]
            # Check if this divisor can be part of the solution
            # i.e., for all j, exps[j] <= remaining[j]
            valid = True
            for j in range(5):
                if exps[j] > remaining[j]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check distinctness: val should not be in current_factors
            # Since divisors are distinct, and we're picking from a list of distinct divisors,
            # we just need to ensure we don't pick the same index twice.
            # But our loop goes through indices, so we won't pick the same index twice.
            # However, we might pick a different index with the same value? No, values are unique.
            
            # Pick this divisor
            new_remaining = tuple(remaining[j] - exps[j] for j in range(5))
            result = solve(new_remaining, i - 1, current_factors + [val])
            if result is not None:
                return result
        
        return None
    
    # We need to try increasing L (max factor)
    # L must be at least the smallest divisor > 12, which is 13? No, 13 is prime, not a divisor of 12!.
    # Smallest divisor of 12! > 12:
    # Divisors: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, ...
    # 14 = 2*7, yes.
    
    # We'll iterate through div_with_exp, treating each as a potential max factor L.
    # For each L, we call solve(target_exps_tuple, index_of_L, []).
    # The first L for which solve returns a valid solution is our answer.
    
    target_exp_tuple = tuple(target_exponents)
    
    for i, (val, exps) in enumerate(div_with_exp):
        # Try to find a solution using only divisors with index <= i
        solution = solve(target_exp_tuple, i, [])
        if solution is not None:
            # We found a solution with max factor <= val
            # But we need to ensure the max factor in the solution is actually minimized.
            # Since we're iterating L in increasing order, the first solution we find
            # will have the smallest possible max factor.
            # However, the solution might not use 'val' itself.
            # But the max factor in the solution is <= val.
            # Since we're looking for the minimal possible max factor, this is correct.
            return solution
    
    # If no solution found (should not happen), return the trivial solution
    return [target]
# EVOLVE-BLOCK-END

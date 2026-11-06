"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache
from bisect import bisect_right

T12 = math.factorial(12)  # 479001600

# Prime factorization of 12!
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11
PRIMES = [2, 3, 5, 7, 11]
INITIAL_EXP = (10, 5, 2, 1, 1)

def exp_to_val(exp):
    """Convert exponent tuple to integer value."""
    result = 1
    for i, e in enumerate(exp):
        if e > 0:
            result *= PRIMES[i] ** e
    return result

def val_to_exp(n):
    """Convert integer to exponent tuple for the known primes. Returns None if n has other prime factors."""
    exp = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exp[i] += 1
            n //= p
    if n != 1:
        return None
    return tuple(exp)

def solve():
    """
    Find distinct factors > 12 multiplying to 12! minimizing the largest factor.
    Uses DFS over exponent vectors with strong pruning.
    """
    # Use a deadline well within the 60s limit to be safe
    deadline = time.time() + 55
    
    best_max = T12
    best_solution = [T12]
    
    # We will build factors in strictly increasing order.
    # State: (remaining_exp_tuple, last_factor_val)
    # We need to find a sequence of factors f1 < f2 < ... < fk such that:
    # 1. fi >= 13
    # 2. Product fi = 12!
    # 3. Minimize fk (the largest)
    
    # Precompute divisors for all possible exponent states
    # Number of states is (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 792
    divisor_cache = {}
    
    def generate_divisors(exp):
        """Generate all divisors (as values) of the number represented by exp."""
        divs = [1]
        for i, e in enumerate(exp):
            if e > 0:
                p = PRIMES[i]
                powers = [1]
                for k in range(1, e + 1):
                    powers.append(powers[-1] * p)
                new_divs = []
                for d in divs:
                    for pow_val in powers:
                        new_divs.append(d * pow_val)
                divs = new_divs
        return divs

    # Precompute all divisors for all possible states
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        exp = (e2, e3, e5, e7, e11)
                        divisor_cache[exp] = sorted(generate_divisors(exp))
    
    # Precompute val_to_exp for all possible divisor values to avoid recomputation
    val_exp_cache = {}
    for exp in divisor_cache:
        for d in divisor_cache[exp]:
            if d >= 13:
                val_exp_cache[d] = val_to_exp(d)

    # DFS
    def dfs(rem_exp, last_f, chosen, current_max):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
            
        if rem_exp == (0, 0, 0, 0, 0):
            # All factors chosen
            if chosen:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(chosen)
            return
        
        rem_val = exp_to_val(rem_exp)
        
        # Strong pruning: if rem_val itself is a valid single factor
        if rem_val > last_f and rem_val >= 13 and rem_val < best_max:
            best_max = rem_val
            best_solution = chosen + [rem_val]
        
        # Get divisors of rem_val
        divs = divisor_cache[rem_exp]
        
        # Use binary search to find the start index for d > last_f
        start_idx = bisect_right(divs, last_f)
        
        # Iterate through valid divisors in increasing order
        for i in range(start_idx, len(divs)):
            d = divs[i]
            
            if d < 13:
                continue
            if d >= best_max:
                break
            
            # Check if remaining part R = rem_val // d is valid
            R = rem_val // d
            
            # If R > 1, the next factor must be > d, so R must be > d
            if R > 1 and R <= d:
                continue
            
            # Get precomputed exponent for d
            d_exp = val_exp_cache.get(d)
            if d_exp is None:
                continue
            
            new_rem_exp = tuple(rem_exp[i] - d_exp[i] for i in range(5))
            if any(x < 0 for x in new_rem_exp):
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max:
                continue
            
            # Recurse
            dfs(new_rem_exp, d, chosen + [d], new_max)
            
            if time.time() > deadline:
                return

    # Initial call
    dfs(INITIAL_EXP, 0, [], 0)
    
    return best_solution

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    try:
        solution = solve()
        # Verify
        product = 1
        for f in solution:
            product *= f
        if product == T12 and all(f > 12 for f in solution) and len(set(solution)) == len(solution) and 1 <= len(solution) <= 12:
            return solution
    except Exception:
        pass
    
    # Fallback
    return [T12]

# EVOLVE-BLOCK-END

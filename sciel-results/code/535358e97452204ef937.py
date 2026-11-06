# EVOLVE-BLOCK-START
import time
import math
from math import factorial, gcd, prod
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = factorial(12)  # 479001600
    MIN_VAL = 13  # all factors must be > 12
    MAX_FACTORS = 12
    
    # Generate all divisors of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Precompute prime factorization for N
    prime_exp_N = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}

    # Cache for divisors of a number
    divisor_cache = {}
    
    def get_divisors(n):
        if n in divisor_cache:
            return divisor_cache[n]
        
        # Generate divisors of n based on its prime factorization
        # Since n is a divisor of 12!, we can factorize it quickly
        temp = n
        factors = []
        for p in primes:
            cnt = 0
            while temp % p == 0:
                cnt += 1
                temp //= p
            if cnt > 0:
                factors.append((p, cnt))
        
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                for i in range(e + 1):
                    new_divs.append(d * (p ** i))
            divs = new_divs
        
        divisor_cache[n] = divs
        return divs

    best_solution = None
    best_max = N  # single factor case [N]
    
    # Precompute factorization for quick lookup if needed, 
    # but get_divisors handles the structure.
    # We need to check if a candidate factor divides remaining.
    # Since we generate divisors of remaining, they all divide it.

    def dfs(remaining, last_factor, current_factors, depth, max_depth, current_max, rem_exp):
        nonlocal best_solution, best_max
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if depth == max_depth:
            if remaining == 1:
                if len(current_factors) == max_depth:
                    max_f = max(current_factors)
                    if max_f < best_max:
                        best_max = max_f
                        best_solution = current_factors[:]
            return
        
        if remaining == 1:
            # We have fewer factors than max_depth, but that's fine
            # Actually we need exactly max_depth factors
            return

        m_remaining = max_depth - depth
        
        # Pruning: if current_max is already >= best_max, no improvement possible
        # Also, the minimum possible max factor for the remaining m_remaining factors
        # is at least remaining^(1/m_remaining).
        min_possible_max = int(remaining ** (1.0 / m_remaining)) + 1
        if current_max >= best_max:
             # Even if we find perfect factors, the max will be at least current_max
             # which is not better than best_max
             # Wait, if current_max < best_max, we can still improve.
             # If current_max >= best_max, we can't improve the *maximum* because the max of the whole set
             # will be at least current_max.
             return

        # Upper bound on next factor
        # The next factor f must satisfy f * (f+1) * ... * (f + m_remaining - 2) <= remaining
        # Also f must be > last_factor
        # And f <= remaining^(1/m_remaining)
        
        upper_bound = int(remaining ** (1.0 / m_remaining))
        
        # Get divisors of remaining
        divs = get_divisors(remaining)
        
        # Filter divisors: > last_factor and <= upper_bound
        # Also, we need to ensure that the remaining product can be formed by m_remaining-1 factors
        # greater than f.
        
        candidates = []
        for f in divs:
            if f <= last_factor:
                continue
            if f > upper_bound:
                break # divs are sorted? get_divisors returns unsorted usually. Let's sort.
            
            # Check if f is a valid candidate
            # 1. f > last_factor
            # 2. f <= upper_bound
            # 3. remaining // f must be factorable into m_remaining-1 distinct factors > f
            #    Min product of m_remaining-1 distinct factors > f is (f+1)(f+2)...(f+m_remaining-1)
            
            rem_prod = remaining // f
            m_rem_next = m_remaining - 1
            
            if m_rem_next == 0:
                if rem_prod == 1:
                    candidates.append(f)
                continue
            
            # Check min product constraint
            min_prod = 1
            # The next factors must be distinct and > f
            # Smallest possible are f+1, f+2, ..., f+m_rem_next
            for j in range(1, m_rem_next + 1):
                min_prod *= (f + j)
                if min_prod > rem_prod:
                    break
            
            if min_prod <= rem_prod:
                candidates.append(f)
        
        if not candidates:
            return
        
        # Sort candidates. 
        # Heuristic: Try factors closer to the geometric mean first?
        # Or just increasing order? Increasing order ensures we find small max factors early?
        # Actually, if we pick a small f, the remaining factors must be larger, potentially increasing the max.
        # If we pick a large f (close to upper_bound), the remaining factors can be smaller?
        # No, factors must be distinct and > last_factor.
        # Let's try increasing order. It's standard for partition problems.
        candidates.sort()
        
        for f in candidates:
            if time.time() > deadline:
                return
            
            # Update state
            new_factors = current_factors + [f]
            new_rem = remaining // f
            
            # Update rem_exp
            # We can just pass remaining and let get_divisors handle it, 
            # but rem_exp is useful for quick checks if we had them.
            # Since we are generating divisors of new_rem, we don't strictly need rem_exp 
            # unless we want to prune based on prime availability before generating divisors.
            # Given the small size, generating divisors is fast.
            
            dfs(new_rem, f, new_factors, depth + 1, max_depth, max(current_max, f), rem_exp)
            
            # Optional: if we found a solution, we can break if we are confident 
            # no better solution exists for this branch? 
            # Hard to say without more info.

    # Try different numbers of factors, starting with more (likely to give smaller max)
    for k in range(MAX_FACTORS, 1, -1):
        if time.time() > deadline:
            break
        
        # Lower bound check: can we even have k factors?
        min_possible = 1
        for i in range(k):
            min_possible *= (13 + i)
        
        if min_possible > N:
            continue
        
        # Theoretical minimum max for k factors is at least N^(1/k)
        lower_bound = int(N ** (1.0 / k)) + 1
        
        # If we already have a solution with max <= lower_bound, we can't improve for this k
        if best_solution is not None and best_max <= lower_bound:
            break
        
        # Start the DFS with k factors
        # last_factor starts at 12 (so first factor > 12)
        dfs(N, 12, [], 0, k, 0, dict(prime_exp_N))
        
        # Early termination if we found a very good solution
        if best_solution and best_max <= lower_bound + 2:
            break
    
    if best_solution is None:
        return [N]
    
    return best_solution

# EVOLVE-BLOCK-END

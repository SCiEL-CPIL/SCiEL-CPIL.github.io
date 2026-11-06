import time
import math
from fractions import Fraction
from math import gcd

def construct():
    # Known best solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 55.0 # Safety margin for 60s limit
    
    # Precompute primes up to initial best_max_denom
    # We might need primes up to best_max_denom
    max_possible_limit = best_max_denom
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(max_possible_limit)
    
    # Precompute harmonic numbers for pruning?
    # H[i] = sum(1/k for k=1..i)
    # We need H[limit] - H[min_d-1]
    # limit changes, but max is 1736.
    H = [0.0] * (max_possible_limit + 1)
    for i in range(1, max_possible_limit + 1):
        H[i] = H[i-1] + 1.0/i

    current_limit = best_max_denom - 1
    
    def is_smooth(n, primes):
        """Check if n is smooth with respect to the given primes (i.e., all prime factors are in primes)."""
        if n == 1:
            return True
        temp = n
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                temp //= p
        return temp == 1

    def find_solution_with_limit(limit):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        """
        # Use integer arithmetic for speed
        # Target: 5/31
        # State: (num, den, min_d, current_set)
        
        def dfs(num, den, min_d, current_set):
            if time.time() - start_time > time_limit:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Pruning 1: Lower bound for next denominator d
            # 1/d <= num/den  => d >= den/num
            # d must be integer, so d >= ceil(den/num)
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Pruning 2: Harmonic Sum Bound
            # Max sum of reciprocals of distinct integers in [min_d, limit] is H[limit] - H[min_d-1]
            # If num/den > max_sum, impossible
            max_sum = H[limit] - (H[min_d-1] if min_d > 0 else 0.0)
            if num/den > max_sum:
                return None
            
            # Iterate possible denominators
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                next_num = num * d - den
                next_den = den * d
                
                if next_num == 0:
                    # Found a solution?
                    # 1/d = num/den => num*d = den
                    # This is a valid completion if distinctness holds (d > min_d-1 is ensured by loop)
                    return current_set + [d]
                
                if next_num < 0:
                    continue
                
                # Reduce fraction
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                # Modular/Smoothness Filter
                # If next_den has a prime factor > limit, it cannot be cancelled
                # because future denominators d' <= limit cannot be multiples of that prime.
                if not is_smooth(next_den, primes):
                    continue
                
                # Recurse
                res = dfs(next_num, next_den, d + 1, current_set + [d])
                if res is not None:
                    return res
                    
            return None
            
        # Start with remainder 5/31, min_denom 1
        return dfs(5, 31, 1, [])

    # Main Search Loop
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 1.0:
            break
            
        sol = find_solution_with_limit(current_limit)
        
        if sol is not None:
            # Found a solution with max <= current_limit
            # The actual max might be smaller
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                
            # Now try to find a solution with even smaller max
            # Set new limit to actual_max - 1
            current_limit = actual_max - 1
        else:
            # No solution found for this limit.
            # This implies no solution exists for any limit < current_limit.
            # So best_solution is optimal.
            break
            
    return best_solution

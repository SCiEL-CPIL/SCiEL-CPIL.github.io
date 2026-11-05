"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import isqrt

def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def _solve_subproblem(y_range, count=12, restarts=100):
    """
    Find 'count' distinct integers from y_range such that the set of 
    prime factors of {1 + y_i + y_j | i<j} is minimized.
    """
    best_primes = set(range(2, 200)) # Large initial set
    best_subset = None
    ys = list(y_range)
    
    for _ in range(restarts):
        # Greedy construction
        current_subset = []
        current_primes = set()
        available = ys[:]
        random.shuffle(available)
        
        # First pick a random start
        if available:
            current_subset.append(available.pop(0))
        
        while len(current_subset) < count and available:
            best_y = None
            best_new_primes = None
            best_count = float('inf')
            
            for y in available:
                new_primes = set(current_primes)
                for x in current_subset:
                    s = 1 + x + y
                    pf = _factorize(s)
                    new_primes.update(pf)
                
                if len(new_primes) < best_count:
                    best_count = len(new_primes)
                    best_y = y
                    best_new_primes = new_primes
            
            if best_y is None:
                break
                
            current_subset.append(best_y)
            current_primes = best_new_primes
            available.remove(best_y)
            
        if len(current_subset) == count:
            # Local search improvement
            improved = True
            while improved:
                improved = False
                for i in range(len(current_subset)):
                    for y in available:
                        old_y = current_subset[i]
                        # Try swapping old_y with y
                        test_subset = current_subset.copy()
                        test_subset[i] = y
                        
                        test_primes = set()
                        for a_idx in range(len(test_subset)):
                            for b_idx in range(a_idx + 1, len(test_subset)):
                                s = 1 + test_subset[a_idx] + test_subset[b_idx]
                                pf = _factorize(s)
                                test_primes.update(pf)
                        
                        if len(test_primes) < len(current_primes):
                            current_subset = test_subset
                            current_primes = test_primes
                            available.remove(y)
                            available.append(old_y)
                            improved = True
                            break
                    if improved:
                        break
            
            if len(current_primes) < len(best_primes):
                best_primes = current_primes
                best_subset = current_subset
                
    if best_subset is None:
        return None, None
    return best_subset, best_primes

def construct():
    """
    Find 12 distinct positive integers <= 10000 minimizing the number of distinct primes
    dividing sums of distinct-index pairs.
    Uses modular arithmetic reduction to a smaller combinatorial problem.
    """
    best_result = None
    best_prime_count = float('inf')
    
    # Try different moduli M where x_i = C + M*y_i
    # C is chosen such that 2*C is divisible by M (so sums are divisible by M)
    # M should be composed of small primes to keep the base prime count low.
    
    candidates_moduli = [
        (210, 105),   # Primes {2,3,5,7}
        (120, 60),    # Primes {2,3,5} (2^3*3*5)
        (60, 30),     # Primes {2,3,5}
        (30, 15),     # Primes {2,3,5}
        (420, 210),   # Primes {2,3,5,7}
        (180, 90),    # Primes {2,3,5}
    ]
    
    for M, C in candidates_moduli:
        # Max y such that C + M*y <= 10000
        max_y = (10000 - C) // M
        if max_y < 11: # Need at least 12 distinct y's (0 to 11)
            continue
            
        y_range = range(0, max_y + 1)
        subset, primes = _solve_subproblem(y_range, count=12, restarts=100)
        
        if subset:
            # Map back to x
            x_vals = [C + M * y for y in subset]
            # Verify distinctness and bounds (should be guaranteed by construction)
            if len(set(x_vals)) == 12 and all(1 <= x <= 10000 for x in x_vals):
                # Calculate actual prime count to be sure
                actual_primes = set()
                for i in range(12):
                    for j in range(i+1, 12):
                        actual_primes.update(_factorize(x_vals[i] + x_vals[j]))
                
                if len(actual_primes) < best_prime_count:
                    best_prime_count = len(actual_primes)
                    best_result = x_vals

    if best_result:
        return best_result
        
    # Fallback if all failed (should not happen)
    return list(range(1, 13))
# EVOLVE-BLOCK-END

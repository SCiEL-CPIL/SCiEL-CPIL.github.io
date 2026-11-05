"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import isqrt

def _is_smooth(n, primes):
    """Check if n is composed only of primes in the set primes."""
    if n <= 1:
        return True
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
    return n == 1

def construct():
    # We want to find 12 distinct positive integers <= 10000 such that
    # all pairwise sums have prime factors only from a small set S.
    
    candidate_prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]
    
    for primes in candidate_prime_sets:
        # Generate all S-smooth numbers up to 10000
        smooth_numbers = set()
        
        def gen_smooth(p_idx, current):
            if p_idx == len(primes):
                if 1 <= current <= 10000:
                    smooth_numbers.add(current)
                return
            p = primes[p_idx]
            val = 1
            while current * val <= 10000:
                gen_smooth(p_idx + 1, current * val)
                val *= p
        
        gen_smooth(0, 1)
        
        smooth_list = sorted(smooth_numbers)
        n_smooth = len(smooth_list)
        
        if n_smooth < 12:
            continue
            
        # Precompute compatibility matrix
        # compat[i][j] is True if smooth_list[i] + smooth_list[j] is S-smooth
        compat = [[False] * n_smooth for _ in range(n_smooth)]
        
        for i in range(n_smooth):
            for j in range(i + 1, n_smooth):
                s = smooth_list[i] + smooth_list[j]
                if _is_smooth(s, primes):
                    compat[i][j] = True
                    compat[j][i] = True
        
        # Find a clique of size 12
        solution_indices = []
        
        def backtrack(start_idx, current_indices):
            if len(current_indices) == 12:
                return True
            
            # Pruning: if not enough nodes left
            if len(current_indices) + (n_smooth - start_idx) < 12:
                return False
                
            for i in range(start_idx, n_smooth):
                # Check if node i is compatible with all in current_indices
                if all(compat[i][c] for c in current_indices):
                    current_indices.append(i)
                    if backtrack(i + 1, current_indices):
                        return True
                    current_indices.pop()
            return False
        
        if backtrack(0, solution_indices):
            return [smooth_list[i] for i in solution_indices]
            
    # Fallback: use the simple approach
    return list(range(1, 13))
# EVOLVE-BLOCK-END

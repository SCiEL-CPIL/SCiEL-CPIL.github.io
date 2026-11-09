"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _sieve_prime_factors(max_n):
    """Precompute distinct prime factors for all numbers up to max_n using sieve."""
    factors = [set() for _ in range(max_n + 1)]
    for d in range(2, max_n + 1):
        if not factors[d]:
            for m in range(d, max_n + 1, d):
                factors[m].add(d)
    return factors

def _count_distinct_primes(arr, pf):
    """Count distinct primes dividing all pairwise sums using precomputed factors."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= pf[s]
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute prime factors for all sums (max sum = 20000)
    pf = _sieve_prime_factors(20000)
    
    best_arr = None
    best_count = float('inf')
    
    # Candidate moduli M such that M/2 is an integer and M has small prime factors.
    # x_i = k_i * (M/2). Sums = (k_i+k_j)*(M/2).
    # Primes in sums = Primes(M/2) U Primes(k_i+k_j).
    # We want Primes(M/2) to be small and Primes(k_i+k_j) to be small.
    
    candidate_moduli = [210, 330, 420, 660, 1001, 1540, 2310, 3003, 5005, 6006, 770, 1155, 1617, 2730]
    
    for M in candidate_moduli:
        if time.time() > deadline - 5:
            break
            
        step = M // 2
        if step < 1:
            continue
            
        # Max k such that k*step <= 10000
        max_k = 10000 // step
        if max_k < 12:
            continue
            
        # Initialize k_arr with first 12 multiples
        k_arr = list(range(1, 13))
        
        # Local search on k_arr
        # Objective: minimize distinct primes in (k_i+k_j)*step
        # This is equivalent to minimizing distinct primes in k_i+k_j 
        # UNION distinct primes in step.
        
        primes_in_step = pf[step]
        
        def score_k(arr_k):
            primes = set(primes_in_step)
            for i in range(12):
                for j in range(i+1, 12):
                    s = arr_k[i] + arr_k[j]
                    primes |= pf[s]
            return len(primes)
        
        current_k = k_arr[:]
        current_score = score_k(current_k)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                # Try swapping with other k in range
                # Or just try random valid k
                # To be fast, try a few random k's
                for _ in range(20):
                    new_k = random.randint(1, max_k)
                    if new_k in current_k:
                        continue
                    old_k = current_k[i]
                    current_k[i] = new_k
                    new_score = score_k(current_k)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Check global best
                        x_arr = [k * step for k in current_k]
                        if current_score < best_count:
                            best_count = current_score
                            best_arr = x_arr
                            if best_count <= 4:
                                return best_arr
                    else:
                        current_k[i] = old_k
                        
    # If we found a good candidate, do a final polish on x_arr directly
    if best_arr is not None:
        # Convert to x_arr for final polish
        # The structure x_i = k_i * step is strong, but maybe we can tweak x_i slightly?
        # No, changing x_i breaks the divisibility. 
        # But we can try other structures.
        
        # Try a generic local search on the best_arr found
        # This might find a better set not constrained to a single modulus.
        arr = best_arr[:]
        count = best_count
        
        random.seed(42)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                old = arr[i]
                # Try a few random replacements
                for _ in range(10):
                    new_val = random.randint(1, 10000)
                    if new_val in arr:
                        continue
                    arr[i] = new_val
                    new_count = _count_distinct_primes(arr, pf)
                    if new_count < count:
                        count = new_count
                        improved = True
                        if count < best_count:
                            best_count = count
                            best_arr = arr[:]
                            if best_count <= 4:
                                return best_arr
                    else:
                        arr[i] = old
                        # If no improvement in this step, maybe try a neighborhood move
                        # But random is okay for now.
                
                # Also try small perturbations
                for delta in [-1, 1, -2, 2, -10, 10]:
                    new_val = old + delta
                    if 1 <= new_val <= 10000 and new_val not in arr:
                        arr[i] = new_val
                        new_count = _count_distinct_primes(arr, pf)
                        if new_count < count:
                            count = new_count
                            improved = True
                            if count < best_count:
                                best_count = count
                                best_arr = arr[:]
                                if best_count <= 4:
                                    return best_arr
                        else:
                            arr[i] = old

    # Fallback if nothing found (should not happen)
    if best_arr is None:
        best_arr = list(range(1, 13))
        
    return best_arr

# EVOLVE-BLOCK-END

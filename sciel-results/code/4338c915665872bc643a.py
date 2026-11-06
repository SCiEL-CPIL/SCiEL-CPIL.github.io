"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Precompute primes up to 10^6 for fast trial division
    # This is used for phi_large and can also help in verification
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve_primes(10**6)
    
    # Fast phi computation using numpy sieve
    def compute_phi_sieve_numpy(limit):
        """Compute phi for all numbers up to limit using numpy sieve."""
        phi = np.arange(limit + 1, dtype=np.int64)
        for p in primes:
            if p > limit:
                break
            # Vectorized update
            indices = np.arange(p, limit + 1, p)
            phi[indices] -= phi[indices] // p
        return phi
    
    # Hybrid phi for individual large numbers using precomputed primes
    def phi_large(x):
        """Compute phi for a single large number using trial division by primes."""
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    # Search for longest run of distinct phi values
    def find_best_run(phi_values, start_idx, end_idx, best_k, best_start):
        """Find the longest run of distinct phi values in the given range."""
        current_best_k = best_k
        current_best_start = best_start
        
        seen = set()
        run_start = start_idx
        run_len = 0
        
        for i in range(start_idx, min(end_idx, len(phi_values))):
            val = phi_values[i]
            if val not in seen:
                seen.add(val)
                run_len += 1
            else:
                # Reset
                if run_len > current_best_k:
                    current_best_k = run_len
                    current_best_start = run_start
                seen = set()
                run_start = i
                run_len = 1
                seen.add(val)
        
        # Check final run
        if run_len > current_best_k:
            current_best_k = run_len
            current_best_start = run_start
        
        return current_best_k, current_best_start

    best_k = 1
    best_start = 1
    
    # Phase 1: Sieve search
    # Try to sieve as large as possible
    # 10^7 takes ~1s with numpy, 10^8 takes ~5-10s
    sieve_sizes = [10**6, 10**7, 10**8]
    
    for size in sieve_sizes:
        if time_remaining() < 20.0:
            break
        
        try:
            phi_vals = compute_phi_sieve_numpy(size)
            
            # Convert to list for faster set operations in Python?
            # Or just iterate numpy array. Iteration over numpy is slow.
            # Let's convert to list if size is manageable, or use a different search method.
            # Actually, finding runs in a numpy array can be done with logic, but 
            # the "distinct" constraint requires a set.
            # Iterating a numpy array element by element in Python is slow.
            # However, 10^7 elements is ~10 million iterations. 
            # In Python, 10 million iterations takes ~0.5-1s.
            # So it is feasible.
            
            best_k, best_start = find_best_run(phi_vals, 1, size + 1, best_k, best_start)
            
            # Cap the length at 200 as required
            if best_k > 200:
                best_k = 200
                
        except MemoryError:
            continue
        except Exception:
            continue
    
    # Phase 2: Random search for larger numbers
    # Use the accelerated phi_large
    if time_remaining() > 5.0 and best_k < 200:
        random.seed(42)
        max_search = 10**12
        # We can afford ~50-100 checks now
        # Each check is 200 numbers * (trial division by ~78k primes worst case, but usually much less)
        # Average case for 10^12: factorization is fast.
        for _ in range(200):
            if time_remaining() < 2.0:
                break
            s = random.randint(1, max_search - 200)
            # Check a window of size 200
            seen = set()
            k = 0
            for i in range(200):
                if s + i > max_search:
                    break
                val = phi_large(s + i)
                if val not in seen:
                    seen.add(val)
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_start = s
                # If we find a very good one, we might stop or continue
                if best_k >= 50: # Heuristic: if we find a long run, it's likely near optimal
                    # But let's keep searching a bit more if time permits
                    pass

    # Ensure constraints are met
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    
    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END

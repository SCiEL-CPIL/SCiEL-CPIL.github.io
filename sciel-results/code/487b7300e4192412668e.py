"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Fast phi computation using sieve
    def compute_phi_sieve(limit):
        """Compute phi for all numbers up to limit using sieve."""
        phi = list(range(limit + 1))
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, limit + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    # Hybrid phi for individual large numbers
    def phi_large(x):
        """Compute phi for a single large number using trial division."""
        result = x
        d = 2
        y = x
        while d * d <= y:
            if y % d == 0:
                result -= result // d
                while y % d == 0:
                    y //= d
            d += 1
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
            if phi_values[i] not in seen:
                seen.add(phi_values[i])
                run_len += 1
            else:
                # Reset
                if run_len > current_best_k:
                    current_best_k = run_len
                    current_best_start = run_start
                seen = set()
                run_start = i
                run_len = 1
                seen.add(phi_values[i])
        
        # Check final run
        if run_len > current_best_k:
            current_best_k = run_len
            current_best_start = run_start
        
        return current_best_k, current_best_start
    
    # Try different sieve sizes
    sieve_sizes = [10**5, 10**6, 10**7, 10**8]
    best_k = 1
    best_start = 1
    
    for size in sieve_sizes:
        if time_remaining() < 10.0:
            break
        
        try:
            phi_vals = compute_phi_sieve(size)
            
            # Search for best run
            best_k, best_start = find_best_run(phi_vals, 1, size + 1, best_k, best_start)
            
            # Cap the length at 200 as required
            if best_k > 200:
                best_k = 200
        except MemoryError:
            continue
        except Exception:
            continue
    
    # If we have time, try some random large ranges
    if time_remaining() > 5.0 and best_k < 200:
        import random
        random.seed(42)
        max_search = 10**12
        for _ in range(100):
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
    
    # Ensure constraints are met
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    
    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END

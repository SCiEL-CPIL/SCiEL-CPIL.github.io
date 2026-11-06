"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_val = 1000024
    
    # Compute smooth part (23-smooth part) for all numbers up to max_val
    # smooth_part[m] = largest divisor of m composed only of primes < 24
    # Method: rough_part[m] = m with all small-prime factors removed
    #         smooth_part[m] = m // rough_part[m]
    
    # Initialize rough_part to identity
    rough = np.arange(max_val + 1, dtype=np.int64)
    
    # For each small prime, divide it out of all multiples (vectorized)
    for p in small_primes:
        # Get all multiples of p
        multiples = np.arange(p, max_val + 1, p)
        # Repeatedly divide by p while divisible
        changed = True
        while changed:
            vals = rough[multiples]
            mask = (vals % p) == 0
            if not np.any(mask):
                break
            # Divide only the divisible ones
            idx = multiples[mask]
            rough[idx] = rough[idx] // p
            changed = True
        # Check if we need another pass (for p^2, p^3, etc.)
        # The while loop above handles this since we keep going until no more divisible
    
    # Compute smooth_part = m // rough[m]
    smooth_part = np.zeros(max_val + 1, dtype=np.int64)
    smooth_part[1:] = np.arange(1, max_val + 1, dtype=np.int64) // rough[1:]
    
    # Search for n minimizing distinct smooth parts in window [n+1, n+24]
    best_n = 1000
    best_count = 999999
    
    # Phase 1: Coarse scan with stride to find promising regions
    stride = 50
    n_start = 1000
    n_end = 1000001
    
    for n in range(n_start, n_end, stride):
        if time.time() > deadline - 5:
            break
        window = smooth_part[n+1:n+25]
        # Count distinct values efficiently using sort
        sorted_w = np.sort(window)
        # Count distinct: number of positions where value changes + 1
        distinct_count = 1 + int(np.sum(sorted_w[1:] != sorted_w[:-1]))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count <= 1:
                break
    
    # Phase 2: Fine scan around the best region found
    if best_count > 1:
        # Scan a window around the best n found in phase 1
        fine_start = max(n_start, best_n - stride * 2)
        fine_end = min(n_end, best_n + stride * 2)
        
        for n in range(fine_start, fine_end):
            if time.time() > deadline - 2:
                break
            window = smooth_part[n+1:n+25]
            sorted_w = np.sort(window)
            distinct_count = 1 + int(np.sum(sorted_w[1:] != sorted_w[:-1]))
            
            if distinct_count < best_count:
                best_count = distinct_count
                best_n = n
                if best_count <= 1:
                    break
    
    # Phase 3: If time remains, do a broader fine scan
    if best_count > 1 and time.time() < deadline - 10:
        # Scan with smaller stride in the most promising region
        region_start = max(n_start, best_n - 500)
        region_end = min(n_end, best_n + 500)
        
        for n in range(region_start, region_end):
            if time.time() > deadline - 2:
                break
            window = smooth_part[n+1:n+25]
            sorted_w = np.sort(window)
            distinct_count = 1 + int(np.sum(sorted_w[1:] != sorted_w[:-1]))
            
            if distinct_count < best_count:
                best_count = distinct_count
                best_n = n
                if best_count <= 1:
                    break
    
    return best_n
# EVOLVE-BLOCK-END

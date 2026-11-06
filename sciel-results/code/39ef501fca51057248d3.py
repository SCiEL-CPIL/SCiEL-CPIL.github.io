"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def smooth_part(m):
        """Extract the factor of m composed entirely of primes < 24."""
        result = 1
        for p in small_primes:
            while m % p == 0:
                result *= p
                m //= p
        return result
    
    def count_distinct(n):
        """Count distinct smooth parts for n+1 ... n+24."""
        parts = set()
        for i in range(1, 25):
            parts.add(smooth_part(n + i))
        return len(parts)
    
    # First, quickly check the seed value
    seed_val = count_distinct(1000)
    
    best_n = 1000
    best_count = seed_val
    
    # Search strategy: 
    # 1) Quick scan of small ranges
    # 2) Targeted search looking for clusters where many numbers are "large-prime-only"
    
    # Strategy: For numbers to have smooth_part = 1, they must be coprime to 
    # 2*3*5*7*11*13*17*19*23 = 223092870
    # So we want n+1...n+24 to each be coprime to 223092870 (smooth part = 1)
    # or at least have very few distinct smooth parts.
    
    # Let's do a scan with increasing step sizes
    # First pass: scan every number from 1000 to some limit
    # But that's too slow for 1M range. Let's be smarter.
    
    # Key insight: smooth_part(m) = 1 iff m is coprime to 223092870.
    # The probability that a random number is coprime to 223092870 is 
    # prod_{p<24}(1-1/p) ≈ 0.17 or so.
    # So on average ~4 of the 24 numbers will have smooth_part=1.
    
    # For better results, we want numbers whose smooth parts are small (1, 2, 3, 4, 5, etc.)
    
    # Let's do a targeted search:
    # Phase 1: Scan 1000 to 50000 with step 1 (fast check)
    limit1 = 50000
    for n in range(1000, min(limit1, 1000000)):
        if time.time() > deadline:
            break
        c = count_distinct(n)
        if c < best_count:
            best_count = c
            best_n = n
            if best_count <= 3:
                break
    
    # Phase 2: If we found something good, try to improve further
    # Scan larger ranges with step 2 or 4 to cover more ground
    if time.time() < deadline - 10:
        step = 2
        for n in range(1000, 1000000, step):
            if time.time() > deadline:
                break
            c = count_distinct(n)
            if c < best_count:
                best_count = c
                best_n = n
                if best_count <= 2:
                    break
    
    # Phase 3: Fine-tune around the best found
    if time.time() < deadline - 5 and best_count > 2:
        search_range = 200
        for n in range(max(1000, best_n - search_range), min(1000000, best_n + search_range)):
            if time.time() > deadline:
                break
            c = count_distinct(n)
            if c < best_count:
                best_count = c
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END

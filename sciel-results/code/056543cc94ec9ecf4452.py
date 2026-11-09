"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    W = 24  # window size
    n_min = 1000
    n_max = 1000000
    # n+i for i in 1..24, so we need values from n_min+1 to n_max+24
    lo = n_min + 1
    hi = n_max + 24
    total_len = hi - lo + 1
    
    # Precompute for each position in [lo, hi], the "small-prime factor" value
    # For a number m, extract its complete factor composed of primes strictly below 24
    # i.e., remove all factors of primes >= 24 (which is 25+... wait, primes >= 24 means primes > 23)
    # Actually "primes STRICTLY BELOW 24" means primes in {2,3,5,7,11,13,17,19,23}
    # The "complete factor composed of these primes" is the largest divisor of m that only has these prime factors.
    # So for m, factor_small(m) = m / (m with all small-prime factors removed)
    # Equivalently, repeatedly divide by each small prime until not divisible.
    
    # We need to compute factor_small for each number in [lo, hi].
    # Then for each window of W=24 consecutive numbers (n+1, ..., n+24), count distinct values of factor_small.
    # We want to minimize this count.
    
    # Strategy: 
    # 1. Compute factor_small array for all numbers in [lo, hi]
    # 2. Use sliding window with frequency counting to find min distinct count
    
    # Step 1: Compute factor_small for each number
    # Use sieve-like approach: for each small prime p, divide out all factors of p from each number.
    # Actually, factor_small(m) is the part of m that is composed ONLY of small primes.
    # So factor_small(m) = product of p^e where p is small prime and e is the exponent of p in m.
    # This is m divided by the "large prime part" of m.
    
    # Let's compute it as: start with m, and for each small prime p, count how many times p divides m,
    # then factor_small = product of p^(count).
    # Alternatively: factor_small(m) = m / (m with small primes removed)
    # Let large_part[m] = m with all small prime factors removed. Then factor_small = m // large_part[m].
    
    # More efficient: for each number, we can compute the small-prime part directly.
    # But let's think about it differently.
    
    # Actually, the "complete factor composed of primes strictly below 24" for m is:
    # Let S = {2,3,5,7,11,13,17,19,23}
    # factor_small(m) = product over p in S of p^(v_p(m))
    # where v_p(m) is the exponent of p in m.
    # This equals m / (product over p not in S of p^(v_p(m)))
    # i.e., m divided by its part that has only prime factors not in S.
    
    # Let's compute this using a sieve approach:
    # Initialize small_part = np.arange(lo, hi+1) as float or int
    # For each small prime p, we don't need to do anything to small_part.
    # Instead, compute large_part: start with large_part = np.arange(lo, hi+1)
    # For each small prime p, divide out all factors of p from large_part.
    # Then factor_small = original / large_part.
    
    # But we need integer arithmetic. Let's use:
    # large_part[i] starts as lo+i
    # For each small prime p, for each multiple of p in the range, divide out all factors of p.
    
    # This is a bit complex. Let's just compute factor_small directly.
    # For each number m in [lo, hi], factor_small(m) = m with all large-prime factors removed.
    # Large primes are primes >= 24, i.e., 29, 31, 37, ...
    
    # Alternative approach: 
    # factor_small(m) can be computed by:
    # result = 1
    # for p in small_primes:
    #     while m % p == 0:
    #         result *= p
    #         m //= p
    # return result
    
    # But doing this for ~1 million numbers in Python would be slow.
    # Let's use NumPy vectorization.
    
    # Actually, let me reconsider. The number of distinct factor_small values in a window of 24
    # is what we're minimizing. The factor_small value for a number m is at most m itself,
    # but typically much smaller if m has large prime factors.
    
    # Key insight: if m is a large prime (>= 29), then factor_small(m) = 1.
    # If m = p * k where p is large prime and k has only small prime factors, factor_small(m) = k.
    
    # Let me just compute factor_small for each number using a vectorized approach.
    
    # Create array of numbers
    nums = np.arange(lo, hi + 1, dtype=np.int64)
    factor_small = nums.copy()
    
    # For each small prime, we need to... wait, I'm confusing myself.
    # factor_small(m) is the part of m that IS composed of small primes.
    # So I need to REMOVE the large prime factors from m.
    
    # Let me compute large_part(m) = m / factor_small(m) = the part of m with only large prime factors.
    # Then factor_small(m) = m // large_part(m).
    
    # To compute large_part: start with large_part = nums.copy()
    # For each small prime p, divide out all factors of p from large_part.
    # After processing all small primes, large_part contains only large prime factors.
    
    large_part = nums.copy()
    for p in primes:
        # For each multiple of p, divide out all factors of p
        # Find indices where large_part is divisible by p
        mask = (large_part % p == 0)
        while np.any(mask):
            large_part[mask] //= p
            mask = (large_part % p == 0)
    
    # Now factor_small = nums // large_part
    factor_small = nums // large_part
    
    # Step 2: Sliding window to find min distinct count
    # For each n in [n_min, n_max], the window is [n+1, n+24]
    # In our array, this corresponds to indices [n+1-lo, n+24-lo] = [n-n_min, n+24-n_min]
    # Let's define offset: index in factor_small array for value v is v - lo = v - (n_min+1)
    # So n+1 maps to index (n+1) - (n_min+1) = n - n_min
    # n+24 maps to index (n+24) - (n_min+1) = n + 24 - n_min - 1 = n + 23 - n_min
    
    # Window in factor_small: from index (n - n_min) to (n + 23 - n_min), inclusive. Size 24.
    
    # Use sliding window frequency count.
    # Since factor_small values can be large, we need a dict or sorted approach.
    # But for speed, let's use a different approach.
    
    # Actually, let's use a dict-based sliding window.
    from collections import defaultdict
    
    best_count = float('inf')
    best_n = n_min
    
    freq = defaultdict(int)
    # Initialize first window: n = n_min, indices 0 to 23
    for i in range(W):
        freq[factor_small[i]] += 1
    
    best_count = len(freq)
    best_n = n_min
    
    # Slide window
    for n in range(n_min + 1, n_max + 1):
        # Remove element at index (n-1 - n_min) = n - n_min - 1
        # Add element at index (n + 23 - n_min)
        out_idx = n - n_min - 1
        in_idx = n + 23 - n_min
        
        val_out = factor_small[out_idx]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
        
        val_in = factor_small[in_idx]
        freq[val_in] += 1
        
        current_count = len(freq)
        if current_count < best_count:
            best_count = current_count
            best_n = n
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END

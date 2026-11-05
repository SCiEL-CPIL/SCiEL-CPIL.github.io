"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START
def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            primes |= _factorize(s)
    return len(primes)

def _is_smooth(n, primes_set):
    """Check if n is smooth with respect to primes_set."""
    for p in primes_set:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Strategy: Find 12 numbers where all pairwise sums use only a small set of primes
    # Try different candidate prime sets and find valid combinations
    
    best_score = float('inf')
    best_nums = None
    
    # Candidate prime sets to try (small sets first)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Try sets of increasing size
    for set_size in range(2, 15):
        if time.time() > deadline:
            break
        # Try different combinations of small primes
        # Focus on sets that include 2 and 3 (most common in sums)
        for combo_idx in range(0, min(50, len(small_primes) - set_size + 1)):
            if time.time() > deadline:
                break
            
            # Generate a candidate prime set
            base_primes = small_primes[:combo_idx + set_size]
            primes_set = set(base_primes)
            primes_list = list(primes_set)
            
            # M = product of primes in set
            M = 1
            for p in primes_list:
                M *= p
            
            # For each prime p, we need a_i + a_j ≡ 0 (mod p) for all pairs
            # This means a_i ≡ -a_j (mod p) for all i ≠ j
            # So all a_i must be in the same residue class mod p, or come in pairs
            # Simplest: all a_i ≡ r (mod p) where 2r ≡ 0 (mod p), so r ≡ 0 (mod p) if p is odd
            # For p=2: r can be 0 or 1, need 2r ≡ 0 mod 2, always true
            
            # Try: all numbers ≡ 0 (mod M) → sums ≡ 0 (mod M), so all prime factors in S
            # Numbers: M, 2M, 3M, ..., 12M (all ≤ 10000)
            if M <= 10000 // 12:
                nums = [M * k for k in range(1, 13)]
                if max(nums) <= 10000:
                    score = _count_distinct_primes(nums)
                    if score < best_score:
                        best_score = score
                        best_nums = nums
            
            # Also try: numbers ≡ M/2 (mod M) if M is even
            # Then a_i + a_j ≡ M (mod M) ≡ 0 (mod M)
            if M % 2 == 0 and M // 2 >= 1:
                offset = M // 2
                nums = [offset + M * k for k in range(12)]
                if max(nums) <= 10000:
                    score = _count_distinct_primes(nums)
                    if score < best_score:
                        best_score = score
                        best_nums = nums
            
            # Try mixed: some ≡ 0, some ≡ M/2 (mod M)
            # Pairs: (0, M/2) → sum ≡ M/2 (mod M), need M/2 to be S-smooth
            # This is more complex, skip for now
            
            # Try: all ≡ r where r is chosen so 2r is S-smooth
            # For simplicity, try r = M//2 (if M even) or r = 0
            
        # Also try: pick 12 numbers all divisible by a large smooth number
        # e.g., all multiples of 60 = 2^2 * 3 * 5
        for smooth_base in [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]:
            if time.time() > deadline:
                break
            if smooth_base * 12 > 10000:
                continue
            nums = [smooth_base * k for k in range(1, 13)]
            score = _count_distinct_primes(nums)
            if score < best_score:
                best_score = score
                best_nums = nums
    
    # Fallback: return the best found or default
    if best_nums is not None:
        return best_nums
    return list(range(1, 13))
# EVOLVE-BLOCK-END

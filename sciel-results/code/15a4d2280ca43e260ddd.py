"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # We want to minimize the largest prime factor of n*(n+1) for n in [10000, 1000000]
    # This is equivalent to finding consecutive integers (n, n+1) that are both "smooth"
    # (i.e., composed of small prime factors).
    
    # Strategy:
    # 1. Generate primes up to a certain limit using a segmented sieve or simple sieve.
    # 2. For a given set of primes P, a number x is P-smooth if x / (product of p^k for p in P) == 1.
    #    Alternatively, we can iteratively divide by primes in P.
    # 3. We want to find n such that both n and n+1 are P-smooth for some small set P.
    #    The largest prime factor of n*(n+1) will then be max(max_prime(n), max_prime(n+1)).
    #    To minimize this, we want the largest prime in P to be as small as possible.
    
    # Let's try increasing sets of primes until we find a pair.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, ...
    
    # First, let's get a list of primes up to, say, 100.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes_100 = sieve(100)
    
    # For a given list of primes, check if a number is smooth (all prime factors in the list).
    # We also want to track the largest prime factor used.
    def largest_prime_factor_smooth(x, primes):
        """Returns the largest prime factor of x if all prime factors are in primes, else None."""
        if x < 2:
            return 1
        orig = x
        max_p = 1
        for p in primes:
            if p * p > x:
                break
            while x % p == 0:
                max_p = max(max_p, p)
                x //= p
        if x > 1:
            # x is a prime factor not in our list (or a prime > sqrt(orig))
            # We need to check if x is in our primes list
            # Actually, if x > 1 after dividing by all p where p*p <= current x, 
            # then x is prime. We need to ensure x is in our primes list.
            if x in primes:
                max_p = max(max_p, x)
            else:
                return None
        return max_p
    
    # To speed up, we can precompute smooth numbers in the range [10000, 1000000].
    # But the range is 1e6, which is manageable.
    # However, checking every number for every prime set might be slow.
    
    # Better approach:
    # Start with small prime sets.
    # For P = [2, 3], find n in [10000, 1000000] such that n and n+1 are 3-smooth.
    # For P = [2, 3, 5], find n such that n and n+1 are 5-smooth.
    # ...
    
    # Let's iterate through prime sets of increasing size.
    # For each set, scan the range [10000, 1000000] for consecutive smooth numbers.
    # We can optimize by generating smooth numbers directly.
    
    # Generating smooth numbers:
    # A number is P-smooth if it can be written as p1^a1 * p2^a2 * ... * pk^ak.
    # We can generate them using a heap (like ugly numbers) or recursive generation.
    # Given the range [10000, 1000000], the number of smooth numbers is small.
    
    # Let's write a function to generate all P-smooth numbers in [lo, hi].
    def generate_smooth_numbers(primes, lo, hi):
        """Generate all numbers in [lo, hi] whose prime factors are all in primes."""
        if not primes:
            return []
        
        smooth = set()
        # Recursive generation
        def gen(idx, current):
            if current > hi:
                return
            if current >= lo:
                smooth.add(current)
            if idx >= len(primes):
                return
            p = primes[idx]
            # Multiply by p repeatedly
            val = current
            while val <= hi:
                gen(idx + 1, val)
                val *= p
                if val == 0:  # overflow guard
                    break
            # Also try skipping p
            gen(idx + 1, current)
        
        # Start with 1
        gen(0, 1)
        return sorted(smooth)
    
    # However, recursive generation might be slow or hit recursion limits.
    # Let's use a different approach: iterate through the range and check smoothness.
    # 1e6 numbers, and for each, we divide by primes. This is fast enough.
    
    best_lpf = float('inf')
    best_n = 10000
    
    # We'll try prime sets of increasing size.
    # For each set, we check if there's a pair (n, n+1) both smooth.
    # The largest prime factor will be max(LPF(n), LPF(n+1)).
    # Since we want to minimize this, we should try smaller prime sets first.
    # If we find a pair with max prime factor M, we can stop if M is small enough?
    # No, we should keep searching for even smaller M.
    # But M is bounded by the largest prime in the set.
    # So if we use primes up to 7, the max LPF is 7.
    # If we find such a pair, that's likely optimal (or close to it).
    
    # Let's try prime sets:
    # [2], [2,3], [2,3,5], [2,3,5,7], [2,3,5,7,11], etc.
    
    # Note: n and n+1 are coprime, so they cannot share any prime factors.
    # This means one must be a power of 2, the other a power of 3? No.
    # If P=[2,3], one number is 2^a, the other is 3^b.
    # 2^a and 3^b differ by 1.
    # 2^1=2, 3^0=1 (diff 1) -> n=1 (out of range)
    # 2^2=4, 3^1=3 (diff 1) -> n=3 (out of range)
    # 2^3=8, 3^2=9 (diff 1) -> n=8 (out of range)
    # 2^a and 3^b for larger a,b: Catalan's conjecture (Mihăilescu's theorem) says the only consecutive perfect powers are 8 and 9.
    # So for P=[2,3], there are no solutions in [10000, 1000000].
    
    # For P=[2,3,5]:
    # One number is 2^a * 5^b, the other is 3^c.
    # Or 2^a * 3^b and 5^c.
    # Etc.
    
    # Let's just scan the range for each prime set.
    
    # To speed up, we can precompute the smoothness for each number in the range
    # for the largest prime set we might use. But the prime set changes.
    
    # Alternative: For each n in [10000, 1000000], compute its prime factors.
    # Then for n and n+1, find the smallest set of primes that covers all factors of both.
    # The largest prime in that set is the LPF of n*(n+1).
    # We want to minimize this.
    
    # This is equivalent to: for each n, compute LPF(n) and LPF(n+1).
    # The LPF of n*(n+1) is max(LPF(n), LPF(n+1)).
    # We want to minimize max(LPF(n), LPF(n+1)).
    
    # So we just need to compute LPF for all numbers in [10000, 1000001].
    # Then scan for n in [10000, 1000000] minimizing max(LPF[n], LPF[n+1]).
    
    # This is much simpler!
    
    # Compute LPF for all numbers up to 1000001.
    max_val = 1000001
    lpf = [0] * (max_val + 1)
    lpf[1] = 1
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
    
    # Now scan for best n
    best_val = float('inf')
    best_n = 10000
    
    for n in range(10000, 1000001):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
        # Early termination: if best_val is 2, we can't do better (min possible is 2)
        if best_val == 2:
            break
    
    return best_n

# EVOLVE-BLOCK-END

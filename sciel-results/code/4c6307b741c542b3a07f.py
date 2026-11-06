import math
import numpy as np

def construct():
       N = 1000000
       MAX_VAL = N + 1
       
       # 1. Sieve primes up to 1000
       limit = int(math.isqrt(MAX_VAL))
       is_prime = np.ones(limit + 1, dtype=bool)
       is_prime[0] = False
       is_prime[1] = False
       for i in range(2, int(math.isqrt(limit)) + 1):
           if is_prime[i]:
               is_prime[i*i : limit+1 : i] = False
       primes = np.where(is_prime)[0]
       
       # 2. Compute lpf_small (largest prime factor <= 1000)
       lpf_small = np.zeros(MAX_VAL + 1, dtype=np.int64)
       for p in primes:
           lpf_small[p : MAX_VAL+1 : p] = p
           
       # 3. Compute rem (largest prime factor > 1000, or 1)
       rem = np.arange(1, MAX_VAL + 2, dtype=np.int64)
       # rem[0] is unused, rem[x] initially x
       
       for p in primes:
           # For each multiple of p, divide out p as much as possible
           # We only need to do this if p <= 1000
           # To optimize, we can iterate multiples
           # But the while loop inside makes vectorization hard.
           # However, the number of operations is small.
           # Let's use a pure Python loop for the division part?
           # Or NumPy?
           # NumPy doesn't have a vectorized "divide while divisible" easily without masking which is slow.
           # Given N=1e6, a Python loop might be okay if optimized, or we can use the fact that
           # most numbers are not divisible by large primes.
           
           # Actually, let's just do it in Python for clarity and safety,
           # but 1.5M operations is fine.
           pass
       
       # Wait, I can just do the division in a Python loop over primes,
       # and for each prime, loop over its multiples.
       
       # Let's refine the `rem` calculation.
       # rem = np.arange(MAX_VAL + 1, dtype=np.int64)
       # rem[0] = 0
       
       # for p in primes:
       #     for j in range(p, MAX_VAL+1, p):
       #         while rem[j] % p == 0:
       #             rem[j] //= p
       
       # This nested loop in Python:
       # Outer: 168 primes.
       # Inner: sum(N/p) ~ 1.5M iterations.
       # Inside inner: average 1-2 divisions.
       # Total ~ 3M Python loop iterations.
       # This might take 1-2 seconds.
       # Acceptable.
       
       # Let's implement this.
       
       rem = np.arange(MAX_VAL + 1, dtype=np.int64)
       rem[0] = 0
       
       for p in primes:
           for j in range(p, MAX_VAL+1, p):
               while rem[j] % p == 0:
                   rem[j] //= p
                   
       # Now lpf(x) = max(lpf_small[x], rem[x])
       # But rem[x] is either 1 or a prime > 1000.
       # lpf_small[x] is largest prime <= 1000, or 0.
       
       # We need this for x in [10000, 1000001]
       start = 10000
       end = 1000001
       
       lpf_small_slice = lpf_small[start : end+1]
       rem_slice = rem[start : end+1]
       
       # If lpf_small is 0, it means no small prime factors.
       # If rem is 1, no large prime factors.
       # lpf = max(lpf_small, rem)
       # Note: if lpf_small is 0 and rem is 1, x=1. But x>=10000.
       # If x is prime > 1000, lpf_small=0, rem=x. max(0, x) = x. Correct.
       # If x is composite with small factors, lpf_small>0, rem=1. max(lpf_small, 1) = lpf_small. Correct.
       # If x has both, max works.
       
       lpf_all = np.maximum(lpf_small_slice, rem_slice)
       
       # Now find n in [10000, 1000000] minimizing max(lpf(n), lpf(n+1))
       # lpf_all is for x in [10000, 1000001]
       # lpf(n) is lpf_all[0 : 1000000-10000 + 1] -> indices 0 to 990000
       # lpf(n+1) is lpf_all[1 : 1000001-10000 + 1] -> indices 1 to 990001
       
       lpf_n = lpf_all[0 : end - start] # length 1000000 - 10000 + 1 = 990001?
       # Range of n: 10000 to 1000000. Count = 990001.
       # lpf(n) corresponds to lpf_all[0 : 990001]
       # lpf(n+1) corresponds to lpf_all[1 : 990002]
       
       lpf_n = lpf_all[: end - start]
       lpf_n1 = lpf_all[1 : end - start + 1]
       
       max_lpf = np.maximum(lpf_n, lpf_n1)
       
       min_idx = np.argmin(max_lpf)
       best_n = start + min_idx
       
       return int(best_n)

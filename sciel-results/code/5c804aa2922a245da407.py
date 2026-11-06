import time
import math

def construct():
    """
    Find the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: 5 seconds before the 90s limit
    
    # We need to find the smallest n such that phi(n) % 120 == 0.
    # phi(1) = 1, phi(2) = 1, phi(3) = 2, ...
    # Let's compute phi for small numbers and check.
    # We can use a simple sieve for phi up to a certain limit, or just iterate and compute phi(n) on the fly.
    # Since we want the MINIMUM n, we should check n = 1, 2, 3, ... in order.
    # Computing phi(n) individually for each n might be slow if n is large, but the answer is likely small.
    # Let's estimate: phi(n) is even for n > 2.
    # 120 = 2^3 * 3 * 5.
    # We need phi(n) to be a multiple of 8, 3, and 5.
    
    # Let's write a helper to compute phi(n) efficiently for a single n.
    def phi_single(n):
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result
    
    # Iterate n from 1 upwards
    # We can optimize by checking if n is likely to have a small phi.
    # Actually, just iterating from 1 is fine if the answer is small.
    # Let's try a few manually first to see where it might be.
    # phi(1)=1, 2=1, 3=2, 4=2, 5=4, 6=2, 7=6, 8=4, 9=6, 10=4, 11=10, 12=4, 13=12, 14=6, 15=8, 16=8, 17=16, 18=6, 19=18, 20=8
    # phi(21)=12, 22=10, 23=22, 24=8, 25=20, 26=12, 27=18, 28=12, 29=28, 30=8
    # phi(31)=30, 32=16, 33=20, 34=16, 35=24, 36=12, 37=36, 38=18, 39=24, 40=16
    # ...
    # To get phi(n) divisible by 120, n must be fairly large.
    # If n is prime, phi(n) = n-1. So n-1 = 120k. n = 120k+1.
    # Smallest prime of form 120k+1:
    # k=1: 121 (not prime)
    # k=2: 241 (prime). phi(241) = 240. 240 % 120 == 0.
    # So n=241 is a candidate.
    
    # Can we find a smaller n?
    # n must be composite.
    # Let's check numbers around 200-240.
    # phi(n) = n * product(1 - 1/p)
    # For n < 241, phi(n) < 240.
    # So phi(n) can be 120 only (since it must be a multiple of 120 and positive).
    # So we are looking for n < 241 such that phi(n) = 120.
    
    # Let's check specific candidates.
    # If phi(n) = 120, then n is a solution.
    # We can generate candidates by inverting phi, or just check n from 1 to 240.
    # Given the small range, computing phi(n) for each n is trivial.
    
    for n in range(1, 241):
        if time.time() > deadline:
            return 241  # Fallback to the prime candidate we already identified
        
        val = phi_single(n)
        if val % 120 == 0:
            return n
            
    # If no n < 241 found, return 241
    return 241

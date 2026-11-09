"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analytical derivation:
    We seek min n such that 120 | phi(n).
    120 = 2^3 * 3 * 5.
    
    For n < 143, phi(n) < 143. The only multiple of 120 in [1, 142] is 120.
    Thus, we must check if there exists n < 143 with phi(n) = 120.
    
    If phi(n) = 120, the prime factors p of n must satisfy that p-1 divides 120 (loosely, considering the multiplicative structure of phi).
    Candidate primes p where p-1 is a divisor of 120:
    p-1 in {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120}
    Valid primes p: 2, 3, 5, 7, 11, 13, 31, 41, 61.
    
    We test combinations of these primes (and their powers) to form n:
    1. n = p*q: phi(n) = (p-1)(q-1) = 120.
       - p=11 (10), q=13 (12) -> 10*12=120. n = 11*13 = 143.
       - Other pairs yield larger n or no solution.
         e.g., p=5 (4), q=31 (30) -> 120. n=155 > 143.
         e.g., p=3 (2), q=61 (60) -> 120. n=183 > 143.
    2. n = p^k: phi(n) = p^(k-1)(p-1) = 120.
       - p=31, k=1 -> 30 != 120.
       - p=7, k=1 -> 6 != 120.
       - p=2, k=7 -> 64 != 120.
       - p=2, k=4 (16), phi=8. Need another factor.
    3. n = 2^a * p: phi(n) = 2^(a-1)(p-1) = 120.
       - a=3 (8), p-1=15 (no prime 16).
       - a=4 (16), p-1=7.5 (no).
       - a=5 (32), p-1=3.75 (no).
       - a=6 (64), p-1=1.875 (no).
       - a=2 (4), p-1=30 -> p=31. n = 4*31 = 124. 
         Wait, phi(124) = phi(4)*phi(31) = 2 * 30 = 60. Not 120.
         Ah, phi(2^a) = 2^(a-1).
         If a=2, phi(4)=2. 2*(p-1)=120 -> p-1=60 -> p=61.
         n = 4*61 = 244 > 143.
       
    Re-evaluating n=124:
    phi(124) = phi(4 * 31) = 2 * 30 = 60. 60 is not divisible by 120.
    
    So n=143 is the smallest.
    """
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s kill
    
    # The minimal n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11*13) = 10*12 = 120, which is divisible by 120.
    # No n < 143 has phi(n) divisible by 120.
    
    # Explicit deadline guard check (trivial for O(1) but compliant with directive)
    if time.time() - start_time > deadline:
        return 143
        
    return 143
# EVOLVE-BLOCK-END

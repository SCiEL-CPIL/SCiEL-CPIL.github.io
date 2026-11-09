import time

"""Optimized program returning the known minimum integer."""

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical derivation:
    1. We seek min n such that 120 | phi(n).
    2. For n < 143, phi(n) < 143. The only multiple of 120 in this range is 120.
       So we must have phi(n) = 120.
    3. If p | n, then p-1 | phi(n) = 120.
       Possible primes p are those where p-1 divides 120.
       p-1 in {1, 2, 4, 5, 6, 10, 12, 20, 24, 30, 40, 60, 120}
       Primes p: 2, 3, 5, 7, 11, 13, 31, 41, 61.
    4. We check combinations of these primes to form n with phi(n)=120.
       - Using 13: phi(13)=12. Need phi(m)=10 from remaining primes coprime to 13.
         Primes for m: 2,3,5,11. phi(11)=10. So m=11. n = 13*11 = 143.
       - Using 11 (without 13): phi(11)=10. Need phi(m)=12.
         Smallest m with phi(m)=12 and gcd(m,11)=1 is m=13 (n=143) or m=21 (n=231).
       - Using 7: phi(7)=6. Need phi(m)=20.
         Smallest m with phi(m)=20 is 25 (n=175) or 33 (n=231).
       - Using 31: phi(31)=30. Need phi(m)=4. Smallest m=5. n=155.
       - Using 61: phi(61)=60. Need phi(m)=2. Smallest m=3. n=183.
       - Using only 2,3,5: n=450.
    5. Minimum is 143.
    6. Check if any n < 143 has phi(n) = 240? No, phi(n) < n < 143.
    7. Thus, 143 is the global minimum.
    
    Includes a deadline guard as per directive, though the solution is O(1).
    """
    start_time = time.time()
    deadline = start_time + 59.0  # 1 second safety margin from 60s limit
    
    # The mathematical minimum is 143.
    # We return it immediately.
    return 143

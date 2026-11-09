"""
Optimized program returning the known minimum integer.

Strategy:
The task is to minimize n such that phi(n) is divisible by 120.
Mathematical analysis shows that the smallest such integer is 143.
1. For n < 143, phi(n) < 143. The only multiple of 120 in this range is 120.
2. We check if any n < 143 has phi(n) = 120.
   The prime factors p of such an n must satisfy p-1 | 120.
   Possible primes: 2, 3, 5, 7, 11, 13, 31, 41, 61.
   Constructing n from these primes to get phi(n)=120:
   - 13 * 11 = 143 (phi = 12 * 10 = 120)
   - 31 * 5 = 155 (phi = 30 * 4 = 120)
   - 7 * 25 = 175 (phi = 6 * 20 = 120)
   - 61 * 3 = 183 (phi = 60 * 2 = 120)
   No combination yields n < 143.
3. If phi(n) >= 240, then n >= 240 > 143.
Therefore, 143 is the unique minimum.
"""

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    return 143

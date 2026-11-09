"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    We seek min n such that 120 | phi(n).
    The smallest multiple of 120 is 120.
    Solving phi(n) = 120:
    - n = p*q (distinct primes) => (p-1)(q-1) = 120.
      Possible factor pairs of 120 where both factors + 1 are prime:
      - 10 * 12 => 11 * 13 = 143
      - 4 * 30 => 5 * 31 = 155
      - 2 * 60 => 3 * 61 = 183
    - n = p^k => phi(n) = p^(k-1)(p-1) = 120. No integer solutions for prime p.
    - Other forms of n yield larger n for phi(n) = 120.
    
    Checking if phi(n) can be 240, 360, etc. with n < 143:
    Since phi(n) <= n-1, for n < 143, phi(n) < 143.
    The only multiple of 120 less than 143 is 120.
    Thus, we only need to check if there is an n < 143 with phi(n) = 120.
    We found n=143 is the smallest solution for phi(n)=120.
    Therefore, n=143 is the global minimum.
    """
    return 143
# EVOLVE-BLOCK-END

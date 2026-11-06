def construct():
    """Return the smallest n in [1, 1000000] such that phi(n) is divisible by 120."""
    # Analytical derivation:
    # 1. phi(n) must be divisible by 120 = 2^3 * 3 * 5.
    # 2. Divisibility by 5 in phi(n) implies n has a prime factor p == 1 (mod 5) or 25 divides n.
    #    Smallest prime p == 1 (mod 5) is 11.
    # 3. Divisibility by 3 in phi(n) implies n has a prime factor q == 1 (mod 3) or 9 divides n.
    #    Smallest prime q == 1 (mod 3) is 7.
    # 4. Divisibility by 8 in phi(n) is satisfied by the product of (p-1) and (q-1) terms or powers of 2.
    
    # Case A: 25 divides n.
    # n = 25 * m. phi(25) = 20. Need 6 | phi(m).
    # Smallest m with phi(m) % 6 == 0 is m=7 (phi(7)=6).
    # n = 25 * 7 = 175.
    
    # Case B: 25 does not divide n.
    # Must have prime p == 1 (mod 5). Smallest is 11. So 11 | n.
    # Subcase B1: 9 divides n.
    # n = 99 * k. phi(99) = 60. Need 2 | phi(k).
    # Smallest k with phi(k) % 2 == 0 is k=3 (phi(3)=2).
    # n = 99 * 3 = 297.
    
    # Subcase B2: 9 does not divide n.
    # Must have prime q == 1 (mod 3). Smallest is 7.
    # If q=7, n is multiple of 77. phi(77) = 60. Need 2 | phi(k).
    # Smallest k=3. n = 77 * 3 = 231.
    # If q=13, n is multiple of 143. phi(143) = 120. Need 1 | phi(k).
    # Smallest k=1. n = 143 * 1 = 143.
    
    # Comparing candidates: 175, 297, 231, 143.
    # Minimum is 143.
    
    return 143

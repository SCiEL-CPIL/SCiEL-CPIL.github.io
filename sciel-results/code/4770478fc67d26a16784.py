"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimum is 143 = 11 * 13.
    phi(143) = phi(11) * phi(13) = 10 * 12 = 120.
    
    Minimality check:
    - phi(n) divisible by 5 implies n has a prime factor p ≡ 1 (mod 5) or 25|n.
    - Smallest p ≡ 1 (mod 5) is 11. phi(11)=10. Need phi(n)/10 divisible by 12.
      Smallest k coprime to 11 with phi(k) % 12 == 0 is 13 (phi(13)=12).
      n = 11 * 13 = 143.
    - Next candidate with p=31: phi(31)=30. Need phi(k) % 4 == 0. Smallest k=5.
      n = 31 * 5 = 155 > 143.
    - Candidate with 25|n: phi(25)=20. Need phi(k) % 6 == 0. Smallest k=7.
      n = 25 * 7 = 175 > 143.
    
    Thus, 143 is the global minimum.
    """
    return 143
# EVOLVE-BLOCK-END

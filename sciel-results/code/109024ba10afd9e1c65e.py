"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # All multiples of 3 up to 255: 3, 6, 9, ..., 255
    # Count: 255 // 3 = 85 elements
    # All pairs share factor 3, so gcd > 1.
    # This is optimal because any two numbers in the set must share a common factor.
    # Since 255 = 3*5*17, the largest prime factor is 17.
    # Multiples of 3 give us 85 elements.
    # Multiples of 5 give us 51 elements.
    # Multiples of 17 give us 15 elements.
    # Any mixed strategy would be constrained by pairwise gcd > 1.
    # The maximum is achieved by taking all multiples of the smallest prime factor of 255, which is 3.
    
    best = list(range(3, 256, 3))
    
    # Verify correctness
    assert 255 in best
    assert all(1 <= x <= 255 for x in best)
    assert len(best) > 0
    for i in range(len(best)):
        for j in range(i+1, len(best)):
            assert gcd(best[i], best[j]) > 1
    
    return best
# EVOLVE-BLOCK-END

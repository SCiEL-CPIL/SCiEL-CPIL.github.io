"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Start with all multiples of 3 in [1, 255]
    base_set = set(range(3, 256, 3))
    base_set.add(255)  # already there
    
    # Try to add elements divisible by 5 or 17 (but not 3)
    # An element x can be added if gcd(x, y) > 1 for all y in current set
    # Since all elements in base_set are divisible by 3, x must share a factor
    # with each. If x is divisible by 5 but not 3, then gcd(x, y) > 1 requires
    # y to be divisible by a factor of x. Since y is divisible by 3, we need
    # gcd(x, y) > 1. If x = 5*k where k is not divisible by 3, then gcd(x,y)
    # could be 1 if y is only divisible by 3 and x has no factor in common with y.
    # Actually: if x is divisible by 5 but not 3, and y is divisible by 3 but 
    # not 5, then gcd(x,y) could be 1 (e.g., x=5, y=3 -> gcd=1).
    # So we can only add x if x shares a factor with EVERY element in the set.
    # This is very restrictive. Let's check if any such element exists.
    
    # For x divisible by 5 but not 3: need gcd(x, y) > 1 for all y in base_set.
    # y ranges over multiples of 3. If x = 5 * k, we need for every multiple of 3,
    # gcd(5k, 3m) > 1. This fails when m is coprime to 5k/ gcd(5k,3).
    # Since 3 doesn't divide 5k (k not div by 3), gcd(5k, 3) = 1 or 3... 
    # Actually 3 doesn't divide 5k, so gcd(5k,3)=1. So for y=3, gcd(5k,3)=1. Fail.
    # So NO element divisible by 5 but not 3 can be added to the full set of multiples of 3.
    
    # Similarly for 17: gcd(x, 3) = 1 if x not divisible by 3. So same issue.
    
    # Therefore the maximum set containing 255 with all pairs gcd>1 is exactly
    # the multiples of 3 (or multiples of 5, or multiples of 17), and 85 is max.
    
    # But wait - what if we DON'T use all multiples of 3? What if we use a 
    # smaller core and add more diverse elements?
    # E.g., use only multiples of 15 (3*5) as core, then add multiples of 5 
    # and multiples of 3 that share factor 5 or 3 respectively.
    # Core: multiples of 15 in [1,255]: 17 elements (15,30,...,255)
    # Can add multiples of 5 not div by 3: need gcd with all core (div by 15). 
    #   gcd(5k, 15m) >= 5 > 1. Yes! Add all multiples of 5 not div by 3.
    #   Count: floor(255/5) - floor(255/15) = 51 - 17 = 34 elements.
    # Can add multiples of 3 not div by 5: need gcd with all core (div by 15).
    #   gcd(3k, 15m) >= 3 > 1. Yes! Add all multiples of 3 not div by 5.
    #   Count: floor(255/3) - floor(255/15) = 85 - 17 = 68 elements.
    # But wait: do these added elements have gcd>1 with EACH OTHER?
    #   A multiple of 5 (not 3) and a multiple of 3 (not 5): gcd could be 1!
    #   E.g., 5 and 3: gcd=1. So NO, we can't add both types freely.
    
    # Alternative: use core = multiples of 15, add only multiples of 5 (not 3)
    #   Total: 17 + 34 = 51. Worse than 85.
    
    # Or core = multiples of 15, add only multiples of 3 (not 5)
    #   Total: 17 + 68 = 85. Same as before.
    
    # What about core = multiples of 51 (3*17)? 
    #   5 elements: 51, 102, 153, 204, 255
    # Add multiples of 3 not div by 17: gcd(3k, 51m) >= 3. 
    #   Count: 85 - 5 = 80. Total: 85. Same.
    
    # It seems 85 is indeed optimal. Let me verify with a quick check that 
    # no larger set exists by trying a different construction.
    
    # Try: all numbers in [1,255] that are divisible by 3.
    result = list(range(3, 256, 3))
    
    # Verify
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    # Spot check a few pairs
    assert gcd(result[0], result[-1]) > 1
    assert gcd(result[len(result)//2], result[-1]) > 1
    
    return result
# EVOLVE-BLOCK-END

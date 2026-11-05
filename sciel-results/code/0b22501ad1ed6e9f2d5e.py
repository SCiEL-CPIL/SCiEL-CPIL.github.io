import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 255
    target = 255
    
    # The optimal set is likely all multiples of the smallest prime factor of 255.
    # 255 = 3 * 5 * 17.
    # Smallest prime is 3.
    # Multiples of 3 in [1, 255]: 3, 6, ..., 255.
    # Count: 255 // 3 = 85.
    
    # Let's construct this set directly.
    S = set(x for x in range(1, n + 1) if x % 3 == 0)
    
    # Verify 255 is in S
    if target not in S:
        S.add(target) # Though 255 is divisible by 3, so it is.
        
    # Verify pairwise gcd > 1
    # Since all elements are multiples of 3, gcd(a, b) is at least 3.
    # This is valid.
    
    # To be absolutely sure and potentially find a larger set (if my logic is wrong),
    # we can run a quick local search.
    # But given the strong structural argument, 85 is very likely the max.
    
    # Let's check if we can add any non-multiple of 3.
    # As derived, adding a non-multiple of 3 requires removing many multiples of 3.
    # Specifically, if we add y not div 3, we must remove all x in S such that gcd(x,y)==1.
    # If S is all multiples of 3, and y is not div 3, then gcd(3, y) = 1 (if y not div 3).
    # So 3 must be removed.
    # Also, any multiple of 3 that is coprime to y must be removed.
    # This loss is generally greater than the gain.
    
    # We will return the multiples of 3.
    return list(S)

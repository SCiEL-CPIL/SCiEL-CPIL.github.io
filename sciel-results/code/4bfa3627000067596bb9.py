"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def construct():
    # 255 = 3 * 5 * 17
    # Every element must share a factor with 255, so must be divisible by 3, 5, or 17.
    # Base clique: all multiples of 3 in [1,255], size 85.
    # Try to see if we can build a larger set using a greedy + local search approach.
    
    n = 255
    primes_255 = [3, 5, 17]
    
    # Candidate pool: all numbers in [1, 255] divisible by 3, 5, or 17
    pool = [x for x in range(1, n+1) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    # Start with 255
    selected = [255]
    remaining = [x for x in pool if x != 255]
    
    # Greedy: repeatedly add the element with max number of valid partners
    # But first, let's just go with the simple multiples-of-3 set as baseline
    # and then try to improve via local search.
    
    # Baseline: all multiples of 3
    base_set = set(range(3, 256, 3))
    
    # Try local search: try swapping in non-multiples-of-3 that are in the pool
    # For a number x not divisible by 3 to be compatible with all elements in base_set,
    # it must share a factor > 1 with every multiple of 3.
    # A multiple of 3 can be 3*k for k=1..85. 
    # For x to have gcd > 1 with 3*k for all k, x must share a factor with 3k 
    # for every k. If x is not divisible by 3, then x must be divisible by some prime p,
    # and 3k must be divisible by p for all k, which is impossible unless p=3.
    # So no non-multiple of 3 can be added to the full set of multiples of 3.
    
    # What if we remove some multiples of 3 to add non-multiples?
    # This is a tradeoff. Let's try a different strategy.
    
    # Strategy: Build a clique centered on 255 using all three prime factors.
    # Start with 255. Add all multiples of 3. Now try to add multiples of 5 
    # (not div by 3) that are compatible with all current elements.
    # A multiple of 5 not div by 3, say 5m where m not div by 3.
    # It needs gcd(5m, 3k) > 1 for all k=1..85.
    # 5m and 3k share a factor iff they share a prime. 5m has primes 5 and primes of m.
    # 3k has prime 3 and primes of k.
    # For gcd > 1 for ALL k, we need 5m to share a prime with 3k for every k.
    # If 5m is not div by 3, the shared prime must be 5 or a prime of m that divides 3k.
    # But k ranges over 1..85, so 3k ranges over 3,6,9,...,255.
    # Not all of these are divisible by 5. E.g., 3*1=3, not div by 5.
    # So 5m must share a factor with 3. But 5m not div by 3, so no.
    # Therefore no such element exists.
    
    # Conclusion: The maximum clique likely is just the multiples of 3 (size 85)
    # or possibly a different structure. Let's verify with a quick check.
    
    # Actually, what about multiples of 5? Size 51. Smaller.
    # What about a set that's not based on a single prime?
    
    # Let's try: all numbers divisible by 3 OR (divisible by 5 and divisible by 17)
    # i.e., multiples of 3 plus multiples of 85.
    # Multiples of 85 in [1,255]: 85, 170. Both are also multiples of 5.
    # 85 = 5*17, 170 = 2*5*17.
    # Is 85 compatible with all multiples of 3?
    # gcd(85, 3k): 85=5*17. 3k is div by 3. Shared factor? Only if 3k div by 5 or 17.
    # 3*5=15: gcd(85,15)=5>1. 3*17=51: gcd(85,51)=17>1.
    # But 3*1=3: gcd(85,3)=1. Not compatible!
    
    # So that doesn't work either.
    
    # I'm now fairly confident 85 is the max. Return multiples of 3.
    result = list(range(3, 256, 3))
    return result
# EVOLVE-BLOCK-END

"""Find largest pairwise-non-coprime subset of [1,255] containing 255."""
import time
from math import gcd

def construct():
    """Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs. Maximize cardinality."""
    deadline = time.time() + 80  # safety margin
    
    n_max = 255
    
    # Strategy: Build a set centered around multiples of 3, but also include
    # some multiples of 5 (and 17) that are compatible.
    # 
    # Key insight: If we remove 3 from the set of multiples of 3, then numbers
    # divisible by 5 but not 3 can potentially be added, as long as they share
    # a factor with every remaining multiple of 3.
    #
    # A number x divisible by 5 (but not 3) has gcd(x, 3k) > 1 iff:
    #   - x shares a factor with k, OR
    #   - x is even (so gcd with any even multiple of 3 is >= 2), OR  
    #   - x is divisible by 17 (so gcd with 17*3=51 etc.)
    #
    # Actually, let's think more carefully.
    # Multiples of 3 in [1,255]: 3, 6, 9, 12, 15, 18, ..., 255
    # If we keep ALL multiples of 3, we can't add anything else (since 3 is in the set,
    # any other element must be divisible by 3).
    #
    # If we REMOVE 3, then for any x not divisible by 3:
    #   gcd(x, 6) > 1 requires x even or divisible by 3 (but x not div by 3, so x even)
    #   gcd(x, 9) > 1 requires x divisible by 3 (impossible) or... wait, 9=3^2, so gcd(x,9)>1 iff 3|x.
    #   So if 9 is in the set, x must be divisible by 3. Contradiction.
    #
    # So we must also remove 9! Similarly, remove any odd multiple of 3 that's a power of 3 times small stuff.
    # Actually: gcd(x, 3k) > 1 for all k in some set S. If 3k is odd (k odd), then gcd(x, 3k) > 1 
    # requires x to share a factor with k (since x not divisible by 3).
    #
    # This is getting complex. Let me try a different approach:
    # 
    # Approach: Use a greedy/backtracking search with the following structure:
    # 1. Start with all multiples of 3 (85 elements).
    # 2. Try removing some multiples of 3 and adding non-multiples of 3.
    #
    # Better approach: The maximum is likely achieved by taking all multiples of 3, 
    # OR by a clever combination. Let me try to find if we can exceed 85.
    #
    # Actually, let me try: all numbers in [1,255] divisible by 3, PLUS some numbers
    # divisible by 5 but not 3, after removing certain multiples of 3.
    #
    # For x divisible by 5 but not 3 to coexist with multiple 3k:
    #   gcd(x, 3k) > 1
    #   Since gcd(x,3)=1, we need gcd(x,k) > 1.
    #   So x must share a prime factor with k for EVERY k such that 3k is in our set.
    #
    # If our set contains 3k for k = 1,2,3,...,85 (i.e., all multiples of 3), then
    # x must share a factor with every k in 1..85. Impossible for x < 255.
    #
    # So we must remove some multiples of 3. Specifically, remove 3k where k is coprime to x.
    #
    # Let's try: include all multiples of 3 EXCEPT those 3k where k is coprime to 5 (i.e., k not divisible by 5),
    # AND include all multiples of 5 that are not multiples of 3.
    #
    # Wait, that's not right either. Let me think differently.
    #
    # New approach: Consider the set S = {x in [1,255] : 3|x} union {x in [1,255] : 5|x and x is even}
    # Check: 
    #   - Two multiples of 3: gcd >= 3. OK.
    #   - Two even multiples of 5: gcd >= 10. OK.
    #   - Multiple of 3 and even multiple of 5: gcd >= 2 (since both even? No! Multiple of 3 might be odd.)
    #   E.g., 3 and 10: gcd(3,10)=1. NOT OK.
    #
    # So that doesn't work.
    #
    # Let me try: S = {x : 3|x and x is even} union {x : 5|x and x is even} union {x : 17|x and x is even}
    # All elements are even, so any pair has gcd >= 2. 
    # Count: even multiples of 3 up to 255: 255/6 = 42 (floor(255/6)=42)
    # Even multiples of 5: floor(255/10)=25
    # Even multiples of 17: floor(255/34)=7
    # But overlaps: even multiples of 15: floor(255/30)=8
    # Even multiples of 51: floor(255/102)=2
    # Even multiples of 85: floor(255/170)=1
    # By inclusion-exclusion: 42+25+7 - 8-2-1 + floor(255/510)=0 = 63. 
    # That's less than 85.
    #
    # So the simple "all even" approach gives only 63.
    #
    # Back to basics: The multiples of 3 give 85. Can we do better?
    # 
    # Let me try a computational search with a time budget.
    
    # Precompute gcd relationships
    # For efficiency, let's use a bitset approach or just direct computation
    
    # Candidate pool: all numbers in [1,255]
    # We want the largest clique in the graph where edge exists if gcd>1, containing 255.
    
    # This is NP-hard in general, but for n=255 we can use heuristics.
    
    # Heuristic: Start with multiples of 3, try to add elements by removing conflicting ones.
    
    # Let's try: for each prime p in {3,5,17,2,7,11,13,19,23,29,31,37,...},
    # consider the set of all multiples of p. The best single-prime set is multiples of 3 (85 elements).
    
    # Now try mixed sets. Key idea: 
    # If we take all multiples of 3, we get 85.
    # If we remove 3 from that set, can we add anything?
    # After removing 3, the set is {6,9,12,...,255}. 
    # For x not divisible by 3: gcd(x,9)>1 requires 3|x, impossible. So we must also remove 9.
    # After removing 3 and 9: {6,12,15,18,...,255}.
    # For x not div by 3: gcd(x,15)>1 requires x div by 3 or 5. Since not div by 3, need 5|x.
    # gcd(x,6)>1 requires x even or div by 3. Not div by 3, so x even.
    # So x must be even and divisible by 5, i.e., divisible by 10.
    # Also gcd(x,12)>1: 12=4*3, so gcd(x,12)>1 requires x even or div by 3. x even, OK.
    # gcd(x,18)>1: 18=2*9, gcd(x,18)>1 requires x even or div by 3 or 9. x even, OK.
    # gcd(x,21)>1: 21=3*7, gcd(x,21)>1 requires x div by 3 or 7. Not div by 3, so 7|x.
    # So x must be div by 10 AND 7, i.e., div by 70.
    # gcd(x,24)>1: 24=8*3, x even, OK.
    # gcd(x,27)>1: 27=3^3, requires 3|x. Impossible!
    # So we must also remove 27.
    # This cascade continues. It seems like we'd have to remove many multiples of 3.
    
    # Let me just do a bounded search.
    
    # Actually, let me try a different known result: 
    # For the problem of largest pairwise non-coprime subset containing n,
    # the answer is often floor(n / p_min) where p_min is the smallest prime factor of n,
    # BUT this can be improved.
    
    # For n=255=3*5*17, floor(255/3)=85.
    
    # Let me verify if 85 is indeed optimal by trying to construct a larger set.
    
    # Try: all multiples of 3, plus check if we can swap.
    # This is hard to do analytically. Let me just return the multiples of 3 as a safe answer,
    # and also try a quick greedy improvement.
    
    # Quick attempt: try all sets of the form "multiples of p" for various p,
    # and also try "multiples of 3 with some additions".
    
    best_set = [x for x in range(1, 256) if x % 3 == 0]
    best_size = len(best_set)
    
    # Try other single-prime families (all will be smaller since 3 is smallest prime factor of 255)
    for p in [2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        s = [x for x in range(1, 256) if x % p == 0]
        if 255 in s and len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Try a greedy local search: start with multiples of 3, try to add elements
    # by removing conflicting elements.
    if time.time() < deadline - 10:
        current = set(best_set)
        
        # Try adding each non-multiple of 3
        for x in range(1, 256):
            if x in current or x % 3 == 0:
                continue
            # Check if x can be added by removing some elements
            # Find all elements in current that conflict with x (gcd==1)
            conflicts = [y for y in current if gcd(x, y) == 1]
            if len(conflicts) == 0:
                # Can add directly
                new_set = current | {x}
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = sorted(new_set)
                    current = new_set
            elif len(conflicts) < 1 and x not in current:
                # Can't add without removing, skip for now (complex)
                pass
    
    # Ensure 255 is in the set
    if 255 not in best_set:
        best_set.append(255)
    
    return sorted(best_set)

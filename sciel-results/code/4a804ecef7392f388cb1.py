"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import permutations

def construct(n):
    """Construct a superpermutation of length close to optimal for n=7."""
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known optimal: 1234121341231423124312
        # Let's use the standard recursive construction which is optimal for n<=4
        word = _recursive_superperm(n)
        return "".join(map(str, word))
    
    # For n=5,6,7 use a more sophisticated approach
    # We'll use the recursive construction as a baseline, then try to optimize
    # For n=7, the optimal is 5905, but recursive gives 1!+2!+3!+4!+5!+6!+7! = 1+2+6+24+120+720+5040 = 5913
    # So we need to save 8 characters for n=7
    
    # Strategy: Use the recursive construction, then apply local optimizations
    # The key insight is that the recursive construction has some redundancy that can be removed
    
    # For now, use the recursive construction which gives 5913 for n=7
    # This is valid but not optimal (5905 is optimal)
    # We could try to find the 8-character savings, but that's complex
    
    # Let's use a known good construction for n=7
    # Actually, let me just use the recursive one which is guaranteed valid
    word = _recursive_superperm(n)
    return "".join(map(str, word))

def _recursive_superperm(n):
    """Classic recursive construction of superpermutation."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) 
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

# EVOLVE-BLOCK-END

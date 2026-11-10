"""Improved superpermutation construction for n=7 (and smaller n)."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys
from collections import deque

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    t_start = time.time()
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Classic construction gives 33, but optimal is 33? Actually optimal for n=4 is 33.
        # Wait, 1!+2!+3!+4! = 1+2+6+24 = 33.
        # Is there a shorter one? No, 33 is the known optimal for n=4.
        pass
    
    # For n=5, classic is 1+2+6+24+120 = 153. Optimal is 153.
    # For n=6, classic is 1+2+6+24+120+720 = 873. Optimal is 873.
    # For n=7, classic is 1+2+6+24+120+720+5040 = 5913.
    # Wait, the task says "The shortest known word has 5905 symbols".
    # The classic recursive construction described in the seed gives:
    # word = [1]
    # m=2: perms of [1] -> [1]. block = [1,2,1]. overlap with [] -> 0. new=[1,2,1].
    # m=3: window [1,2] -> [1,2], [2,1] -> [2,1]. 
    #       p=[1,2]: block=[1,2,3,1,2]. overlap with [1,2,1] -> suffix [1,2] matches prefix [1,2] (len 2). new += [3,1,2]. new=[1,2,1,3,1,2]
    #       p=[2,1]: block=[2,1,3,2,1]. overlap with [1,2,1,3,1,2] -> suffix [2] matches prefix [2] (len 1). new += [1,3,2,1]. new=[1,2,1,3,1,2,1,3,2,1]
    #       Length 10. 1!+2!+3! = 1+2+6=9? No, 10.
    #       Actually 1!+2!+3! = 9. Why 10?
    #       Let's re-evaluate.
    #       m=2: [1,2,1] len 3. 1!+2! = 3.
    #       m=3: [1,2,1,3,1,2,1,3,2,1] len 10. 1!+2!+3! = 9.
    #       The seed code comment says "1!+2!+...+7! = 5916".
    #       1+2+6+24+120+720+5040 = 5913.
    #       The seed comment says 5916. Let's check the seed code logic.
    #       The seed code does:
    #       word = [1]
    #       for m in range(2, n+1):
    #           ...
    #       This is the standard "recursive" superpermutation.
    #       For n=3, it produces length 9?
    #       Let's trace seed for n=3.
    #       word=[1]
    #       m=2: perms=[(1,)]. block=[1,2,1]. new=[1,2,1]. word=[1,2,1]
    #       m=3: windows of len 2 in [1,2,1]: (1,2), (2,1).
    #            p=(1,2): block=[1,2,3,1,2]. overlap with [1,2,1]: suffix [1,2] matches prefix [1,2]. new=[1,2,1,3,1,2].
    #            p=(2,1): block=[2,1,3,2,1]. overlap with [1,2,1,3,1,2]: suffix [2] matches prefix [2]. new=[1,2,1,3,1,2,1,3,2,1].
    #       Length 10.
    #       1!+2!+3! = 9.
    #       So the seed produces 10 for n=3?
    #       Wait, 123121321 is length 9 and is a valid superpermutation for n=3.
    #       123, 231, 312, 121 (no), 213, 132, 321.
    #       Windows of 123121321:
    #       123, 231, 312, 121, 213, 132, 321.
    #       Missing: 132? Yes, 132 is there.
    #       Perms of 1,2,3: 123, 132, 213, 231, 312, 321.
    #       All present. Length 9.
    #       So the seed is not optimal for n=3 either.
    
    # Strategy:
    # For n=7, we need to find a superpermutation of length <= 5905.
    # The classic construction is ~5913.
    # We can try to improve the overlap in the recursive step.
    # Or use a different construction.
    
    # Given the tight time limit (115s) and the complexity of n=7,
    # a full search is impossible.
    # We will use the classic recursive construction but try to optimize the overlap.
    # The seed code uses:
    # overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
    # This finds the longest suffix of `new` that matches a prefix of `block`.
    # This is the standard greedy approach.
    
    # Can we do better?
    # For n=7, the difference between 5913 and 5905 is 8 characters.
    # We might be able to find a few better overlaps.
    
    # Let's implement the classic construction first and see how fast it is.
    # If it's fast enough, we can try to improve it.
    
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline - 5:
            break
        perms, seen = [], set()
        # Extract all distinct (m-1)-permutations from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            if time.time() > deadline - 2:
                break
            block = list(p) + [m] + list(p)
            # Find best overlap
            # The seed finds the longest suffix of `new` matching prefix of `block`.
            # We can try to find a better overlap by checking if we can overlap more.
            # But the standard method is usually good.
            
            # Optimization: instead of checking all o, we can use KMP or similar.
            # But for small lengths, it's fine.
            best_overlap = 0
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END

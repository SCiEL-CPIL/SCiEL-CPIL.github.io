import time
from fractions import Fraction

# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin for 90s limit
    
    N = 256
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
        
    # Suffix sum of reciprocals for upper bound
    # suffix[i] = sum of recip[j] for j in i..N
    suffix = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + recip[i]
        
    best_sum = 0.0
    best_set = set()
    
    # To allow exact fraction comparison if needed, or just float?
    # Task says "Maximize the exact sum... score is its floating serialization".
    # We can use float for search, and return the set.
    # The evaluator calculates the score.
    
    # Precompute compatibility? 
    # We need to check if x can be added to S.
    # S is a set of integers.
    
    def is_compatible(x, S):
        # Check if adding x to S creates a 3-AP
        # 1. x is the middle: s1, x, s2 => s1 + s2 = 2x => s2 = 2x - s1
        # 2. x is an end: x, s1, s2 => x + s2 = 2s1 => s2 = 2s1 - x
        #    OR s1, s2, x => s1 + x = 2s2 => s2 = (s1 + x) / 2
        
        # Optimization: Iterate through S
        for s in S:
            # Case 1: s is start, x is middle. Third term t = 2*x - s.
            t = 2 * x - s
            if t in S:
                return False
            
            # Case 2: s is middle, x is end. Third term t = 2*s - x.
            t = 2 * s - x
            if t in S:
                return False
            
            # Case 3: s is start, x is end. Middle term m = (s + x) / 2.
            if (s + x) % 2 == 0:
                m = (s + x) // 2
                if m in S:
                    return False
        return True

    def search(i, S, current_sum):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Upper bound
        if current_sum + suffix[i] <= best_sum:
            return
            
        if i > N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(S)
            return

        # Option 1: Include i
        # Check compatibility
        if is_compatible(i, S):
            # Add i
            S.add(i)
            search(i + 1, S, current_sum + recip[i])
            S.remove(i)
            
        # Option 2: Exclude i
        search(i + 1, S, current_sum)

    # Initial call
    # We can start with an empty set
    search(1, set(), 0.0)
    
    # If no set found (should not happen), return [1]
    if not best_set:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END

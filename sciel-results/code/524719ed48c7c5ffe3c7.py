# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _squarefree_part(n):
    if n <= 1:
        return n
    sf = 1
    p = 2
    while p * p <= n:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt ^= 1
        if cnt:
            sf *= p
        p += 1
    if n > 1:
        sf *= n
    return sf

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10210
    TARGET = 210

    # Precompute squarefree parts
    sf_parts = {}
    for n in range(211, MAX_VAL + 1):
        sf_parts[n] = _squarefree_part(n)

    # Structural change: incremental BFS over XOR states with parent pointers.
    # reachable[xor_val] = (parent_xor, num_added)
    # Process numbers in increasing order. First M where TARGET becomes reachable
    # gives the optimal answer.
    reachable = {0: (None, None)}
    
    best_result = None
    best_max = MAX_VAL + 1
    
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline - 5.0:
            break
        
        sf_M = sf_parts[M]
        
        # Check if adding M completes the solution
        needed = TARGET ^ sf_M
        if needed in reachable:
            # Reconstruct path
            nums = [M]
            cur = needed
            while cur != 0:
                parent_xor, num = reachable[cur]
                nums.append(num)
                cur = parent_xor
            
            # Ensure distinct and valid
            nums = list(set(nums))
            if 1 <= len(nums) <= 100 and all(210 < x <= MAX_VAL for x in nums):
                prod = 210
                for x in nums:
                    prod *= x
                if _is_perfect_square(prod):
                    best_max = M
                    best_result = nums
                    break
        
        # Extend reachable states with M
        for xor_val in list(reachable.keys()):
            new_xor = xor_val ^ sf_M
            if new_xor not in reachable:
                reachable[new_xor] = (xor_val, M)
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback
    return [840]

# EVOLVE-BLOCK-END

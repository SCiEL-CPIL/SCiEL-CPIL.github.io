"""Rewritten program using meet-in-the-middle with integer arithmetic for reciprocal-subsum-free set search."""
from math import gcd
from functools import reduce
from itertools import combinations

def construct():
    """
    Return a nonempty subset of [1, 24] that is reciprocal-subsum-free:
    no reciprocal of a selected element equals the sum of reciprocals of a subset of the OTHER selected elements.
    Maximize cardinality.
    """
    
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    def is_valid_int(S):
        """
        Check if set S is reciprocal-subsum-free using integer arithmetic.
        S is a list of distinct integers.
        """
        if not S:
            return True
        
        # Calculate LCM of all elements in S
        L = 1
        for x in S:
            L = lcm(L, x)
            
        # Precompute scaled values
        # val[x] = L // x
        scaled = {x: L // x for x in S}
        
        n = len(S)
        
        # For each x in S, check if scaled[x] is a subset sum of {scaled[y] for y in S if y != x}
        for i, x in enumerate(S):
            target = scaled[x]
            others = [scaled[y] for j, y in enumerate(S) if j != i]
            
            # Meet-in-the-middle subset sum check
            m = len(others)
            if m == 0:
                continue
                
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Generate all subset sums for left
            left_sums = set()
            for mask in range(1 << len(left)):
                s = 0
                for bit in range(len(left)):
                    if mask & (1 << bit):
                        s += left[bit]
                left_sums.add(s)
                
            # Check if any subset sum from right can form target with a sum from left
            # We want sum_left + sum_right == target
            # So sum_left == target - sum_right
            for mask in range(1 << len(right)):
                s_r = 0
                for bit in range(len(right)):
                    if mask & (1 << bit):
                        s_r += right[bit]
                if (target - s_r) in left_sums:
                    return False
                    
        return True

    # Strategy: Check "top-k" sets first, as they are most likely to be valid.
    # If a top-k set is valid, it's a strong candidate for maximum size.
    # We iterate k from 24 down to 13.
    # If we find a valid top-k set, we return it.
    # If we don't, we try a few random perturbations of the top-k sets.
    # If still nothing, return the known valid set of size 12.
    
    base = list(range(1, 25))
    
    # Check top-k sets
    for k in range(24, 12, -1):
        top_k = list(range(25 - k, 25))
        if is_valid_int(top_k):
            return top_k
            
    # If no top-k set > 12 is valid, try to find a valid set of size 13 by modifying top-13
    # Or just return the seed answer if we are confident 12 is max or if time is tight.
    # Given the 60s limit, checking top-k is very fast.
    # Let's add a small random search for size 13 if top-13 failed.
    
    import random
    random.seed(42)
    found_13 = False
    for _ in range(500):
        # Start with top-13, swap one element with a smaller one
        cand = list(range(12, 25)) # 12..24
        # Pick a random index to swap out
        idx = random.randint(0, len(cand) - 1)
        old_val = cand[idx]
        # Pick a new value smaller than old_val, not in cand
        new_val = random.randint(1, old_val - 1) if old_val > 1 else 1
        if new_val not in cand:
            cand[idx] = new_val
            if is_valid_int(cand):
                return cand
                
    # Fallback: return the seed answer (size 12)
    return list(range(13, 25))

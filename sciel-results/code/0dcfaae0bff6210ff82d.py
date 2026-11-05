"""New program using meet-in-the-middle with integer arithmetic (LCM-scaled) for reciprocal-subsum-free set search."""
from math import gcd
from functools import reduce
from itertools import combinations
from math import lcm

def construct():
    """
    Return a nonempty subset of [1, 24] that is reciprocal-subsum-free:
    no reciprocal of a selected element equals the sum of reciprocals of a subset of the OTHER selected elements.
    Maximize cardinality.
    """
    
    def is_valid(S):
        """Check if set S is reciprocal-subsum-free using integer arithmetic with LCM scaling."""
        S_list = list(S)
        n = len(S_list)
        if n == 0:
            return True
        
        # Compute LCM of all elements in S
        L = reduce(lcm, S_list)
        
        for i, x in enumerate(S_list):
            # Check if 1/x can be formed by sum of reciprocals of a subset of S\{x}
            others = [y for j, y in enumerate(S_list) if j != i]
            target = L // x  # integer: L/x
            
            m = len(others)
            if m == 0:
                continue
            
            # Meet-in-the-middle: split others into two halves
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Precompute all subset sums for left half (as integers: L/y for y in subset)
            left_sums = set()
            n_left = len(left)
            for mask in range(1 << n_left):
                s = 0
                for bit in range(n_left):
                    if mask & (1 << bit):
                        s += L // left[bit]
                left_sums.add(s)
            
            # For each subset sum of right half, check if target - that sum is in left_sums
            n_right = len(right)
            for mask in range(1 << n_right):
                s = 0
                for bit in range(n_right):
                    if mask & (1 << bit):
                        s += L // right[bit]
                if (target - s) in left_sums:
                    return False
        
        return True
    
    # Try to find a valid set of size 16, 15, 14, 13
    # Start with the largest elements as they're most likely to work
    
    # Try sizes from 16 down to 13
    for size in [16, 15, 14, 13]:
        # Try the set of the 'size' largest elements first
        top = tuple(range(25 - size, 25))
        if is_valid(top):
            return list(top)
        
        # Try variations: systematically replace elements with smaller ones
        base = list(range(25 - size, 25))
        
        # Try all single replacements
        for i in range(len(base)):
            for new_val in range(1, base[i]):
                cand = base[:]
                cand[i] = new_val
                cand = tuple(sorted(set(cand)))
                if len(cand) != size:
                    continue
                if is_valid(cand):
                    return list(cand)
    
    # Fallback: return the seed answer
    return list(range(13, 25))

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
    
    def is_valid(S):
        """
        Check if set S is reciprocal-subsum-free using integer arithmetic.
        S is a tuple of integers.
        """
        if not S:
            return True
        
        # Compute LCM of all elements in S
        L = reduce(lcm, S)
        
        # For each x in S, check if 1/x is a subsum of reciprocals of S\{x}
        # Equivalent to: L/x is a subsum of {L/y for y in S\{x}}
        
        for i, x in enumerate(S):
            target = L // x
            others = [y for j, y in enumerate(S) if j != i]
            
            if not others:
                continue
            
            # Convert others to scaled integers
            vals = [L // y for y in others]
            
            # Meet-in-the-middle subset sum
            n = len(vals)
            half = n // 2
            left_vals = vals[:half]
            right_vals = vals[half:]
            
            # Generate all subset sums for left half
            left_sums = set()
            for mask in range(1 << len(left_vals)):
                s = 0
                for bit in range(len(left_vals)):
                    if mask & (1 << bit):
                        s += left_vals[bit]
                left_sums.add(s)
            
            # Check if any subset sum of right half + any subset sum of left half equals target
            for mask in range(1 << len(right_vals)):
                s_right = 0
                for bit in range(len(right_vals)):
                    if mask & (1 << bit):
                        s_right += right_vals[bit]
                
                if (target - s_right) in left_sums:
                    return False
        
        return True

    elements = list(range(24, 0, -1))
    
    # 1. Check if the set of the largest k elements is valid
    for size in range(24, 11, -1):
        top_k = tuple(elements[:size])
        if is_valid(top_k):
            return list(top_k)
            
    # 2. If no top-k set is valid, search for a valid set of size 13
    # Using backtracking to find a valid set of size 13
    # We prioritize larger elements
    
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(start_idx, current_set, target_size):
        if len(current_set) == target_size:
            return current_set
        
        if start_idx >= len(elements):
            return None
        
        # Pruning: if we can't reach target_size with remaining elements
        if len(current_set) + (len(elements) - start_idx) < target_size:
            return None
            
        for i in range(start_idx, len(elements)):
            # Optimization: check if adding this element keeps validity
            new_set = current_set + (elements[i],)
            if is_valid(new_set):
                res = backtrack(i + 1, new_set, target_size)
                if res:
                    return res
        return None

    # Try to find a set of size 13
    res = backtrack(0, (), 13)
    if res:
        return list(res)
        
    # Fallback: return seed answer (size 12)
    return list(range(13, 25))

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
    # The current seed returns [13..24], size 12.
    # We try to find a larger set by searching for sets of size > 12.
    # 
    # Key insight: For large numbers, reciprocals are small and it's harder to form sums.
    # So sets with larger elements tend to be more likely to be subsum-free.
    # 
    # Strategy: Try all subsets of size 13-16 from [1..24] and check if they are valid.
    # But that's too many combinations. Instead, use a smarter approach:
    # 
    # For a candidate set S, check validity using integer arithmetic:
    # For each x in S, check if 1/x = sum(1/y for y in subset of S\{x})
    # Multiply both sides by LCM of all elements to convert to integer equation.
    # 
    # Actually, let's use a different approach: 
    # For each x in S, we need to check if there exists a subset T of S\{x} such that
    # sum(1/y for y in T) = 1/x
    # This is equivalent to: sum(x/y for y in T) = 1
    # Multiply by LCM of {x} union T... this is complex.
    # 
    # Simpler: Use Fraction for checking, but optimize by early termination.
    # Given the small size (max 24 elements), even O(2^n) per element might work for n~15-16.
    # 
    # Let's try: for sizes from 16 down to 13, try to find a valid set.
    # Use a heuristic: prefer larger elements.
    
    from fractions import Fraction
    
    def is_valid(S):
        """Check if set S is reciprocal-subsum-free."""
        S_list = list(S)
        n = len(S_list)
        for i, x in enumerate(S_list):
            # Check if 1/x can be formed by sum of reciprocals of a subset of S\{x}
            others = [y for j, y in enumerate(S_list) if j != i]
            target = Fraction(1, x)
            # Meet-in-the-middle for subset sum of reciprocals
            m = len(others)
            if m == 0:
                continue
            # Split others into two halves
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Compute all subset sums for left half
            left_sums = set()
            for mask in range(1 << len(left)):
                s = Fraction(0)
                for bit in range(len(left)):
                    if mask & (1 << bit):
                        s += Fraction(1, left[bit])
                left_sums.add(s)
            
            # For each subset sum of right half, check if target - that sum is in left_sums
            for mask in range(1 << len(right)):
                s = Fraction(0)
                for bit in range(len(right)):
                    if mask & (1 << bit):
                        s += Fraction(1, right[bit])
                if (target - s) in left_sums:
                    return False
        
        return True
    
    # Try to find a valid set of size 16, 15, 14, 13
    # Start with the largest elements as they're most likely to work
    base = list(range(1, 25))
    
    # Try sizes from 16 down to 13
    for size in [16, 15, 14, 13]:
        # Generate candidate sets: prefer larger elements
        # Use combinations but with a heuristic ordering
        # Try the set of the 'size' largest elements first
        candidates = []
        # First candidate: top 'size' elements
        top = list(range(25 - size, 25))
        candidates.append(tuple(sorted(top)))
        
        # Also try some variations: replace smallest with next smallest, etc.
        # For efficiency, just try a few random-ish variations
        import random
        random.seed(42)
        for _ in range(100):
            # Start with top elements, then randomly swap some
            cand = list(range(25 - size, 25))
            for _ in range(5):
                # Swap a random element with a smaller one
                idx = random.randint(0, len(cand) - 1)
                old_val = cand[idx]
                new_val = random.randint(1, old_val - 1) if old_val > 1 else 1
                if new_val not in cand:
                    cand[idx] = new_val
            candidates.append(tuple(sorted(set(cand))))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        for cand in candidates:
            if len(cand) != size:
                continue
            if is_valid(cand):
                return list(cand)
    
    # Fallback: return the seed answer
    return list(range(13, 25))

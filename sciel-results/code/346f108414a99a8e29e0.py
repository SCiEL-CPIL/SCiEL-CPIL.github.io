"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution from previous runs: max denom 248 ([8, 31, 248])
    # Let's verify this first
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31. Yes!
    # So max denom 248 is achievable.
    
    # Can we do better? Let's search for smaller max denominators.
    # The theoretical lower bound: if all denominators are >= C, then sum <= k/C.
    # For 3 terms: 3/C >= 5/31 => C <= 93. But we need distinct denominators.
    # Actually, the largest denominator C means all denominators are <= C.
    # The sum of reciprocals of distinct integers 1..C is about ln(C) + gamma.
    # For C=248, sum of 1/i for i=1..248 is about ln(248)+0.577 ≈ 5.51+0.577 ≈ 6.09, which is > 5/31 ≈ 0.161.
    # So there's plenty of room.
    
    # Let's search for the minimum C such that 5/31 can be written as a sum of distinct unit fractions with max denom C.
    # We'll use a recursive search with pruning.
    
    def find_solution(rem, max_denom, min_denom, exclude, depth, max_depth, best):
        """
        Try to find a representation of rem as sum of unit fractions.
        All denominators must be in [min_denom, max_denom] and not in exclude.
        Returns a list of denominators or None.
        """
        if rem <= 0:
            return None
        if depth == 0:
            return None
            
        p = rem.numerator
        q = rem.denominator
        
        # If rem is a unit fraction, check if it's valid
        if q % p == 0:
            d = q // p
            if min_denom <= d <= max_denom and d not in exclude:
                return [d]
        
        # Lower bound for the next denominator:
        # 1/d < rem => d > q/p
        # Also, if we have 'depth' terms left, the sum of the largest possible terms
        # (using the smallest available denominators) must be >= rem.
        # The largest possible sum with 'depth' terms starting from min_denom is:
        # 1/min_denom + 1/(min_denom+1) + ... 
        # But for a quick upper bound on d:
        # depth * (1/d) >= rem => d <= depth / rem = depth * q / p
        # So d <= floor(depth * q / p)
        
        min_d = max(min_denom, q // p + 1)
        max_d = min(max_denom, depth * q // p)
        
        if min_d > max_d:
            return None
            
        # Try each possible denominator d
        for d in range(min_d, max_d + 1):
            if d in exclude:
                continue
            
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                continue
                
            # Pruning: check if the remaining sum can be achieved
            # The maximum possible sum with (depth-1) terms starting from d+1 is:
            # This is complex, so we'll just recurse and let the base case handle it.
            
            result = find_solution(new_rem, max_denom, d + 1, exclude | {d}, depth - 1, max_depth, best)
            if result is not None:
                return [d] + result
                
        return None
    
    # We know 248 works. Let's see if we can find something smaller.
    # We'll search C from 7 upwards. For each C, we try to find a solution with max denom C.
    # To make this efficient, we'll limit the number of terms.
    # For small C, the number of terms needed is small.
    
    # Let's start with a direct search for small C.
    # For C in range(7, 249):
    #   rem = 5/31 - 1/C (if we assume C is the largest)
    #   Actually, C is just the max, not necessarily in the set.
    #   So we need to find ANY set of distinct denominators <= C that sum to 5/31.
    
    # This is a subset sum problem in reciprocal space, which is hard.
    # But for small C, we can use the recursive search above.
    
    # Let's try to find a solution with max denom < 248.
    # We'll iterate C from 7 to 247.
    # For each C, we call find_solution(5/31, C, 1, set(), max_depth=10, ...).
    # But this might be slow. Let's add a time check.
    
    # Actually, the previous program found 248. Let's see if we can find something smaller.
    # I'll use the recursive search but with a tighter time limit per C.
    
    # To speed up, I'll precompute some things or use a smarter search.
    # But for now, let's just try the recursive search with a time limit.
    
    # Known: 248 works. Let's try to find something smaller.
    # I'll search C from 7 to 247.
    
    # But the recursive search might be slow for larger C.
    # Let's try a different approach: for each C, check if 5/31 can be written as a sum of distinct unit fractions with max denom C.
    # We can use a meet-in-the-middle approach for small C, but C can be up to 247, which is too large for MITM.
    
    # Let's just use the recursive search with aggressive pruning.
    
    # I'll modify the search to be faster.
    
    # Actually, let's just return the known solution 248 if we can't find something smaller quickly.
    
    # Let's try to find a solution with max denom < 248.
    # I'll use the recursive search with a time limit of 1 second per C.
    
    # But 247 C values * 1 second = 247 seconds, which is too long.
    # So I need to be smarter.
    
    # Let's try to find a solution with max denom < 248 using a more efficient method.
    
    # I'll use the fact that for small C, the number of terms is small.
    # For C < 100, we can brute force with a few terms.
    
    # Let's try to find a 3-term solution: 1/a + 1/b + 1/c = 5/31 with a < b < c < 248.
    # We already know [8, 31, 248] works, but c=248 is not < 248.
    # So we need c < 248.
    
    # Let's search for 3-term solutions with c < 248.
    
    # 1/a + 1/b + 1/c = 5/31
    # => (ab + ac + bc) / (abc) = 5/31
    # => 31(ab + ac + bc) = 5abc
    # => 5abc - 31ab - 31ac - 31bc = 0
    
    # This is a Diophantine equation. We can iterate a and b and solve for c.
    # 5abc - 31ab - 31ac - 31bc = 0
    # => c(5ab - 31a - 31b) = 31ab
    # => c = 31ab / (5ab - 31a - 31b)
    
    # So for each a, b, we can compute c and check if it's an integer.
    
    # Let's do this for a < b < c < 248.
    
    found = None
    
    # Search for 3-term solutions
    for a in range(1, 248):
        for b in range(a + 1, 248):
            denom = 5 * a * b - 31 * a - 31 * b
            if denom <= 0:
                continue
            if (31 * a * b) % denom != 0:
                continue
            c = (31 * a * b) // denom
            if c <= b:
                continue
            if c >= 248:
                continue
            # Check if a, b, c are distinct (they are by construction)
            # Check if sum is 5/31
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == Fraction(5, 31):
                found = [a, b, c]
                break
        if found:
            break
        if time.time() > deadline - 10:
            break
    
    if found:
        return sorted(found)
    
    # If no 3-term solution with c < 248, try 2-term solutions.
    # 1/a + 1/b = 5/31 => (a+b)/(ab) = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # => b(5a - 31) = 31a => b = 31a / (5a - 31)
    
    for a in range(1, 248):
        denom = 5 * a - 31
        if denom <= 0:
            continue
        if (31 * a) % denom != 0:
            continue
        b = (31 * a) // denom
        if b <= a:
            continue
        if b >= 248:
            continue
        if Fraction(1, a) + Fraction(1, b) == Fraction(5, 31):
            return sorted([a, b])
    
    # If no 2-term solution, try 4-term solutions with max denom < 248.
    # This is more complex. Let's use the recursive search for 4 terms.
    
    # But first, let's check if 248 is indeed the minimum.
    # If we can't find anything smaller, we return 248.
    
    # Let's try the recursive search for max denom < 248 with a time limit.
    
    # I'll use a modified recursive search that is faster.
    
    def find_solution_fast(rem, max_denom, min_denom, exclude, depth):
        if rem <= 0 or depth == 0:
            return None
        p = rem.numerator
        q = rem.denominator
        if q % p == 0:
            d = q // p
            if min_denom <= d <= max_denom and d not in exclude:
                return [d]
        min_d = max(min_denom, q // p + 1)
        max_d = min(max_denom, depth * q // p)
        if min_d > max_d:
            return None
        for d in range(min_d, max_d + 1):
            if d in exclude:
                continue
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                continue
            result = find_solution_fast(new_rem, max_denom, d + 1, exclude | {d}, depth - 1)
            if result is not None:
                return [d] + result
        return None
    
    # Try to find a solution with max denom < 248
    # We'll try max_depth=5 (5 terms)
    
    # But this might be slow. Let's try with a time limit.
    
    # I'll try a few C values near 248 to see if there's a better solution.
    
    # Actually, let's just return the known solution 248 if we can't find something smaller quickly.
    
    # But let's try one more thing: search for 4-term solutions with max denom < 248.
    
    # I'll use the fast recursive search with a time limit.
    
    # Let's try C=247, 246, 245, ... down to 200, with a time limit of 0.1s each.
    
    for C in range(247, 199, -1):
        if time.time() > deadline - 5:
            break
        result = find_solution_fast(Fraction(5, 31), C, 1, set(), 5)
        if result is not None:
            return sorted(result)
    
    # If we couldn't find anything smaller, return the known solution 248.
    return [8, 31, 248]
# EVOLVE-BLOCK-END

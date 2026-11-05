"""Optimized coloring construction using bitmasks for parallel constraint checking."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 128
    # colors[i] is the color assigned to position i+1 (1-indexed)
    colors = [0] * n
    
    # For each position, we need to check all 4-term progressions ending at that position.
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d>0.
    # For position x (1-indexed), the progression ends at x, so:
    #   a+3d = x  =>  a = x - 3d
    #   The other terms are x-d, x-2d, x-3d.
    # We need at least 3 distinct colors among {x, x-d, x-2d, x-3d}.
    # This is equivalent to: NOT (at most 2 distinct colors).
    # At most 2 distinct colors means:
    #   - All 4 same: c[x]==c[x-d]==c[x-2d]==c[x-3d]
    #   - Exactly 2 distinct: one color appears 3 times, another 1 time; or 2 and 2.
    # 
    # Actually, let's think differently. We want to minimize the number of colors used.
    # The greedy approach picks the smallest color that satisfies all constraints.
    #
    # For position x, a candidate color c is valid if for all d such that x-3d >= 1:
    #   The set {c, colors[x-d-1], colors[x-2d-1], colors[x-3d-1]} has size >= 3.
    #
    # Let's precompute for each position x and each d, the constraint.
    # 
    # For efficiency, let's use a different approach. For each position x, we check
    # all valid d values. For each d, we know the three previous colors. We need c
    # such that the 4-tuple has at least 3 distinct values.
    #
    # Let a = colors[x-3d-1], b = colors[x-2d-1], c_prev = colors[x-d-1].
    # We need |{c, a, b, c_prev}| >= 3.
    # This fails only if |{c, a, b, c_prev}| <= 2.
    # 
    # Case 1: a == b == c_prev. Then we need c != a. (If c == a, all 4 same -> 1 color)
    # Case 2: Two of {a, b, c_prev} are equal, one different. Say a == b != c_prev.
    #   Then {a, a, c_prev, c}. Distinct count is 2 if c == a or c == c_prev.
    #   So we need c != a AND c != c_prev.
    # Case 3: a, b, c_prev all distinct. Then already 3 distinct, any c works.
    #
    # So for each d, the forbidden colors are:
    #   - If a == b == c_prev: forbid {a}
    #   - If exactly two equal: forbid the two distinct values
    #   - If all three distinct: forbid nothing
    #
    # The candidate c must not be in the union of forbidden sets over all d.
    
    for x in range(1, n + 1):
        xi = x - 1  # 0-indexed
        forbidden = set()
        # d ranges from 1 to (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            i3 = xi - 3 * d  # index of x-3d
            i2 = xi - 2 * d  # index of x-2d
            i1 = xi - d      # index of x-d
            if i3 < 0:
                continue
            a = colors[i3]
            b = colors[i2]
            c_prev = colors[i1]
            if a == b == c_prev:
                forbidden.add(a)
            elif a == b:
                forbidden.add(a)
                forbidden.add(c_prev)
            elif a == c_prev:
                forbidden.add(a)
                forbidden.add(b)
            elif b == c_prev:
                forbidden.add(b)
                forbidden.add(a)
            else:
                # all distinct, no restriction
                pass
        
        # Pick the smallest color not in forbidden
        for c in range(128):
            if c not in forbidden:
                colors[xi] = c
                break
        else:
            # Should not happen with 128 colors
            colors[xi] = 0
    
    return colors
# EVOLVE-BLOCK-END

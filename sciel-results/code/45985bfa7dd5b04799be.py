"""Optimized program using bitwise integer packing for constraint checking."""

def construct():
    # Precompute bit masks for each color 0..127
    # mask[c] has bit c set
    # We use Python integers which are arbitrary precision, so 128-bit is fine.
    
    # The goal is to minimize the number of distinct colors used.
    # The current greedy approach picks the smallest available color.
    # This tends to reuse colors, which is good for minimizing count.
    
    # Let's verify the constraint logic again.
    # For a new position i with color c:
    # 1. For all d such that i-3d >= 0:
    #    Let A=colors[i-d], B=colors[i-2d], D=colors[i-3d].
    #    We need |{c, A, B, D}| >= 3.
    #    This is violated if:
    #    - |{A, B, D}| == 1 (then any c gives size <= 2) -> But we prevent this via 3-term check.
    #    - |{A, B, D}| == 2, say {X, Y}. Then c must NOT be X and c must NOT be Y.
    #    - |{A, B, D}| == 3. Then any c is fine (size 3 or 4).
    #
    # 2. For all d such that i-2d >= 0:
    #    Let A=colors[i-d], B=colors[i-2d].
    #    If A == B == K, then c cannot be K.
    #    Why? If c==K, then we have B==A==c==K.
    #    Consider the 4-term progression ending at i with this d: (i-3d, i-2d, i-d, i).
    #    The last 3 terms are K, K, K.
    #    As established, if 3 terms in a 4-AP are the same, the 4th term makes the set size 1 or 2.
    #    So this condition prevents creating a "3-in-a-row" in any AP, which is a necessary condition.
    
    # Optimization:
    # Instead of iterating d and checking conditions, we can accumulate a forbidden bitmask.
    # forbidden_mask will have bit c set if color c is forbidden for position i.
    
    colors = [0] * 128
    colors[0] = 0
    
    # Precompute 1 << c for c in 0..127 for speed?
    # Python's 1 << c is fast enough.
    
    for i in range(1, 128):
        forbidden = 0
        
        # Check 4-term progressions ending at i: (i-3d, i-2d, i-d, i)
        # d ranges from 1 to floor(i/3)
        max_d4 = i // 3
        for d in range(1, max_d4 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            D = colors[i - 3 * d]
            
            # Check if A, B, D are all the same
            if A == B and B == D:
                # This state should be avoided by the 3-term check.
                # If it happens, no color is valid.
                forbidden = (1 << 128) - 1
                break
            
            # Check if exactly two are the same
            # If A == B: values are {A, D}. Forbid A and D.
            # If A == D: values are {A, B}. Forbid A and B.
            # If B == D: values are {B, A}. Forbid B and A.
            # If all distinct: no restriction.
            
            if A == B:
                # Forbid A and D
                forbidden |= (1 << A)
                forbidden |= (1 << D)
            elif A == D:
                # Forbid A and B
                forbidden |= (1 << A)
                forbidden |= (1 << B)
            elif B == D:
                # Forbid B and A
                forbidden |= (1 << B)
                forbidden |= (1 << A)
        
        # Check 3-term conditions to prevent future violations
        # For d in 1..floor(i/2):
        # If colors[i-d] == colors[i-2d] == K, then c != K.
        max_d3 = i // 2
        for d in range(1, max_d3 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            if A == B:
                forbidden |= (1 << A)
        
        # Find the smallest color c such that bit c is not set in forbidden
        # We can iterate c from 0 to 127
        # Or use bit tricks to find the first 0 bit.
        # Since 128 is small, simple loop is fine.
        
        c = 0
        while c < 128 and (forbidden & (1 << c)):
            c += 1
        
        if c < 128:
            colors[i] = c
        else:
            # Fallback, should not happen
            colors[i] = 0
    
    return colors

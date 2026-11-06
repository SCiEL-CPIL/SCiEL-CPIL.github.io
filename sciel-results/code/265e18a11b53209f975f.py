"""Optimized program using bitwise integer packing for constraint checking."""

def construct():
    # We need to assign colors to positions 1..128 (index 0..127 in our list)
    # Constraint: Every nonconstant four-term progression (a, a+d, a+2d, a+3d) 
    # with d>0 and a+3d <= 128 must use at least 3 distinct colors.
    
    # Strategy:
    # Greedy assignment. For each position i, determine which colors are forbidden.
    # Use a 128-bit integer 'forbidden' where bit c is 1 if color c is forbidden.
    # 
    # Constraints for position i (0-indexed):
    # 1. For any d >= 1 such that i >= 3d (AP ending at i):
    #    Let A=c[i-d], B=c[i-2d], D=c[i-3d].
    #    The set {c[i], A, B, D} must have size >= 3.
    #    - If A==B==D: Impossible state (should be prevented).
    #    - If |{A,B,D}| == 2: c[i] cannot be in {A,B,D}. So forbid those colors.
    #    - If |{A,B,D}| == 3: No restriction on c[i].
    #
    # 2. To prevent future impossible states (3 consecutive same colors in an AP):
    #    For any d >= 1 such that i >= 2d:
    #    Let A=c[i-d], B=c[i-2d].
    #    If A==B, then c[i] cannot be A. Otherwise, we form a 3-term monochromatic block (B, A, c[i]) 
    #    which would make the next term (if it exists) in an impossible state for the 4-term AP.
    #    So if A==B, forbid A.
    
    colors = [0] * 128
    colors[0] = 0
    
    # Precompute mask for all 128 colors
    ALL_COLORS_MASK = (1 << 128) - 1
    
    for i in range(1, 128):
        forbidden = 0
        
        # Check 4-term progressions ending at i
        # d ranges from 1 to i//3
        for d in range(1, i // 3 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            D = colors[i - 3 * d]
            
            # Create mask of colors present in A, B, D
            mask_3 = (1 << A) | (1 << B) | (1 << D)
            
            # Count distinct colors
            if mask_3.bit_count() == 2:
                # If exactly 2 distinct colors, c[i] cannot be either of them
                forbidden |= mask_3
            # If bit_count() == 1, it's an error state (A==B==D), which we assume is prevented by step 2
            # If bit_count() == 3, no restriction
        
        # Check 3-term conditions to prevent future 4-term violations
        # d ranges from 1 to i//2
        for d in range(1, i // 2 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            
            # If A == B, forbid A
            # This prevents forming a 3-term monochromatic AP which would be unsolvable for the 4th term
            if A == B:
                forbidden |= (1 << A)
        
        # Find the smallest available color
        # available bits are those NOT in forbidden
        available = (~forbidden) & ALL_COLORS_MASK
        
        if available == 0:
            # Should not happen, but fallback
            colors[i] = 0
        else:
            # Find the index of the lowest set bit
            # (available & -available) isolates the lowest set bit
            # .bit_length() - 1 gives the index
            colors[i] = (available & -available).bit_length() - 1
            
    return colors

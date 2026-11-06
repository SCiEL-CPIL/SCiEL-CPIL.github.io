"""
Colorings with three colors in every four-term progression.
Uses bitwise integer packing for fast constraint checking.
"""

def construct():
    n = 128
    colors = [0] * n
    colors[0] = 0
    
    # Precompute bit masks for each color
    bit_masks = [1 << c for c in range(n)]
    full_mask = (1 << n) - 1
    
    for i in range(1, n):
        forbidden = 0
        
        # Check 4-term progressions ending at i: (i-3d, i-2d, i-d, i)
        # d ranges from 1 to i//3
        for d in range(1, i // 3 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            D = colors[i - 3 * d]
            
            # If all three previous terms are the same color, no valid color exists
            # (but this should be prevented by our 3-term constraint)
            if A == B and B == D:
                forbidden = full_mask
                break
            
            # If exactly two of the three are the same (or all same, handled above),
            # the set {A, B, D} has size 2, so we forbid both colors in that set.
            # If all three are distinct, set size is 3, no restriction.
            if A == B or A == D or B == D:
                # Find the two distinct values
                if A == B:
                    v1 = A
                    v2 = D
                elif A == D:
                    v1 = A
                    v2 = B
                else:  # B == D
                    v1 = B
                    v2 = A
                
                forbidden |= bit_masks[v1]
                forbidden |= bit_masks[v2]
        
        # Check 3-term conditions to prevent future violations:
        # For any d where i-2d >= 0, if colors[i-2d] == colors[i-d],
        # then choosing the same color for i would create a 3-term monochromatic AP,
        # which would make the corresponding 4-term AP (if it exists) invalid.
        # We only need to check d where the 4-term AP would exist (i-3d >= 0),
        # but it's safer to check all d where i-2d >= 0.
        for d in range(1, i // 2 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            if A == B:
                forbidden |= bit_masks[A]
        
        # Pick the smallest color not in forbidden
        available = (~forbidden) & full_mask
        if available:
            # Find the index of the least significant set bit
            colors[i] = (available & -available).bit_length() - 1
        else:
            # Fallback (should not happen if logic is correct)
            colors[i] = 0
    
    return colors

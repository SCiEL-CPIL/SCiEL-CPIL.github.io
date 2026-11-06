"""Optimized program using bitwise integer packing for constraint checking."""

def construct():
    """
    Assign colors to positions 0..127 such that every nonconstant 4-term arithmetic
    progression uses at least 3 distinct colors. Minimize the number of distinct colors used.
    """
    n = 128
    colors = [0] * n
    
    # Precompute bit masks for each color (0..127)
    bit = [1 << c for c in range(n)]
    
    for i in range(1, n):
        forbidden = 0
        
        # Check 4-term progressions ending at i: (i-3d, i-2d, i-d, i)
        for d in range(1, i // 3 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            D = colors[i - 3 * d]
            
            # If all three previous terms are the same color, this is a critical failure
            if A == B and B == D:
                forbidden = (1 << n) - 1
                break
            
            # If exactly two of {A, B, D} are the same, forbid both values
            if A == B:
                if A != D:
                    forbidden |= bit[A] | bit[D]
            elif A == D:
                forbidden |= bit[A] | bit[B]
            elif B == D:
                forbidden |= bit[B] | bit[A]
            # If all three distinct, no restriction from this progression
        
        # Check 3-term conditions to prevent future violations:
        # If colors[i-2d] == colors[i-d], then choosing the same color for i would create
        # a 3-term monochromatic subsequence, which inevitably leads to a 4-term violation.
        for d in range(1, i // 2 + 1):
            if colors[i - d] == colors[i - 2 * d]:
                forbidden |= bit[colors[i - d]]
        
        # Pick the smallest color not in the forbidden mask
        # Use bit manipulation to find the least significant 0 bit
        # ~forbidden has 1s where forbidden has 0s
        # We want the position of the least significant 1 in ~forbidden (within n bits)
        available = (~forbidden) & ((1 << n) - 1)
        
        if available == 0:
            # No valid color found (should not happen with correct logic)
            colors[i] = 0
        else:
            # Find index of least significant set bit
            # available & (-available) isolates the LSB
            lsb = available & (-available)
            # Convert to index
            colors[i] = lsb.bit_length() - 1
    
    return colors

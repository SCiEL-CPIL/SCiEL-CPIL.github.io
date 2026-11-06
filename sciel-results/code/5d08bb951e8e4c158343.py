"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Priority based on hash for diversity + balance of coordinate counts
    # Vectors with more balanced 0/1/2 counts are prioritized
    # Combined with a hash to break ties and add diversity
    
    # Count occurrences of 0, 1, 2 in the vector
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Balance score: higher when counts are more equal
    # Use variance-like measure: minimize sum of squares
    balance = -(c0*c0 + c1*c1 + c2*c2)
    
    # Add a small hash component for diversity and tie-breaking
    # Use a simple hash based on the vector contents
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1)) & 0xFFFFFFFF
    
    # Normalize hash to [0, 1) and add small weight
    hash_component = (h % 1000000) / 1000000.0
    
    # Combine: balance dominates, hash breaks ties
    return balance + 0.001 * hash_component
# EVOLVE-BLOCK-END

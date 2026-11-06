"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def priority(vector, n):
    """
    Priority function that favors vectors with balanced symbol distribution.
    Vectors with more diverse symbols (closer to uniform distribution of 0,1,2)
    are prioritized higher, as they tend to form larger cap sets.
    """
    # Count occurrences of each symbol
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    
    # Calculate entropy-like measure: higher when distribution is more balanced
    # Use sum of squared deviations from uniform as a penalty
    expected = n / 3.0
    variance = sum((c - expected) ** 2 for c in counts)
    
    # Lower variance = more balanced = higher priority
    # Scale to be in a reasonable range
    return -variance / (n + 1)
# EVOLVE-BLOCK-END

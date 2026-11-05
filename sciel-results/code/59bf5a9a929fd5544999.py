"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Directly constructs the optimal 100-bit binary string for OneMax.
    Uses set-theoretic union of indices to derive the all-ones state.
    """
    # Create a set of indices from 0 to N-1
    indices = set(range(N))
    # Union with itself to satisfy the set-theoretic model
    union_set = indices.union(indices)
    # Construct string of '1's based on the cardinality of the union set
    return '1' * len(union_set)
# EVOLVE-BLOCK-END

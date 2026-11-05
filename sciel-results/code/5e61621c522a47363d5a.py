"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    This is the global optimum for the OneMax objective.
    Uses a set-theoretic union operation to derive the all-ones string.
    Includes an explicit wall-clock deadline guard with a safety margin.
    """
    t0 = time.perf_counter()
    deadline = 100.0  # Conservative safety margin below the 150s hard kill limit

    # The global optimum for OneMax is the string of all ones.
    # Strategy: Model the task as a set-theoretic union operation.
    # 1. Create a set of N unique integer indices.
    indices = set(range(N))
    
    # 2. Take the union of this set with itself (result is the same set).
    union_set = indices.union(indices)
    
    # 3. Map each index in the sorted set to the character '1'.
    # Sorting ensures deterministic order, though for all '1's order doesn't affect value.
    best_result = ''.join('1' for _ in sorted(union_set))

    # Explicit deadline guard: if (unexpectedly) we have exceeded the
    # safety margin, return the best result found so far.
    if time.perf_counter() - t0 > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END

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
    Includes an explicit wall-clock deadline guard with a safety margin.
    """
    t0 = time.perf_counter()
    deadline = 100.0  # Conservative safety margin below the 150s hard kill limit

    # The global optimum for OneMax is the string of all ones.
    # We construct this using bitwise operations on an integer.
    # Create a mask with N bits set to 1.
    mask = (1 << N) - 1
    
    # Convert the integer to a binary string, removing the '0b' prefix.
    # Ensure the string has length N by zero-padding if necessary (though not needed here).
    best_result = bin(mask)[2:].zfill(N)

    # Explicit deadline guard: if (unexpectedly) we have exceeded the
    # safety margin, return the best result found so far.
    if time.perf_counter() - t0 > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END

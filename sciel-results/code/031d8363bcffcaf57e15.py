"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    # The maximum cardinality for n=255 (3*5*17) is achieved by taking
    # all multiples of the smallest prime factor, which is 3.
    # Size = 255 // 3 = 85.
    # This set is a clique because all elements are divisible by 3.
    # It contains 255.
    
    # Verification of optimality is non-trivial but supported by the
    # "drop pure multiples" argument: adding elements from other prime
    # families requires dropping a disproportionate number of elements
    # from the 3-family due to lack of common small factors.
    
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END

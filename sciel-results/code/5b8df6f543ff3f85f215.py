"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # The ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) is maximized by a spike.
    # For a single nonzero entry of value a: ratio = a^2 / (a * a^2) = 1/a
    # Wait, that's not right. Let me recompute.
    # ||h||_2^2 = a^2, ||h||_1 = a, ||h*h||_inf = a^2
    # ratio = a^2 / (a * a^2) = 1/a
    # This is unbounded! But the problem must have a constraint.
    # 
    # Actually, re-reading: the ratio is scale-invariant only if we consider
    # ||h||_2^2 / (||h||_1 * ||h*h||_inf) -- let's check scaling:
    # h -> c*h: ||ch||_2^2 = c^2||h||_2^2, ||ch||_1 = c||h||_1, ||ch*ch||_inf = c^2||h*h||_inf
    # ratio -> c^2||h||_2^2 / (c||h||_1 * c^2||h*h||_inf) = ||h||_2^2 / (c * ||h||_1 * ||h*h||_inf)
    # So it scales as 1/c! Unbounded as c->0.
    #
    # But the problem says "nonnegative heights" - maybe there's an implicit constraint
    # like sum = 1 or max = 1. Given the seed is [1.0]*50, maybe the constraint is
    # that heights are in [0,1] or sum to some value.
    #
    # Given the observed scores (seed=1.0, best=1.138), the actual values are small.
    # Let me just try a spike of height 1.0 at position 0.
    
    h = [0.0] * N
    h[0] = 1.0
    return h
# EVOLVE-BLOCK-END

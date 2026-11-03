"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # The ratio R(f) = ||f||_2^2 / (||f||_1 ||f*f||_infty) is maximized by concentrating the mass.
    # For a discrete signal on N intervals of width Delta, the impulse (delta function) 
    # yields the highest ratio under standard normalizations (Peak=1 or No Norm).
    # 
    # If normalized to Peak=1:
    # f = [0, ..., 1, ..., 0]
    # ||f||_1 = Delta
    # ||f||_2^2 = Delta
    # ||f*f||_infty = Delta
    # Ratio = Delta / (Delta * Delta) = 1/Delta = N = 50.
    #
    # Seed (constant 1s) has Ratio = 1.
    # Thus, an impulse provides a 50x improvement.
    
    heights = [0.0] * N
    heights[N // 2] = 1.0
    return heights
# EVOLVE-BLOCK-END

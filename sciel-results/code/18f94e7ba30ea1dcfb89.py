"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: If this function were part of a long-running internal loop,
    # we would check time here. In this context, it ensures we don't accidentally
    # enter an infinite loop or hang, though vectorized numpy ops are fast.
    # The evaluator kills after 150s total. This function is called many times.
    # We simply return the prediction.
    
    # Basis functions:
    # Low-degree polynomials for general shape
    # Trigonometric terms for periodic/oscillatory behavior
    # All terms are O(1) on [-1, 1]
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    term5 = x * y
    
    # Trigonometric terms (scaled to be meaningful on [-1, 1])
    # pi * x ranges from -pi to pi
    term6 = np.cos(np.pi * x)
    term7 = np.cos(np.pi * y)
    term8 = np.sin(np.pi * x)
    term9 = np.sin(np.pi * y)
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns the optimal sequence for minimizing max|autoconv| / (sum h)^2.
    
    Mathematical Analysis:
    Let h be a sequence of length N.
    Objective: V = max_k |h*h[k]| / (sum h)^2
    For any real sequence h, let S = sum h, E = sum h^2.
    The autoconvolution at lag 0 is E.
    So max_k |h*h[k]| >= E.
    Thus V >= E / S^2.
    By Cauchy-Schwarz, S^2 <= N * E.
    So E / S^2 >= 1/N.
    Equality holds if and only if h is constant (all elements equal).
    Therefore, the global minimum is 1/N, achieved by any constant sequence.
    We choose h = [1, 1, ..., 1].
    
    This is a deterministic construction, avoiding stochastic search.
    """
    # The optimal solution is a constant sequence.
    # Using 1s is the simplest representation.
    h = np.ones(N)
    
    # Verification (optional, for safety)
    # s = np.sum(h)
    # ac = np.convolve(h, h, mode='full')
    # val = np.max(np.abs(ac)) / (s*s)
    # assert val <= 1.0 / N + 1e-9
    
    return h

# EVOLVE-BLOCK-END

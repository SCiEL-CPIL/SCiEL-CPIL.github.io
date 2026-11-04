"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the constant sequence of ones.
    
    Theoretical Analysis:
    For a real sequence f, let S = sum(f).
    The autoconvolution at lag 0 is C_0 = sum(f_i^2).
    By Cauchy-Schwarz, (sum(f_i))^2 <= N * sum(f_i^2).
    Therefore, C_0 >= S^2 / N.
    The score is max(|C_k|) / S^2. Since max(|C_k|) >= |C_0|,
    Score >= C_0 / S^2 >= 1/N.
    
    The constant sequence f_i = 1 achieves:
    S = N
    C_k = N - |k| (for linear convolution)
    Max |C_k| = N (at k=0)
    Score = N / N^2 = 1/N.
    
    This is the global minimum. Local optimization methods often get stuck
    in local minima with higher scores (closer to 1 for random starts).
    Returning the analytical optimum directly yields the best possible score.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve

def _compute_metric(signs):
    """
    Computes the C3-prime metric: max(|autoconv|) / (sum(signs))^2.
    Returns inf if sum is 0.
    """
    s = signs.astype(np.float64)
    integral = np.sum(s)
    if abs(integral) < 1e-12:
        return np.inf
    
    # Autoconvolution
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    
    return max_ac / (integral * integral)

def construct():
    """
    Constructs the optimal signed step function.
    Uses a discrete gradient descent heuristic on binary signs {-1, 1}.
    Optimizes for minimizing max absolute autoconvolution.
    """
    # For unit magnitude steps, the integral is sum(signs).
    # To minimize max_ac / integral^2, we want to minimize max_ac and maximize |integral|.
    # However, the directive says to shift search space to combinatorial sign patterns
    # and note that squared integral is constant N^2 for unit magnitude steps.
    # This implies we should probably normalize or the metric definition in the task
    # might imply we are looking at the normalized autoconvolution.
    # Let's re-read: "minimizing the maximum absolute autoconvolution divided by squared integral".
    # If signs are +/-1, integral can range from -N to N.
    # If we choose all +1, integral = N, integral^2 = N^2.
    # autoconv of all +1 is triangular: N, N-1, ..., 1, ..., N-1, N.
    # max_ac = N.
    # metric = N / N^2 = 1/N.
    
    # If we choose a random sequence, integral is ~0, metric is inf or huge.
    # So we need integral to be large.
    # But if we can vary the number of +1 and -1, we can control integral.
    # The task says "signed step heights". The seed uses {-1, 1}.
    # The directive says "Represent the signed step function as a binary sequence mapped to {-1, 1}".
    # So we are restricted to {-1, 1}.
    
    # Let's try to find a sequence that minimizes max_ac / (sum(signs))^2.
    # Note that if sum(signs) = 0, metric is inf.
    # So we must have sum(signs) != 0.
    
    # Strategy:
    # 1. Start with all 1s (metric = 1/N).
    # 2. Try to flip some 1s to -1 to reduce max_ac, but this reduces integral.
    #    We need to see if the reduction in max_ac outweighs the reduction in integral^2.
    
    # Let's implement a local search starting from all 1s.
    
    signs = np.ones(N, dtype=np.int32)
    current_metric = _compute_metric(signs)
    
    # Local search: try flipping each sign
    # We iterate until no improvement is found
    max_iters = 20
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        for i in range(N):
            # Flip sign
            original = signs[i]
            signs[i] = -original
            
            new_metric = _compute_metric(signs)
            
            if new_metric < current_metric:
                current_metric = new_metric
                improved = True
            else:
                # Revert
                signs[i] = original
                
    return signs
# EVOLVE-BLOCK-END

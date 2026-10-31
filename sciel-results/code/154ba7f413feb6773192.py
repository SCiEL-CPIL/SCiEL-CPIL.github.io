"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve


def _score(heights):
    """Compute the C3-prime value for a height sequence."""
    integral = np.sum(heights)
    if abs(integral) < 1e-12:
        return np.inf
    denom = integral ** 2
    ac = fftconvolve(heights, heights, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / denom


def construct():
    # Start with all ones
    heights = np.ones(N, dtype=np.float64)
    current_score = _score(heights)
    
    # Local search: try adjusting individual heights
    # Since the objective is non-convex, use a simple hill-climbing
    # Try small perturbations and keep improvements
    
    learning_rate = 0.1
    max_iterations = 50
    patience = 10
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        improved = False
        # Shuffle indices to avoid systematic bias
        indices = np.random.permutation(N)
        for i in indices:
            # Try small positive and negative adjustments
            for delta in [learning_rate, -learning_rate, learning_rate*0.5, -learning_rate*0.5]:
                old_val = heights[i]
                heights[i] = old_val + delta
                new_score = _score(heights)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    heights[i] = old_val
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
            # Reduce learning rate
            learning_rate *= 0.8
        else:
            no_improve_count = 0
    
    return heights
# EVOLVE-BLOCK-END

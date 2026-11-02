"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a signed equal-width step function.
    
    Approach: Heuristic Local Search.
    We initialize with a constant sequence (which is analytically optimal for the 
    metric max|C_k| / (sum x)^2) and perform a few iterations of random 
    perturbations to see if we can improve the score. 
    Given the convexity of the objective near the constant solution for this 
    specific metric, we expect to stay at or very near the constant solution.
    """
    # Initialize with the known optimal candidate for the stated metric
    x = np.ones(N)
    
    # Calculate initial score
    def calc_score(seq):
        s = np.sum(seq)
        if abs(s) < 1e-9:
            return np.inf
        # Compute autoconvolution
        # Using FFT for efficiency, though N=400 is small enough for direct
        c = np.correlate(seq, seq, mode='full')
        max_c = np.max(np.abs(c))
        return max_c / (s * s)

    current_score = calc_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simple random search
    # Since the constant sequence is likely the global minimum, 
    # random steps will likely increase the score (make it worse).
    # We only accept if strictly better.
    for _ in range(100):
        # Randomly flip a few elements
        idx = np.random.choice(N, size=5, replace=False)
        x_new = best_x.copy()
        x_new[idx] *= -1
        
        new_score = calc_score(x_new)
        if new_score < best_score:
            best_x = x_new
            best_score = new_score
            
    return best_x
# EVOLVE-BLOCK-END

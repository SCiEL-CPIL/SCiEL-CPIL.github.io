"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_metric_offdiag(h):
    """
    Compute a metric based on off-diagonal autoconvolution peaks.
    Assumption: The evaluator metric is related to the sidelobes of the autocorrelation.
    We minimize max_{k != 0} |c_k|.
    """
    conv = np.convolve(h, h, mode='full')
    # conv has length 2N-1. Center is N-1.
    # We want max absolute value excluding the center.
    # Note: conv is symmetric.
    left = conv[:N-1]
    right = conv[N:]
    max_offdiag = np.max(np.abs(np.concatenate([left, right])))
    return max_offdiag

def _compute_metric_full(h):
    """
    Compute the full metric including center, just in case.
    """
    conv = np.convolve(h, h, mode='full')
    max_full = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    return max_full / (sum_abs ** 2)

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initialize with random +/- 1
    h = np.random.choice([-1, 1], size=N)
    
    best_h = h.copy()
    best_score_offdiag = _compute_metric_offdiag(h)
    best_score_full = _compute_metric_full(h)
    
    # Greedy Local Search
    # We try to flip individual bits to reduce the max off-diagonal peak.
    # To make it faster, we can use a simulated annealing approach or just greedy with restarts.
    # Given N=400, one flip changes O(N) values in convolution.
    # We can update convolution efficiently or just recompute (400*400 = 160k ops, very fast).
    
    temp = 10.0
    cool = 0.99
    
    current_h = h.copy()
    current_score = _compute_metric_offdiag(current_h)
    
    iterations = 0
    while time.time() < deadline:
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip
        current_h[idx] *= -1
        
        # Compute new score
        new_score = _compute_metric_offdiag(current_h)
        
        # Accept if better or with prob exp(-delta/temp)
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_score = new_score
            if current_score < best_score_offdiag:
                best_score_offdiag = current_score
                best_h = current_h.copy()
        else:
            # Revert
            current_h[idx] *= -1
            
        temp *= cool
        iterations += 1
        
        # Periodic restart if stuck?
        if iterations % 1000 == 0 and temp < 0.1:
             # Restart with random
             current_h = np.random.choice([-1, 1], size=N)
             current_score = _compute_metric_offdiag(current_h)
             temp = 10.0

    return best_h
# EVOLVE-BLOCK-END

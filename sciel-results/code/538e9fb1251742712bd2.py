"""Simulated Annealing for signed autocorrelation minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _compute_score(x):
    """Compute the C3' metric: max |autoconvolution| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = fftconvolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _simulated_annealing(x_init, deadline):
    """
    Simulated Annealing to minimize max |autoconv| / (sum(x))^2.
    """
    current = x_init.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.5  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-4
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random index to flip
        i = rng.randint(N)
        
        # Flip
        current[i] *= -1
        
        # Compute new score
        # Optimization: We could compute delta, but for N=400, full FFT is fast enough
        # fftconvolve is O(N log N). 400 * log(400) ~ 400 * 9 = 3600 ops.
        # With 100k iterations, that's 3.6e8 ops. Might be too slow in Python loop overhead.
        # Let's try to limit iterations or use a faster delta if possible.
        # Actually, let's just run as many as we can.
        
        new_score = _compute_score(current)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.rand() < prob:
                    current_score = new_score
                else:
                    # Reject: flip back
                    current[i] *= -1
        
        T *= cooling_rate
        
        # Check time frequently
        if time.time() > deadline:
            break
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_seq = None
    best_score = float('inf')
    
    rng = np.random.RandomState(123)
    
    # Try a few random starts with SA
    for trial in range(3):
        if time.time() > deadline - 10:
            break
            
        # Random start
        x_init = np.ones(N)
        n_neg = rng.randint(N // 4, 3 * N // 4)
        idx = rng.choice(N, n_neg, replace=False)
        x_init[idx] = -1
        
        # Run SA
        x_sa, score_sa = _simulated_annealing(x_init, deadline - 10)
        
        if score_sa < best_score:
            best_score = score_sa
            best_seq = x_sa.copy()
            
    # If we have a good candidate, do a final local polish (hill climbing)
    if best_seq is not None and time.time() < deadline:
        current = best_seq.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Single flip local search
            for i in range(N):
                if time.time() > deadline:
                    break
                current[i] *= -1
                new_score = _compute_score(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_seq = current.copy()
                    improved = True
                else:
                    current[i] *= -1 # Revert
                    
    if best_seq is None:
        best_seq = np.ones(N)
        
    return best_seq
# EVOLVE-BLOCK-END

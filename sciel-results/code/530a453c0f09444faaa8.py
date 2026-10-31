"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Deadline guard
_DEADLINE = 145.0  # seconds, leave 5s safety margin before 150s kill

def _autoconvolution(x):
    """Compute full linear autoconvolution of x."""
    n = len(x)
    size = 2 * n - 1
    # Use FFT for efficiency
    f = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac

def _score(x):
    """
    Compute the C3' metric: max |autoconvolution| / (integral)^2.
    For a step function sampled at N points with unit width, 
    integral = sum(x) * 1 (each step has width 1).
    Autoconvolution at lag k: sum_{i} x_i * x_{i+k}
    The 'maximum absolute autoconvolution' is max_k |ac_k|.
    Score = max_k |ac_k| / (sum(x))^2
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconvolution(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _construct():
    """
    Construct signed step heights to minimize max|autoconv| / (sum)^2.
    
    Strategy: 
    1. Start with random signs (+1/-1)
    2. Use local search: flip individual signs to reduce the score
    3. Use simulated annealing for escaping local minima
    4. Deadline-guarded
    """
    start_time = time.time()
    
    # Initialize with random signs
    x = np.array([random.choice([1.0, -1.0]) for _ in range(N)])
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Precompute for efficiency
    # We'll use a smarter local search
    
    # Try different random seeds for initialization
    best_overall = best_score
    best_x_overall = best_x.copy()
    
    # Simulated annealing
    current_x = x.copy()
    current_score = best_score
    T = 1.0  # initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    # Precompute autoconvolution for current state
    # For efficiency, we'll compute the score incrementally when possible
    # But for simplicity, let's just recompute each time with FFT
    
    n_iter = 0
    max_iter = 500000  # safety cap
    
    while time.time() - start_time < _DEADLINE and n_iter < max_iter:
        n_iter += 1
        
        # Choose a random index to flip
        i = random.randint(0, N - 1)
        
        # Flip sign
        old_val = current_x[i]
        current_x[i] = -old_val
        
        new_score = _score(current_x)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
                if new_score < best_overall:
                    best_overall = new_score
                    best_x_overall = current_x.copy()
        else:
            # Reject
            current_x[i] = old_val
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Check deadline
        if time.time() - start_time > _DEADLINE - 5:
            break
    
    # Return the best found
    return best_x_overall

def construct():
    """Entry point: return sequence of signed step heights."""
    return _construct()

# EVOLVE-BLOCK-END

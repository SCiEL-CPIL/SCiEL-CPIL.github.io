"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import time

def _autoconv(x):
    """Full autoconvolution of x."""
    return fftconvolve(x, x[::-1], mode='full')

def _score(x):
    """Compute max|autoconv| / (sum)^2."""
    s = x.sum()
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    return float(np.max(np.abs(ac)) / (s * s))

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin
    
    best_x = np.ones(N, dtype=np.int64)
    best_score = _score(best_x)
    
    rng = np.random.default_rng(123)
    
    # Multiple restarts with local search
    for trial in range(50):
        if time.time() > deadline - 15.0:
            break
        
        # Initialize with random ±1
        x = rng.choice([-1, 1], size=N).astype(np.float64)
        
        # Bias to have positive sum
        if x.sum() < 0:
            x *= -1
        
        cur_score = _score(x)
        
        # If sum is too small, amplify
        s = x.sum()
        if abs(s) < 20:
            # Add a constant offset to increase sum
            x += 1  # now values in {0, 2}
            cur_score = _score(x)
        
        # Local search with single-bit flips
        max_iters = 5000
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 10.0:
                break
            
            # Try flipping a random element
            idx = rng.integers(0, N)
            x[idx] *= -1
            new_score = _score(x)
            if new_score < cur_score:
                cur_score = new_score
            else:
                x[idx] *= -1  # revert
            
            # Periodically try larger moves
            if iters % 100 == 0:
                # Try adding/subtracting 1 from a random element
                idx = rng.integers(0, N)
                x[idx] += 1
                new_score = _score(x)
                if new_score < cur_score:
                    cur_score = new_score
                else:
                    x[idx] -= 1  # revert
                
                idx = rng.integers(0, N)
                x[idx] -= 1
                new_score = _score(x)
                if new_score < cur_score:
                    cur_score = new_score
                else:
                    x[idx] += 1  # revert
        
        if cur_score < best_score:
            best_score = cur_score
            best_x = x.copy()
    
    # Round to integers
    result = np.round(best_x).astype(np.int64)
    return result
# EVOLVE-BLOCK-END

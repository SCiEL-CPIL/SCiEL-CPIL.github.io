import time
import numpy as np
from scipy.signal import fftconvolve

N = 4096
DEADLINE = 140.0  # Safety margin below 150s limit

def _score(x):
    """Compute max |autocorr| / (sum x)^2. Lower is better."""
    ac = fftconvolve(x, x[::-1], mode='full')
    max_abs = np.max(np.abs(ac))
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return max_abs / (s * s)

def _eval_binary(x):
    """Evaluate a binary {-1, +1} sequence."""
    return _score(x)

def _random_binary(n):
    return np.random.choice([-1.0, 1.0], size=n)

def _hill_climb(x, time_limit=5.0, step_size=32):
    """
    Local search: try flipping contiguous blocks of size step_size,
    keep the flip if it improves the score.
    """
    best_score = _eval_binary(x)
    x = x.copy()
    start_time = time.time()
    n = len(x)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try several random block flips
        for _ in range(20):
            if time.time() - start_time >= time_limit:
                break
            pos = np.random.randint(0, max(1, n - step_size + 1))
            candidate = x.copy()
            candidate[pos:pos + step_size] *= -1
            cand_score = _eval_binary(candidate)
            if cand_score < best_score:
                x = candidate
                best_score = cand_score
                improved = True
        if not improved:
            # Try smaller step sizes
            step_size = max(1, step_size // 2)
            if step_size <= 1:
                break
    return x, best_score

def construct():
    start_time = time.time()
    best_x = None
    best_score = np.inf
    
    # Phase 1: Random restarts with hill climbing
    n_restarts = 0
    while time.time() - start_time < DEADLINE - 30:  # Leave 30s for final refinement
        x = _random_binary(N)
        x, sc = _hill_climb(x, time_limit=min(8.0, DEADLINE - (time.time() - start_time)))
        n_restarts += 1
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
        
        # Check if we're running out of time
        if time.time() - start_time > DEADLINE - 15:
            break
    
    if best_x is None:
        # Fallback: just return ones (shouldn't happen)
        return np.ones(N)
    
    # Phase 2: Intensive refinement on the best solution
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        x, sc = _hill_climb(best_x, time_limit=remaining - 2, step_size=64)
        if sc < best_score:
            best_score = sc
            best_x = x
    
    return best_x

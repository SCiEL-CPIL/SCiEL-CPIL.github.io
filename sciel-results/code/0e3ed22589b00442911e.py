"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(f):
    """Compute full autoconvolution of f."""
    return np.correlate(f, f, mode='full')


def _score(f, exclude_lag0=True):
    """Compute the score: max |autoconv| / (sum f)^2.
    If exclude_lag0, exclude lag 0 (index N-1 in full mode)."""
    ac = _autoconv(f)
    s = np.sum(f)
    if s == 0:
        return float('inf')
    if exclude_lag0:
        # Full mode: lag 0 is at index N-1
        ac_no0 = np.delete(ac, N - 1)
        max_abs = np.max(np.abs(ac_no0))
    else:
        max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def construct():
    """
    Returns a signed step function minimizing max absolute autoconv / (sum f)^2.
    Uses stochastic local search with block flips and sign changes.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Start with a random ±1 sequence
    best_f = np.random.choice([-1.0, 1.0], size=N)
    best_score = _score(best_f, exclude_lag0=True)
    
    # Also try all-ones as baseline
    ones_f = np.ones(N)
    ones_score = _score(ones_f, exclude_lag0=True)
    if ones_score < best_score:
        best_f = ones_f.copy()
        best_score = ones_score
    
    rng = np.random.RandomState(42)
    
    # Local search with block flips
    while time.time() < deadline:
        # Try different block sizes
        block_size = rng.randint(1, 50)
        
        for _ in range(100):
            # Random block flip
            start = rng.randint(0, N - block_size + 1)
            f_trial = best_f.copy()
            f_trial[start:start + block_size] *= -1
            
            s = np.sum(f_trial)
            if abs(s) < 1e-6:
                continue
            
            sc = _score(f_trial, exclude_lag0=True)
            if sc < best_score:
                best_f = f_trial
                best_score = sc
        
        # Also try single element sign flips
        for _ in range(200):
            idx = rng.randint(0, N)
            f_trial = best_f.copy()
            f_trial[idx] *= -1
            
            s = np.sum(f_trial)
            if abs(s) < 1e-6:
                continue
            
            sc = _score(f_trial, exclude_lag0=True)
            if sc < best_score:
                best_f = f_trial
                best_score = sc
    
    return best_f
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def _objective(f):
    """Compute max(|autoconv|) / (sum(f))^2"""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    auto = fftconvolve(f, f, mode='full')
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def _try_candidate(f):
    return _objective(f)


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    best_f = np.ones(N)
    best_score = _objective(best_f)
    
    # Try random ±1 sequences
    rng = np.random.default_rng(42)
    for i in range(200):
        if time.time() > deadline:
            break
        f = rng.choice([-1, 1], size=N)
        score = _objective(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Try sequences with mostly +1 and some -1
    for n_neg in range(0, 50):
        if time.time() > deadline:
            break
        for trial in range(20):
            if time.time() > deadline:
                break
            f = np.ones(N)
            neg_idx = rng.choice(N, size=n_neg, replace=False)
            f[neg_idx] = -1
            score = _objective(f)
            if score < best_score:
                best_score = score
                best_f = f.copy()
    
    # Local search on best candidate
    if time.time() < deadline - 10:
        f = best_f.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                old = f[i]
                f[i] = -old
                new_score = _objective(f)
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] = old
    
    return best_f
# EVOLVE-BLOCK-END

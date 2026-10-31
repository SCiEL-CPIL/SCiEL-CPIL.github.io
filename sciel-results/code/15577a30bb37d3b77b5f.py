"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Compute max absolute value of linear autoconvolution."""
    n = len(x)
    # Linear autoconvolution via FFT
    size = 2 * n - 1
    f = np.fft.fft(x, size=size)
    ac = np.fft.ifft(f * np.conj(f)).real
    return np.max(np.abs(ac))


def _score_signal(x):
    """Compute max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    return _autoconv_max_abs(x) / (s * s)


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start with all-ones (known optimal by Cauchy-Schwarz)
    x = np.ones(N)
    best_score = _score_signal(x)
    best_x = x.copy()
    
    # Try small perturbations to see if we can do better
    # Theory says constant is optimal, but let's verify numerically
    # and try local search just in case
    
    rng = np.random.default_rng(42)
    
    # Try random binary-like signals with nonzero sum
    for trial in range(20):
        if time.time() > deadline:
            break
        # Random signal with mostly +1 and some -1 to keep sum positive
        x_try = np.ones(N)
        n_flips = rng.integers(1, N // 2)
        idx = rng.choice(N, size=n_flips, replace=False)
        x_try[idx] = -1
        s = np.sum(x_try)
        if abs(s) < 1:
            continue
        sc = _score_signal(x_try)
        if sc < best_score:
            best_score = sc
            best_x = x_try.copy()
    
    # Local search: flip individual elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = best_x[i]
            # Try flipping sign
            best_x[i] = -old_val
            sc = _score_signal(best_x)
            if sc < best_score - 1e-15:
                best_score = sc
                improved = True
            else:
                best_x[i] = old_val
    
    # Try continuous values: small perturbations around 1
    for trial in range(50):
        if time.time() > deadline:
            break
        x_try = np.ones(N)
        # Small random perturbations
        x_try += rng.normal(0, 0.1, size=N)
        # Ensure positive sum
        s = np.sum(x_try)
        if s < 1:
            x_try = x_try - np.min(x_try) + 0.5
        sc = _score_signal(x_try)
        if sc < best_score:
            best_score = sc
            best_x = x_try.copy()
    
    return best_x
# EVOLVE-BLOCK-END


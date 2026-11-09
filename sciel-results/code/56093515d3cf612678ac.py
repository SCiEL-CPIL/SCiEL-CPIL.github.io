"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft


def _autocorr_full(x):
    """Compute full autocorrelation via FFT (linear, not circular)."""
    n = len(x)
    # Need size >= 2n-1 for linear autocorrelation
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = fft(x, size)
    ac = ifft(X * np.conj(X)).real[:2 * n - 1]
    return ac


def _objective(x):
    """Compute max |autocorrelation| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    ac = _autocorr_full(x)
    return np.max(np.abs(ac)) / (s * s)


def _try_candidate(x, best_score, best_x, deadline):
    """Try a candidate, update best if better. Returns (best_score, best_x)."""
    if time.time() > deadline:
        return best_score, best_x
    sc = _objective(x)
    if sc < best_score:
        best_score = sc
        best_x = x.copy()
    return best_score, best_x


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s kill
    
    best_score = float('inf')
    best_x = np.ones(N)
    
    # First, evaluate the baseline
    best_score, best_x = _try_candidate(np.ones(N), best_score, best_x, deadline)
    
    # Try structured patterns
    # 1. Rademacher-like: alternating blocks
    for block_size in [1, 2, 4, 8, 16, 32]:
        if time.time() > deadline:
            break
        x = np.ones(N)
        for i in range(0, N, block_size):
            end = min(i + block_size, N)
            if (i // block_size) % 2 == 1:
                x[i:end] = -1
        best_score, best_x = _try_candidate(x, best_score, best_x, deadline)
    
    # 2. Walsh-Hadamard-like patterns
    # 3. Random restarts with local search
    rng = np.random.RandomState(42)
    
    # Try random sign patterns
    for trial in range(200):
        if time.time() > deadline:
            break
        x = rng.choice([-1, 1], size=N)
        best_score, best_x = _try_candidate(x, best_score, best_x, deadline)
    
    # Local search on the best found: flip individual elements
    if time.time() < deadline:
        x = best_x.copy()
        improved = True
        max_iter = 5000
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False
            # Try flipping each element
            for i in range(N):
                if time.time() > deadline:
                    break
                x[i] = -x[i]
                sc = _objective(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = -x[i]  # revert
            if not improved:
                break
    
    # Try a few more random restarts with local search
    for trial in range(50):
        if time.time() > deadline:
            break
        x = rng.choice([-1, 1], size=N)
        sc = _objective(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
            # Quick local search
            if time.time() < deadline:
                improved = True
                for it in range(1000):
                    if time.time() > deadline:
                        break
                    if not improved:
                        break
                    improved = False
                    # Random subset of positions to try flipping
                    idxs = rng.choice(N, size=min(50, N), replace=False)
                    for i in idxs:
                        if time.time() > deadline:
                            break
                        x[i] = -x[i]
                        sc2 = _objective(x)
                        if sc2 < best_score:
                            best_score = sc2
                            best_x = x.copy()
                            improved = True
                        else:
                            x[i] = -x[i]
    
    return best_x
# EVOLVE-BLOCK-END

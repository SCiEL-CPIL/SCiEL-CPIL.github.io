"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time


def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]


def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Helper: try a candidate and update best
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Phase 1: Try various constant-like patterns with small perturbations
    # All ones is the baseline
    # Try sequences with small deviations from constant
    
    # Phase 2: Random continuous values
    rng = np.random.default_rng(42)
    
    # Try random positive values (since negative sum might cause issues)
    for i in range(3000):
        if time.time() > deadline:
            break
        # Random positive values
        x = rng.uniform(0.5, 1.5, N)
        try_candidate(x)
        
        # Random with some structure
        if i % 2 == 0:
            x = 1.0 + 0.1 * rng.normal(0, 1, N)
            try_candidate(x)
    
    # Phase 3: Local optimization using gradient-like approach
    # Start from best and try small perturbations
    if best_score < _score(np.ones(N)) * 0.99:  # Only if we improved
        x = best_x.copy()
        sc = best_score
        for iteration in range(50):
            if time.time() > deadline:
                break
            improved = False
            # Try perturbing random subsets
            for _ in range(200):
                if time.time() > deadline:
                    break
                n_perturb = rng.integers(1, 10)
                perturb_idx = rng.choice(N, size=n_perturb, replace=False)
                x_new = x.copy()
                x_new[perturb_idx] += rng.normal(0, 0.1, n_perturb)
                sc_new = _score(x_new)
                if sc_new < sc - 1e-10:
                    x = x_new
                    sc = sc_new
                    improved = True
                    best_x = x.copy()
                    best_score = sc
            if not improved:
                break
    
    # Phase 4: Systematic local search on best
    if time.time() < deadline - 10.0:
        x = best_x.copy()
        sc = best_score
        # Try flipping/perturbing each position
        for iteration in range(10):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                # Try small perturbation
                orig_val = x[i]
                for delta in [0.01, -0.01, 0.1, -0.1]:
                    x[i] = orig_val + delta
                    sc_new = _score(x)
                    if sc_new < sc - 1e-10:
                        sc = sc_new
                        best_x = x.copy()
                        best_score = sc
                        improved = True
                        break
                else:
                    x[i] = orig_val  # revert if no improvement
            if not improved:
                break
    
    # Phase 5: Try structured patterns with continuous values
    # Low-frequency patterns
    for i in range(500):
        if time.time() > deadline:
            break
        freq = rng.uniform(0.01, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        x = 1.0 + 0.2 * np.sin(2 * np.pi * freq * np.arange(N) + phase)
        try_candidate(x)
        
        # Two-frequency patterns
        if i % 2 == 0:
            freq2 = rng.uniform(0.01, 0.5)
            phase2 = rng.uniform(0, 2 * np.pi)
            x = 1.0 + 0.1 * np.sin(2 * np.pi * freq * np.arange(N) + phase) + \
                0.1 * np.sin(2 * np.pi * freq2 * np.arange(N) + phase2)
            try_candidate(x)
    
    # Phase 6: More aggressive local search with larger steps
    if time.time() < deadline - 5.0:
        x = best_x.copy()
        sc = best_score
        for iteration in range(30):
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try larger random perturbations
            for _ in range(100):
                if time.time() > deadline - 2.0:
                    break
                n_perturb = rng.integers(1, 20)
                perturb_idx = rng.choice(N, size=n_perturb, replace=False)
                x_new = x.copy()
                x_new[perturb_idx] += rng.normal(0, 0.5, n_perturb)
                sc_new = _score(x_new)
                if sc_new < sc - 1e-10:
                    x = x_new
                    sc = sc_new
                    improved = True
                    best_x = x.copy()
                    best_score = sc
            if not improved:
                break
    
    return best_x
# EVOLVE-BLOCK-END

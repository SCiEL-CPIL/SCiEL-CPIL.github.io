"""Rewritten program: minimize max|autocorr| / autocorr[0] via random search + local search."""
import numpy as np
import time

N = 400

def _score(x):
    """Compute max|autocorr| / autocorr[0]. Returns inf if autocorr[0] ~ 0."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    # Full autocorrelation via FFT
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    ac0 = ac[0]
    if abs(ac0) < 1e-12:
        return float('inf')
    return float(np.max(np.abs(ac)) / ac0)

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Start with all-ones (score 1.0)
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Phase 1: Random search with various initializations
    # Try different families of signals
    iterations = 0
    while time.time() < deadline - 30.0:
        iterations += 1
        # Generate random signed sequence
        # Mix strategies:
        # 1. Random signs
        # 2. Random signs with decay
        # 3. Random continuous values
        strategy = iterations % 3
        
        if strategy == 0:
            # Random ±1
            x = rng.choice([-1.0, 1.0], size=N)
        elif strategy == 1:
            # Random signs with exponential decay envelope
            envelope = np.exp(-np.linspace(0, 2, N))
            x = rng.choice([-1.0, 1.0], size=N) * envelope
        else:
            # Random continuous in [-1, 1]
            x = rng.uniform(-1.0, 1.0, size=N)
        
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
            
        # If we found something good, do local refinement
        if s < best_score * 1.01 and s < 0.99:
            # Local search: flip random signs
            for _ in range(50):
                if time.time() > deadline - 30.0:
                    break
                idx = rng.integers(0, N)
                x[idx] = -x[idx]
                s2 = _score(x)
                if s2 < best_score:
                    best_score = s2
                    best_x = x.copy()
    
    # Phase 2: More aggressive local search on best found
    while time.time() < deadline - 5.0:
        improved = False
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            idx = rng.integers(0, N)
            best_x[idx] = -best_x[idx]
            s = _score(best_x)
            if s < best_score:
                best_score = s
                improved = True
            else:
                best_x[idx] = -best_x[idx]  # revert
        
        if not improved:
            # Try a few random restarts of local search
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                x = best_x.copy()
                # Perturb a few elements
                for _ in range(5):
                    idx = rng.integers(0, N)
                    x[idx] = -x[idx]
                s = _score(x)
                if s < best_score:
                    best_score = s
                    best_x = x.copy()
                    break
    
    return best_x

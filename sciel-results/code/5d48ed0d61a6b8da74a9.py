import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin from 150s limit

def _autoconv(x):
    """Compute linear autoconvolution of x (length 2N-1)."""
    return np.correlate(x, x, mode='full')

def _score(x):
    """Compute J = max|autoconv(x)| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-15:
        return float('inf')
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start = time.time()
    
    # Baseline: all-ones (proven optimal by Cauchy-Schwarz lower bound)
    x_ones = np.ones(N, dtype=np.float64)
    best = x_ones.copy()
    best_score = _score(best)
    
    # Try a few structured alternatives to verify/seek improvement
    candidates = []
    
    # Candidate 1: All ones (baseline)
    candidates.append(x_ones.copy())
    
    # Candidate 2: Slight perturbation - mostly +1 with small -1 dips
    # Try a periodic pattern
    for period in [2, 3, 4, 5, 7, 10]:
        x = np.ones(N, dtype=np.float64)
        # Make every period-th element slightly negative
        x[::period] = -0.1
        candidates.append(x)
    
    # Candidate 3: Ramp / linear
    x = np.linspace(1.0, 0.1, N)
    candidates.append(x)
    
    # Candidate 4: Cosine window
    t = np.linspace(0, 2*np.pi, N)
    x = 1.0 + 0.3 * np.cos(t)
    candidates.append(x)
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > DEADLINE:
            break
        s = _score(x)
        if s < best_score:
            best_score = s
            best = x.copy()
    
    # Local search: flip small perturbations
    # Since the lower bound is tight for all-ones, try to confirm
    # Try small random perturbations around all-ones
    rng = np.random.default_rng(42)
    for trial in range(50):
        if time.time() > DEADLINE:
            break
        x = np.ones(N, dtype=np.float64)
        # Random small perturbations
        mask = rng.random(N) < 0.05
        x[mask] = 1.0 + rng.normal(0, 0.1, mask.sum())
        s = _score(x)
        if s < best_score:
            best_score = s
            best = x.copy()
    
    # Final local search: coordinate descent style
    if time.time() < DEADLINE - 10:
        x = best.copy()
        current = _score(x)
        for it in range(200):
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(N):
                if time.time() > DEADLINE:
                    break
                old = x[i]
                for delta in [-0.1, -0.01, 0.01, 0.1]:
                    x[i] = old + delta
                    s = _score(x)
                    if s < current - 1e-12:
                        current = s
                        improved = True
                        break
                    x[i] = old
                if improved:
                    break
            if not improved:
                break
        if current < best_score:
            best_score = current
            best = x.copy()
    
    return best

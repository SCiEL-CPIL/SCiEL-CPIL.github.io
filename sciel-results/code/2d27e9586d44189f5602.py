"""New design: signed step heights minimizing max |autoconv| / (sum x)^2."""
import time
import numpy as np
from scipy import signal

N = 400

# EVOLVE-BLOCK-START


def _score(x):
    """Compute the objective: max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    ac = signal.fftconvolve(x, x, mode='full')
    return np.max(np.abs(ac)) / (s * s)


def construct():
    """
    Build a signed step-height vector of length N that minimizes the ratio
    max |autoconv| / (sum x)^2.
    
    Strategy: start from ones (theoretical lower bound 1/N if k=0 included),
    then try small perturbations and random restarts to see if we can do better.
    """
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try a few random sign-flip perturbations from ones
    rng = np.random.default_rng(42)
    n_attempts = 200
    for i in range(n_attempts):
        if time.time() > deadline:
            break
        x = np.ones(N)
        # Flip a random subset of signs
        flip_mask = rng.random(N) < 0.3
        x[flip_mask] = -1.0
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Also try: ones with a few -1s to reduce sidelobes
    for k in range(0, 20):
        if time.time() > deadline:
            break
        for trial in range(5):
            x = np.ones(N)
            if k > 0:
                idx = rng.choice(N, size=k, replace=False)
                x[idx] = -1.0
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
    
    return best_x

# EVOLVE-BLOCK-END

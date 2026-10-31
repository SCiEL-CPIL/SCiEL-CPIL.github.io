"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def autoconv_peak(x):
        """Compute max absolute autoconvolution value."""
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))

    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_peak(x) / (s * s)

    # Strategy: full signed binary sequences {-1, 1}
    # For unit magnitude steps, sum is bounded by N, and we want to maximize |sum|
    # while minimizing max|autoconv|
    
    # Start with all ones (constant) as baseline
    best_x = np.ones(N)
    best_val = score(best_x)

    rng = np.random.default_rng(42)

    # Try random binary sign patterns with local search
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() - start_time > deadline:
            break

        # Random sign pattern
        x = rng.choice([-1.0, 1.0], size=N)
        
        val = score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

        # Local search: try flipping individual elements
        improved = True
        max_iters = 200
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                break
            if not improved:
                break
            improved = False

            # Try random subset of positions
            trial_indices = rng.choice(N, size=min(30, N), replace=False)
            for idx in trial_indices:
                if time.time() - start_time > deadline:
                    break
                orig = x[idx]
                # Try flipping sign
                x[idx] = -orig
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x.copy()
                    improved = True
                else:
                    x[idx] = orig

    # Final polish: try small perturbations on the best solution
    if time.time() - start_time < deadline - 10:
        x = best_x.copy()
        for trial in range(200):
            if time.time() - start_time > deadline - 5:
                break
            idx = rng.choice(N)
            orig = x[idx]
            x[idx] = -orig
            new_val = score(x)
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_x = x.copy()
            else:
                x[idx] = orig

    return best_x
# EVOLVE-BLOCK-END

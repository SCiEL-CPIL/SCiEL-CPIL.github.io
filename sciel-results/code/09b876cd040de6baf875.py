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

    # Strategy: sparse random signed sequences
    # Try different densities and pick the best
    best_x = np.ones(N)
    best_val = score(best_x)

    rng = np.random.default_rng(42)

    # Try a range of sparsity levels
    for density in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        if time.time() - start_time > deadline:
            break

        # Multiple restarts at each density
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - start_time > deadline:
                break

            # Create sparse signed sequence
            mask = rng.random(N) < density
            signs = rng.choice([-1.0, 1.0], size=N)
            x = np.zeros(N)
            x[mask] = signs[mask]

            # Ensure non-zero sum
            if abs(np.sum(x)) < 1e-6:
                # Flip a few elements to break symmetry
                flip_idx = rng.choice(N, size=max(1, N // 10), replace=False)
                x[flip_idx] *= -1

            val = score(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()

            # Local search: try flipping individual elements
            if time.time() - start_time > deadline:
                break

            improved = True
            max_iters = 500
            for it in range(max_iters):
                if time.time() - start_time > deadline:
                    break
                if not improved:
                    break
                improved = False

                # Try random subset of positions
                trial_indices = rng.choice(N, size=min(50, N), replace=False)
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

    # Final polish: try continuous adjustments on the best sparse solution
    if time.time() - start_time < deadline - 10:
        x = best_x.copy()
        # Try scaling individual non-zero elements
        nonzero_idx = np.nonzero(x)[0]
        if len(nonzero_idx) > 0:
            for trial in range(100):
                if time.time() - start_time > deadline - 5:
                    break
                idx = rng.choice(nonzero_idx)
                orig = x[idx]
                # Try scaling by random factor
                factor = rng.uniform(0.5, 1.5)
                x[idx] = orig * factor
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x.copy()
                else:
                    x[idx] = orig

    # Ensure the result has reasonable sum
    if abs(np.sum(best_x)) < 1e-6:
        best_x = np.ones(N)

    return best_x
# EVOLVE-BLOCK-END

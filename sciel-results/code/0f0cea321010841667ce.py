"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    h = 0.5 / N  # interval width

    def compute_value(x):
        """Compute V = max_disc / (s^2 * dx). Lower is better."""
        x = np.asarray(x, dtype=np.float64)
        x[x < 0] = 0
        s = np.sum(x)
        if s == 0:
            return float('inf')
        # Linear autoconvolution via FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
        max_disc = np.max(conv)
        return max_disc / (s * s * h)

    # Initial guess: Gaussian with moderate sigma
    mu = (N - 1) / 2.0
    sigma = 10.0
    x = np.exp(-0.5 * ((np.arange(N) - mu) / sigma) ** 2)
    x = np.maximum(x, 0)

    best_x = x.copy()
    best_val = compute_value(best_x)

    # Coordinate descent: modify one element at a time, keep best
    # Use random order to avoid getting stuck
    rng = np.random.RandomState(42)
    order = rng.permutation(N)

    # Multiple passes
    n_passes = 20
    for p in range(n_passes):
        if time.time() > deadline - 15:
            break
        improved = False
        for idx in order:
            if time.time() > deadline - 15:
                break
            old = best_x[idx]
            # Try several multipliers
            for mult in [0.5, 0.75, 1.25, 1.5, 0.25, 2.0]:
                candidate = best_x.copy()
                candidate[idx] = old * mult
                if candidate[idx] < 0:
                    continue
                v = compute_value(candidate)
                if v < best_val:
                    best_val = v
                    best_x = candidate
                    improved = True
            # Also try setting to 0
            if old > 0:
                candidate = best_x.copy()
                candidate[idx] = 0
                v = compute_value(candidate)
                if v < best_val:
                    best_val = v
                    best_x = candidate
                    improved = True
        if not improved:
            break

    # Refinement: try small perturbations around current best
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            idx = rng.randint(N)
            old = best_x[idx]
            for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                candidate = best_x.copy()
                candidate[idx] = max(0, old + delta)
                v = compute_value(candidate)
                if v < best_val:
                    best_val = v
                    best_x = candidate

    return list(best_x)
# EVOLVE-BLOCK-END

import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio: (L2^2) / (L1 * sup|autoconv|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = float(np.sum(h**2))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    n = len(h)
    # Use FFT for autoconvolution
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, fft_size)
    C = np.fft.irfft(H * H, fft_size)
    # C contains circular autoconv; extract linear part
    c = C[:size]
    sup_c = float(np.max(np.abs(c)))
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s

    # Start with uniform
    h = np.ones(N)
    best_h = h.copy()
    best_ratio = _compute_ratio(best_h)

    # Local optimization
    np.random.seed(42)
    step_size = 0.1
    max_iters = 20000

    for i in range(max_iters):
        if time.time() > deadline:
            break

        # Random perturbation of a random coordinate
        h_trial = best_h.copy()
        idx = np.random.randint(0, N)
        h_trial[idx] += np.random.randn() * step_size
        h_trial = np.maximum(h_trial, 0.0)

        l1 = float(np.sum(h_trial))
        if l1 < 1e-15:
            continue
        # Scale invariant, but keep L1=1 for consistency
        h_trial = h_trial / l1

        ratio = _compute_ratio(h_trial)
        if ratio > best_ratio:
            best_h = h_trial.copy()
            best_ratio = ratio
            step_size = max(0.005, step_size * 0.95)
        else:
            step_size = min(0.5, step_size * 1.03)

    return list(best_h)
# EVOLVE-BLOCK-END

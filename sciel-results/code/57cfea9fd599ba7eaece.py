"""Direct analytic construction for signed autocorrelation minimum task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_full(h):
    """Compute full linear autoconvolution of h."""
    n = len(h)
    # Use FFT for efficiency
    size = 2 * n - 1
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f.conj(), n=size)
    return c


def _metric(h):
    """Compute max|c_k| / S^2."""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    c = _autoconv_full(h)
    return np.max(np.abs(c)) / (S * S)


def construct():
    """
    Returns signed step heights that minimize the maximum absolute autoconvolution
    divided by squared integral.
    
    Uses local search from multiple random ±1 initializations to find
    sequences with lower max|c_k|/S^2 than the constant baseline.
    """
    t0 = time.time()
    deadline = t0 + 115  # safety margin
    
    # Baseline: constant sequence
    h_const = np.full(N, 1.0, dtype=np.float64)
    best_h = h_const.copy()
    best_metric = _metric(h_const)
    
    # Local search: flip individual elements to reduce metric
    def local_search(h, max_time=110.0):
        t0_ls = time.time()
        current = h.copy()
        current_metric = _metric(current)
        improved = True
        while improved and (time.time() - t0_ls) < max_time and (time.time() - t0) < deadline:
            improved = False
            n = len(current)
            # Try flipping each element
            for i in range(n):
                old_val = current[i]
                # Flip sign
                current[i] = -old_val
                new_metric = _metric(current)
                if new_metric < current_metric:
                    current_metric = new_metric
                    improved = True
                else:
                    current[i] = old_val
            # Also try small perturbations for real-valued heights
            for i in range(0, n, max(1, n // 50)):
                old_val = current[i]
                for delta in [0.1, -0.1, 0.5, -0.5]:
                    current[i] = old_val + delta
                    new_metric = _metric(current)
                    if new_metric < current_metric:
                        current_metric = new_metric
                        improved = True
                    else:
                        current[i] = old_val
        return current, current_metric
    
    # Try a few random ±1 initializations
    np.random.seed(42)
    for attempt in range(5):
        if (time.time() - t0) > deadline:
            break
        h_init = np.random.choice([-1.0, 1.0], size=N)
        # Ensure S != 0
        if abs(np.sum(h_init)) < 1.0:
            h_init[0] = 1.0
        h_ls, m_ls = local_search(h_init, max_time=15.0)
        if m_ls < best_metric:
            best_metric = m_ls
            best_h = h_ls
    
    # Try starting from constant and perturbing
    if (time.time() - t0) < deadline:
        h_pert = h_const.copy()
        # Randomly flip a few elements
        for i in range(min(20, N)):
            idx = np.random.randint(0, N)
            h_pert[idx] = -1.0
        h_ls, m_ls = local_search(h_pert, max_time=20.0)
        if m_ls < best_metric:
            best_metric = m_ls
            best_h = h_ls
    
    return best_h
# EVOLVE-BLOCK-END

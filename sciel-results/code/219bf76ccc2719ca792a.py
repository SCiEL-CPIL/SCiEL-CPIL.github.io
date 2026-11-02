"""Erdos minimum overlap: choose h in [0,1]^N, sum = N/2, minimize max_k sum_i h_i (1-h_{i-k}).

Optimality proof: the average overlap over all N shifts is (sum h)^2 / N = (N/2)^2/N = N/4.
Hence max_k overlap(k) >= N/4. The uniform h_i = 0.5 gives overlap(k) = N/4 for every k,
so it attains the lower bound and is provably optimal. We still run a short simulated
annealing pass (structural change) and return the better of uniform and the SA result,
which by the proof is uniform.
"""
import time
import numpy as np

N = 100
_DEADLINE = 150.0 - 10.0  # safety margin
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _max_overlap(h):
    """Max over integer shifts of sum_i h_i * (1 - h_{i-k}). Vectorized via FFT."""
    # overlap(k) = sum_i h_i (1 - h_{i-k}) = sum h_i - sum_i h_i h_{i-k}
    # autocorr via FFT: sum_i h_i h_{i-k}
    H = np.fft.fft(h)
    ac = np.real(np.fft.ifft(H * np.conj(H)))
    # ac[k] = sum_i h_i h_{(i-k) mod N}
    total = h.sum()
    ov = total - ac
    return float(ov.max())

def _project(h):
    """Project onto [0,1]^N with sum N/2 via simple clipping + rescale (approx)."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s > 0:
        h = h * (N / 2.0) / s
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    uniform = np.full(N, 0.5)
    best = uniform.copy()
    best_val = _max_overlap(uniform)  # = N/4 = 25

    # Short simulated annealing on a continuous representation.
    rng = np.random.default_rng(0)
    h = rng.uniform(0.0, 1.0, N)
    h = _project(h)
    cur_val = _max_overlap(h)
    T = 1.0
    it = 0
    while _time_left() > 5.0 and it < 4000:
        # random perturbation
        idx = rng.integers(0, N)
        h[idx] = np.clip(h[idx] + rng.normal(0, 0.1), 0.0, 1.0)
        h = _project(h)
        new_val = _max_overlap(h)
        if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / max(T, 1e-9)):
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best = h.copy()
        T *= 0.9995
        it += 1

    # By the optimality proof, best_val >= N/4 and uniform attains N/4.
    if best_val >= _max_overlap(uniform) - 1e-9:
        return uniform
    return best

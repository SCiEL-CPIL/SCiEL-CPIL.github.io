"""Direct 600-dim simulated annealing with FFT-accelerated autoconvolution.

Instead of parameterising the step function as a small number of pulses,
we directly optimise all 600 nonnegative heights.  The autoconvolution peak
is computed via FFT (O(N log N)) enabling thousands of SA iterations within
the time budget.  Multi-start with diverse initial configurations explores
the landscape broadly.
"""
import time
import numpy as np

N = 600
DX = 0.5 / N

# FFT size for linear autoconvolution (need at least 2N-1, use next power of 2)
_FFT_N = 1024  # next power of 2 >= 2*600-1 = 1199? No, 1024 < 1199. Use 2048.
_FFT_N = 2048

_DEADLINE = 0.0
_TIME_MARGIN = 15.0


def _time_left():
    return _DEADLINE - _TIME_MARGIN - time.time()


def _score_fft(heights):
    """Normalised max of linear autoconvolution via FFT. Returns peak/N."""
    # Zero-pad to _FFT_N for linear convolution
    h_pad = np.zeros(_FFT_N, dtype=np.float64)
    h_pad[:N] = heights
    H = np.fft.rfft(h_pad)
    conv = np.fft.irfft(H * H, n=_FFT_N)
    # Linear autoconv has length 2N-1; the peak should be in [0, 2N-2]
    # For nonnegative heights, the max is typically near the center
    peak = conv[:2 * N - 1].max()
    return float(peak / N)


def _score_numpy(heights):
    """Fallback: direct np.convolve for verification."""
    conv = np.convolve(heights, heights, mode='full')
    return float(conv.max() / N)


def _sa_search(initial_heights, n_iter, T_start, T_end, rng):
    """Simulated annealing on the 600-dim height vector."""
    global _DEADLINE
    current = initial_heights.copy()
    current_score = _score_fft(current)
    best = current.copy()
    best_score = current_score

    # Cooling schedule
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, n_iter - 1))

    for it in range(n_iter):
        if _time_left() <= 1.0:
            break

        # Perturb: Gaussian noise on a random subset of indices
        n_perturb = rng.integers(5, 30)
        idx = rng.choice(N, size=n_perturb, replace=False)
        noise_scale = 0.5 + 1.0 * (T / T_start)  # scale with temperature
        current[idx] += rng.normal(0, noise_scale, size=n_perturb)
        # Enforce nonnegativity
        current[idx] = np.maximum(current[idx], 0.0)

        new_score = _score_fft(current)

        # Accept if better, or with probability exp(-delta/T)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current.copy()
        else:
            delta = new_score - current_score
            if delta < 0.01:  # small worsening, more likely to accept
                accept_prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < accept_prob:
                    current_score = new_score

        T *= cooling

    return best, best_score


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 135.0

    rng = np.random.default_rng(42)

    best_heights = np.ones(N, dtype=float)
    best_score = _score_fft(best_heights)  # = 1.0

    # Multiple diverse initial configurations
    # Strategy 1: uniform (baseline)
    # Strategy 2: two blocks (high on left, low on right)
    # Strategy 3: sinusoidal modulation
    # Strategy 4: random sparse peaks
    # Strategy 5: smooth bump

    initializations = []

    # 1: uniform
    initializations.append(np.ones(N, dtype=float))

    # 2: two-block step
    h2 = np.ones(N, dtype=float)
    h2[:N // 2] = 2.0
    h2[N // 2:] = 0.5
    initializations.append(h2)

    # 3: sinusoidal
    x = np.linspace(-1, 1, N)
    h3 = 1.0 + 0.8 * np.sin(2 * np.pi * x)
    h3 = np.maximum(h3, 0.0)
    initializations.append(h3)

    # 4: random sparse peaks
    h4 = np.zeros(N, dtype=float)
    n_peaks = 5
    peak_positions = rng.choice(N, size=n_peaks, replace=False)
    for p in peak_positions:
        width = rng.integers(3, 10)
        amp = rng.uniform(1.5, 3.0)
        lo = max(0, p - width)
        hi = min(N, p + width + 1)
        h4[lo:hi] += amp
    initializations.append(h4)

    # 5: smooth Gaussian bump at center
    x5 = np.linspace(-1, 1, N)
    h5 = np.exp(-5.0 * x5 ** 2) + 0.3
    initializations.append(h5)

    # 6: anti-symmetric-ish (high at edges, low in middle)
    h6 = np.ones(N, dtype=float)
    edge_width = N // 4
    h6[:edge_width] = 2.5
    h6[-edge_width:] = 2.5
    h6[edge_width:N - edge_width] = 0.3
    initializations.append(h6)

    # 7: random smooth
    h7_base = rng.uniform(0.5, 1.5, N)
    # Smooth it
    kernel = np.ones(11) / 11.0
    h7 = np.convolve(h7_base, kernel, mode='same')
    initializations.append(h7)

    # 8: three alternating blocks
    h8 = np.ones(N, dtype=float)
    block = N // 3
    h8[:block] = 2.0
    h8[block:2 * block] = 0.5
    h8[2 * block:] = 2.0
    initializations.append(h8)

    # Run SA on each initialization
    n_iter = 3000
    T_start = 0.05
    T_end = 1e-5

    for init_idx, h0 in enumerate(initializations):
        if _time_left() <= 5.0:
            break

        h_best, s_best = _sa_search(h0, n_iter, T_start, T_end, rng)

        if s_best < best_score:
            best_score = s_best
            best_heights = h_best

        # If we have time, do a second pass with different params
        if _time_left() > 30.0 and init_idx < 4:
            h_best2, s_best2 = _sa_search(h0, n_iter, T_start * 1.5, T_end * 0.1, rng)
            if s_best2 < best_score:
                best_score = s_best2
                best_heights = h_best2

    # Final refinement: a few more SA iterations from the best solution
    if _time_left() > 10.0:
        h_ref, s_ref = _sa_search(best_heights, 2000, 0.02, 1e-6, rng)
        if s_ref < best_score:
            best_score = s_ref
            best_heights = h_ref

    # Return as a plain sequence of floats (nonnegative step heights).
    result = [max(0.0, float(h)) for h in best_heights]
    return result

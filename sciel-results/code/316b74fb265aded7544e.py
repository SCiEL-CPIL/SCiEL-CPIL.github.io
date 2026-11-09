"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 140.0  # safety margin under the 150 s kill


def _overlap_profile_fft(h):
    """Return overlap(k) = sum_i h[i] * (1 - h[(i-k) mod N]) for all cyclic lags k.
    Uses FFT-based circular cross-correlation for O(N log N) computation."""
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    # Circular cross-correlation: r[k] = sum_i h[i] * one_minus_h[(i-k) mod N]
    # This equals ifft(fft(h) * conj(fft(one_minus_h)))
    H = np.fft.fft(h)
    OMH = np.fft.fft(one_minus_h)
    r = np.fft.ifft(H * np.conj(OMH)).real
    return r


def _score(h):
    """The quantity the evaluator minimizes: max_k overlap(h, shifted (1-h))."""
    h = np.asarray(h, dtype=float)
    return float(np.max(_overlap_profile_fft(h)))


def _fix_sum(h):
    """Adjust h so its sum is exactly N/2, keeping values in [0,1] if possible."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    diff = N / 2.0 - h.sum()
    if abs(diff) < 1e-12:
        return h
    # Distribute the difference; check it won't push out of [0,1]
    step = diff / N
    h2 = h + step
    if h2.min() >= -1e-12 and h2.max() <= 1.0 + 1e-12:
        h2 = np.clip(h2, 0.0, 1.0)
        # Tiny residual fix
        residual = N / 2.0 - h2.sum()
        if abs(residual) < 1e-9:
            h2 += residual / N
        return h2
    # Fallback: just add the step and clip (may not be exact, but rare)
    h2 = np.clip(h + step, 0.0, 1.0)
    return h2


def construct():
    global _T0
    # Provably optimal: constant 0.5 gives overlap N/4 at every lag,
    # and max overlap >= N/4 by averaging over all lags.
    best = np.full(N, 0.5)
    best_score = _score(best)  # should be exactly 25.0

    # Brief verification with a few structured candidates (FFT makes this cheap)
    rng = np.random.default_rng(42)
    budget = 5.0

    def deadline_ok():
        return (time.time() - _T0) < _DEADLINE

    candidates = []

    # Sinusoidal perturbations (low frequency)
    for amp, f, phase in [(0.1, 1, 0.0), (0.2, 2, 0.5), (0.15, 1, 0.3),
                          (0.25, 3, 0.0), (0.1, 4, 1.0), (0.2, 1, 0.7)]:
        i = np.arange(N)
        arr = 0.5 + amp * np.sin(2 * np.pi * f * i / N + phase)
        arr = _fix_sum(arr)
        candidates.append(arr)

    # Block candidates
    for frac in (0.3, 0.4, 0.5, 0.6, 0.7):
        arr = np.zeros(N)
        n_ones = int(N * frac)
        arr[:n_ones] = 1.0
        arr = _fix_sum(arr)
        candidates.append(arr)

    # Random candidates
    for _ in range(50):
        if not deadline_ok():
            break
        arr = rng.uniform(0.0, 1.0, size=N)
        arr = _fix_sum(arr)
        candidates.append(arr)

    for arr in candidates:
        if not deadline_ok():
            break
        sc = _score(arr)
        if sc < best_score - 1e-10:
            best = arr.copy()
            best_score = sc

    # Final deadline guard
    if time.time() - _T0 > _DEADLINE - 2.0:
        return best
    return best
# EVOLVE-BLOCK-END

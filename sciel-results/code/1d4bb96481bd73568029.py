"""Program for the Erdos minimum-overlap task: search over step-height
candidates and keep the one minimizing the maximum overlap with shifted 1-h.
Combines FFT-based scoring with gradient descent refinement."""
import time
import numpy as np

N = 100
_DEADLINE = 140.0  # safety margin under the 150 s kill
_t0 = time.time()


def _overlap_profile(h):
    """Return overlap(k) = sum_i h[i] * (1 - h[(i-k) mod N]) for all cyclic lags k.
    Uses FFT-based circular cross-correlation for O(N log N) computation."""
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    OMH = np.fft.fft(one_minus_h)
    r = np.fft.ifft(H * np.conj(OMH)).real
    return r


def _score(h):
    """The quantity the evaluator minimizes: max_k overlap(h, shifted (1-h))."""
    h = np.asarray(h, dtype=float)
    return float(np.max(_overlap_profile(h)))


def _fix_sum(h):
    """Adjust h so its sum is exactly N/2, keeping values in [0,1] if possible."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    diff = N / 2.0 - h.sum()
    if abs(diff) < 1e-12:
        return h
    step = diff / N
    h2 = h + step
    if h2.min() >= -1e-12 and h2.max() <= 1.0 + 1e-12:
        h2 = np.clip(h2, 0.0, 1.0)
        residual = N / 2.0 - h2.sum()
        if abs(residual) < 1e-9:
            h2 += residual / N
        return h2
    h2 = np.clip(h + step, 0.0, 1.0)
    return h2


def construct():
    global _t0
    best = np.full(N, 0.5)
    best_score = _score(best)  # should be exactly 25.0

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    # Phase 1: Generate diverse candidates
    rng = np.random.default_rng(12345)
    candidates = []

    # (a) constant family
    candidates.append(np.full(N, 0.5))

    # (b) two-level / block candidates
    for frac in (0.3, 0.4, 0.5, 0.6, 0.7):
        arr = np.zeros(N)
        n_ones = int(N * frac)
        arr[:n_ones] = 1.0
        arr = _fix_sum(arr)
        candidates.append(arr)

    # (c) sinusoidal candidates
    for amp, f, phase in [(0.1, 1, 0.0), (0.2, 2, 0.5), (0.15, 1, 0.3),
                          (0.25, 3, 0.0), (0.1, 4, 1.0), (0.2, 1, 0.7),
                          (0.3, 2, 0.0), (0.35, 3, 0.5)]:
        i = np.arange(N)
        arr = 0.5 + amp * np.sin(2 * np.pi * f * i / N + phase)
        arr = _fix_sum(arr)
        candidates.append(arr)

    # (d) random candidates
    for _ in range(100):
        if not deadline_ok():
            break
        arr = rng.uniform(0.0, 1.0, size=N)
        arr = _fix_sum(arr)
        candidates.append(arr)

    # Evaluate all candidates
    for arr in candidates:
        if not deadline_ok():
            break
        arr = _fix_sum(arr)
        sc = _score(arr)
        if sc < best_score - 1e-12:
            best = arr.copy()
            best_score = sc

    # Phase 2: Gradient descent refinement from best candidate
    if deadline_ok():
        h = best.copy()
        eta = 0.01
        no_improve = 0
        max_no_improve = 50

        while deadline_ok() and no_improve < max_no_improve:
            C = _overlap_profile(h)
            k_max = int(np.argmax(C))

            # Compute gradient of C[k_max] w.r.t. h
            g = np.zeros(N)
            if k_max == 0:
                g = 1.0 - 2.0 * h
            else:
                # d/dh_i C_k = (1 - h_{(i+k) mod N}) for the shifted part
                # For cyclic: C_k = sum_i h[i] * (1 - h[(i-k) mod N])
                # dC_k/dh_i = (1 - h[(i-k) mod N]) - h[(i+k) mod N] * 0 ... 
                # Actually: C_k = sum_i h[i]*(1-h[(i-k)%N])
                # dC_k/dh_i = (1-h[(i-k)%N]) - h[i]*0 (from the second term when j=i-k)
                # Wait, let's be careful. C_k = sum_i h[i]*(1-h[(i-k)%N])
                # = sum_i h[i] - sum_i h[i]*h[(i-k)%N]
                # dC_k/dh_i = 1 - [h[(i-k)%N] + h[(i+k)%N]] (from product rule on sum_i h[i]h[(i-k)%N])
                # Actually sum_i h[i]h[(i-k)%N], derivative w.r.t h_i:
                # term i: h[(i-k)%N]
                # term j where (j-k)%N = i, i.e. j = (i+k)%N: h[i]
                # So d/dh_i sum_j h[j]h[(j-k)%N] = h[(i-k)%N] + h[(i+k)%N]
                # Therefore dC_k/dh_i = 1 - h[(i-k)%N] - h[(i+k)%N]
                shifted_minus = np.roll(h, k_max)  # h[(i-k_max)%N]
                shifted_plus = np.roll(h, -k_max)  # h[(i+k_max)%N]
                g = 1.0 - shifted_minus - shifted_plus

            # Update
            h_new = h - eta * g
            h_new = _fix_sum(h_new)

            sc_new = _score(h_new)
            if sc_new < best_score - 1e-10:
                best_score = sc_new
                best = h_new.copy()
                h = h_new.copy()
                no_improve = 0
            else:
                no_improve += 1
                # Reduce learning rate and add small perturbation
                eta *= 0.9
                h = best + rng.normal(0, 0.005, size=N)
                h = _fix_sum(h)

    # Final deadline guard
    if time.time() - _t0 > _DEADLINE - 2.0:
        return best
    return best

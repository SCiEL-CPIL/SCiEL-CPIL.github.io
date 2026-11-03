"""Search for a low-autocorrelation signed step function (400 equal intervals).

The C3 autoconvolution upper bound is governed by the peak aperiodic
autocorrelation sidelobe of the step heights. We minimize it using
analytic single-bit flip updates to the autocorrelation array, which is
significantly faster than FFT-based recalculation, allowing for more search
iterations within the time budget.
"""
import numpy as np
import time

N = 400


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to {-1, +1}."""
    tm = np.zeros(n, dtype=np.int8)
    tm[0] = 0
    for i in range(1, n):
        tm[i] = tm[i // 2] if (i % 2 == 0) else 1 - tm[i // 2]
    return 1.0 - 2.0 * tm.astype(np.float64)


def _calc_ac(h):
    """Calculate aperiodic autocorrelation using FFT."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.fft(h, fft_size)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(ac):
    return float(np.max(np.abs(ac[1:])))


def _flip_delta(h, i):
    """
    Compute the delta to add to autocorrelation ac[1:] after flipping h[i].
    Does NOT modify h or ac.
    Delta R[k] = -2 * h[i] * (h[i+k] + h[i-k]) for valid k.
    """
    shift_right = np.zeros(N, dtype=np.float64)
    shift_left = np.zeros(N, dtype=np.float64)

    if i + 1 < N:
        shift_right[1 : N - i] = h[i + 1 : N]
    if i > 0:
        shift_left[1 : i + 1] = h[i - 1 : -1 : -1]

    return -2.0 * h[i] * (shift_right + shift_left)


def _apply_flip(h, i, ac):
    """Flip h[i] and update ac in-place. Returns the new peak sidelobe."""
    delta = _flip_delta(h, i)
    ac[1:] += delta
    h[i] *= -1.0
    return _peak_sidelobe(ac)


def _revert_flip(h, i, ac):
    """Revert the flip at h[i] and update ac in-place. Returns the new peak sidelobe."""
    # h[i] is currently the flipped value. We need to revert it.
    # The delta that was applied when flipping was: -2 * h_old * S
    # Now h[i] = h_new = -h_old, so h_old = -h[i]
    # We need to subtract the original delta: ac -= (-2 * h_old * S) = ac += 2 * h_old * S = ac += 2 * (-h[i]) * S = ac -= 2 * h[i] * S
    # But S is computed from the current h (after flip). The neighbors haven't changed.
    # Actually, let's just compute the delta for flipping h[i] again (which would revert it) and apply it.
    # Flipping h[i] again: delta = -2 * h[i]_current * S_current
    # Since h[i]_current = -h_old, and S is the same (neighbors unchanged),
    # delta = -2 * (-h_old) * S = 2 * h_old * S
    # This is exactly -original_delta. So applying this delta reverts ac.
    delta = _flip_delta(h, i)
    ac[1:] += delta
    h[i] *= -1.0
    return _peak_sidelobe(ac)


def _anneal(seed, time_budget, start_time=None):
    if start_time is None:
        start_time = time.time()
    rng = np.random.default_rng(seed)

    h = _thue_morse(N)
    ac = _calc_ac(h)
    cur = _peak_sidelobe(ac)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-6

    while time.time() - start_time < time_budget:
        i = int(rng.integers(0, N))

        # Try the flip
        new = _apply_flip(h, i, ac)

        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Revert
            _revert_flip(h, i, ac)

        T = max(T_min, T * 0.99995)

    return best_h, best


def _greedy_polish(h, time_budget, start_time=None):
    if start_time is None:
        start_time = time.time()
    h = h.copy()
    ac = _calc_ac(h)
    cur = _peak_sidelobe(ac)
    best = cur
    best_h = h.copy()

    while time.time() - start_time < time_budget:
        best_i = -1
        best_c = cur

        for i in range(N):
            # Try flip
            new = _apply_flip(h, i, ac)

            if new < best_c:
                best_c = new
                best_i = i

            # Revert
            _revert_flip(h, i, ac)

        if best_i < 0:
            break

        # Commit the best flip
        _apply_flip(h, best_i, ac)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()

    return best_h, best


def construct():
    try:
        hard_deadline = 135.0
        start = time.time()

        # Initialize with Thue-Morse
        global_best_h = _thue_morse(N)
        global_best = _peak_sidelobe(_calc_ac(global_best_h))

        # Long initial anneal
        elapsed = time.time() - start
        budget = min(35.0, hard_deadline - elapsed - 5.0)
        if budget > 1.0:
            h, c = _anneal(12345, budget, start)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Multiple shorter restarts
        restart_id = 1
        while time.time() - start < hard_deadline - 5.0:
            elapsed = time.time() - start
            sub = min(10.0, hard_deadline - elapsed - 5.0)
            if sub < 1.0:
                break
            h, c = _anneal(100000 + restart_id, sub, start)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Final greedy single-flip polish
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 1.0
        if remaining > 0.5:
            h, c = _greedy_polish(global_best_h, remaining, start)
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)

"""Search for a low-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound.

Design: simulated annealing with variable-length block flips for exploration,
interleaved with greedy single-flip and adjacent-pair-flip polish for exploitation.
Each evaluation is O(N log N) via FFT, enabling 1000+ coordinate flips within budget.
Polish runs after each anneal restart to lock in gains before exploring further.
"""
import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _anneal(seed, time_budget):
    """Simulated annealing with variable-length block flips.

    Block length shrinks over time: longer blocks early for exploration,
    shorter blocks late for fine-tuning.
    """
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T_start = 4.0
    T_end = 1e-6
    start = time.time()
    while time.time() - start < time_budget:
        # Time-based cooling: temperature interpolates from T_start to T_end
        # over the full budget, independent of how fast the machine runs.
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        T = T_start * (T_end / T_start) ** progress
        # Variable block length: longer early, shorter late
        max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy polish with single-flip and adjacent-pair-flip sweeps.

    Alternates between flipping one bit and flipping two adjacent bits.
    Pair flips help escape flat plateaus that single flips cannot cross.
    """
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        improved = False

        # --- single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i >= 0:
            h[best_i] *= -1.0
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c2:
                best_c2 = c
                best_pair = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        # --- second-neighbor pair-flip sweep (flips i and i+2) ---
        best_pair2 = -1
        best_c3 = cur
        for i in range(N - 2):
            h[i] *= -1.0
            h[i + 2] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c3:
                best_c3 = c
                best_pair2 = i
            h[i] *= -1.0
            h[i + 2] *= -1.0
        if best_pair2 >= 0:
            h[best_pair2] *= -1.0
            h[best_pair2 + 2] *= -1.0
            cur = best_c3
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        if not improved:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tighter single-flip polish for the final pass."""
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i < 0:
            break
        h[best_i] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def construct():
    try:
        hard_deadline = 135.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: long initial anneal from a random +-1 start.
        budget = min(30.0, _remaining() - 10.0)
        if budget > 2.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Polish right after to lock in gains
            sub = min(6.0, _remaining() - 10.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity, with polish after each.
        restart_id = 1
        while _remaining() > 14.0:
            sub = min(12.0, _remaining() - 14.0)
            if sub < 1.5:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Quick polish after each restart
            sub2 = min(3.0, _remaining() - 14.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision single-flip polish.
        remaining = _remaining() - 2.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: one more combined polish pass if time remains.
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)

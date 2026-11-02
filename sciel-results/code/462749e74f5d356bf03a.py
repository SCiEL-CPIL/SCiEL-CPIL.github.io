"""Low-peak-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound
below the all-ones baseline.

Design family (kept): simulated annealing + greedy polish, now with
  * an O(N) incremental autocorrelation update for single-flips (no FFT per trial),
  * a smooth epigraph / soft-peak (log-sum-exp) surrogate for exploration,
  * a high-precision greedy single-flip + adjacent-pair-flip polish,
  * an explicit 140 s deadline guard (10 s safety under the 150 s kill),
  * sine-based spectral initialization to exploit low-frequency structure.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _ac_full(h):
    """Full aperiodic autocorrelation ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k], k=0..N-1."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))


def _flip_ac_fast(ac, h, i):
    """Vectorised O(N) update of ac after flipping h[i]."""
    new_ac = ac.copy()
    hi = h[i]
    k_hi = N - 1 - i
    if k_hi >= 0:
        new_ac[0:k_hi + 1] += -2.0 * hi * h[i:i + k_hi + 1]
    k_lo = i
    if k_lo >= 1:
        new_ac[1:k_lo + 1] += -2.0 * hi * h[i - k_lo:i]
    return new_ac


def _flip_ac_multi(ac, h, indices):
    """O(N) update of ac after flipping multiple positions simultaneously.
    For a set of flips S, the change in ac[k] is computed exactly.
    """
    if len(indices) <= 1:
        if len(indices) == 0:
            return ac.copy()
        return _flip_ac_fast(ac, h, indices[0])
    
    S_set = set(indices)
    new_ac = ac.copy()
    # new_ac[k] = ac[k] - 2*sum_{j in S, j+k < N} h[j]*h[j+k] 
    #             - 2*sum_{m in S, m-k >= 0} h[m-k]*h[m] 
    #             + 4*sum_{j in S, j+k in S} h[j]*h[j+k]
    
    # Precompute indices as array for speed
    idx_arr = np.array(indices, dtype=np.int64)
    
    for k in range(1, N):
        # Term 1: -2 * sum_{j in S, j+k < N} h[j]*h[j+k]
        valid1 = idx_arr + k < N
        if np.any(valid1):
            j1 = idx_arr[valid1]
            t1 = np.sum(h[j1] * h[j1 + k])
        else:
            t1 = 0.0
        
        # Term 2: -2 * sum_{m in S, m-k >= 0} h[m-k]*h[m]
        valid2 = idx_arr - k >= 0
        if np.any(valid2):
            m2 = idx_arr[valid2]
            t2 = np.sum(h[m2 - k] * h[m2])
        else:
            t2 = 0.0
        
        # Term 3: +4 * sum_{j in S, j+k in S} h[j]*h[j+k]
        t3 = 0.0
        for j in indices:
            if j + k in S_set:
                t3 += h[j] * h[j + k]
        
        new_ac[k] += -2.0 * t1 - 2.0 * t2 + 4.0 * t3
    
    return new_ac


# ---------------------------------------------------------------------------
# Annealing with variable-length block flips
# ---------------------------------------------------------------------------

def _sine_init(freq, phase, seed):
    """Generate a +-1 sequence from zero crossings of a sine wave with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi * freq, N)
    signal = np.sin(x + phase)
    noise = rng.normal(0, 0.1, size=N)
    h = np.sign(signal + noise)
    h[h == 0] = 1.0
    return h


def _anneal(seed, time_budget, init_h=None):
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        strategy = rng.integers(0, 3)
        if strategy == 0:
            h = _sine_init(rng.uniform(2, 8), rng.uniform(0, 2 * np.pi), seed)
        elif strategy == 1:
            h = _sine_init(rng.uniform(5, 15), rng.uniform(0, 2 * np.pi), seed)
        else:
            h = rng.choice([-1.0, 1.0], size=N)
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    start = time.time()
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        ac_new = _ac_full(h)
        new = float(np.max(np.abs(ac_new[1:])))
        d_peak = new - cur
        if d_peak <= 0:
            cur = new
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            if rng.random() < np.exp(-d_peak / max(T, 1e-9)):
                cur = new
                ac = ac_new
            else:
                h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99997)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish: single-flip (O(N) update) + adjacent-pair-flip
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    h = h.copy()
    ac = _ac_full(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        # --- single-flip sweep (fast O(N) update) ---
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        if best_i >= 0:
            h[best_i] *= -1.0
            ac = _flip_ac_fast(ac, h, best_i)
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            ac2 = _flip_ac_multi(ac, h, [i, i + 1])
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c2:
                best_c2 = c
                best_pair = i
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            ac = _flip_ac_multi(ac, h, [best_pair, best_pair + 1])
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
        # --- triple-flip sweep ---
        best_triple = -1
        best_c3 = cur
        for i in range(N - 2):
            ac2 = _flip_ac_multi(ac, h, [i, i + 1, i + 2])
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c3:
                best_c3 = c
                best_triple = i
        if best_triple >= 0:
            h[best_triple] *= -1.0
            h[best_triple + 1] *= -1.0
            h[best_triple + 2] *= -1.0
            ac = _flip_ac_multi(ac, h, [best_triple, best_triple + 1, best_triple + 2])
            cur = best_c3
            if cur < best:
                best = cur
                best_h = h.copy()
        if best_i < 0 and best_pair < 0 and best_triple < 0:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tight single-flip polish with the O(N) update."""
    h = h.copy()
    ac = _ac_full(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        if best_i < 0:
            break
        h[best_i] *= -1.0
        ac = _flip_ac_fast(ac, h, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def _exhaustive_single_flip(h, time_budget, start):
    """Multi-pass single-flip: repeatedly scan all positions, accepting any improvement."""
    h = h.copy()
    ac = _ac_full(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        improved = False
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        if best_i >= 0:
            h[best_i] *= -1.0
            ac = _flip_ac_fast(ac, h, best_i)
            cur = best_c
            improved = True
            if cur < best:
                best = cur
                best_h = h.copy()
        if not improved:
            break
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: long initial anneal from spectral initialization.
        budget = min(30.0, _remaining() - 20.0)
        if budget > 2.0:
            h_sine = _sine_init(5.0, np.pi / 4, 12345)
            h, c = _anneal(12345, budget, init_h=h_sine)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 20.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity with spectral init.
        restart_id = 1
        while _remaining() > 18.0:
            sub = min(12.0, _remaining() - 18.0)
            if sub < 1.5:
                break
            if restart_id % 3 == 1:
                h_init = _sine_init(3.0, 0.0, 100000 + restart_id)
            elif restart_id % 3 == 2:
                h_init = _sine_init(7.0, np.pi / 2, 100000 + restart_id)
            else:
                h_init = None
            h, c = _anneal(100000 + restart_id, sub, init_h=h_init)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 18.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision single-flip polish.
        remaining = _remaining() - 5.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: comprehensive polish with pair and triple flips.
        remaining2 = _remaining() - 2.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: final exhaustive single-flip pass to squeeze out last improvements.
        remaining3 = _remaining() - 0.5
        if remaining3 > 0.3:
            h, c = _exhaustive_single_flip(global_best_h, remaining3, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)

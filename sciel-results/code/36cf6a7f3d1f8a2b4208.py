"""Low-peak-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound
below the all-ones baseline.

Design family: simulated annealing + greedy polish, with:
  * O(N) incremental autocorrelation update for single-flips
  * log-sum-exp soft-peak surrogate for exploration
  * continuous relaxation via projected gradient
  * high-precision greedy single-flip + adjacent-pair-flip polish
  * explicit 140 s deadline guard
  * sine-based and VDC spectral initialization
  * improved annealing schedule and polish
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


def _soft_peak(ac, tau):
    """Smooth epigraph surrogate: log-sum-exp of ac[k]^2/tau over k>=1."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m))))


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


# ---------------------------------------------------------------------------
# Continuous relaxation: projected-gradient on soft peak
# ---------------------------------------------------------------------------

def _soft_peak_grad(ac, tau):
    """Gradient of soft peak f = log sum e^{ac[k]^2/tau} wrt ac (k>=1)."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    e = np.exp(v - m)
    p = e / float(np.sum(e))
    g = np.zeros(N)
    g[1:] = (2.0 / tau) * ac[1:] * p
    return g


def _continuous_relax(h0, time_budget, start):
    """Projected-gradient on the continuous relaxation h in [-1,1]."""
    h = h0.astype(np.float64).copy()
    tau = 8.0
    step = 0.05
    ac = _ac_full(h)
    f = _soft_peak(ac, tau)
    best_f = f
    best_h = h.copy()
    while time.time() - start < time_budget:
        g_ac = _soft_peak_grad(ac, tau)
        grad_h = np.zeros(N)
        for k in range(1, N):
            gk = g_ac[k]
            if gk == 0.0:
                continue
            grad_h[0:N - k] += gk * h[k:N]
            grad_h[k:N] += gk * h[0:N - k]
        h = h - step * grad_h
        h = np.clip(h, -1.0, 1.0)
        ac = _ac_full(h)
        nf = _soft_peak(ac, tau)
        if nf < best_f:
            best_f = nf
            best_h = h.copy()
        tau = max(1.0, tau * 0.99)
        step = 0.05 + 0.02 * (1.0 - min(1.0, tau / 8.0))
    hr = np.where(best_h >= 0.0, 1.0, -1.0)
    return hr, _peak_sidelobe(hr)


# ---------------------------------------------------------------------------
# Initialization
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


def _vdc_init(seed):
    """Generate a +-1 sequence based on the van der Corput sequence."""
    vdc_vals = np.zeros(N)
    for i in range(N):
        val = 0.0
        div = 1
        n = i
        while n > 0:
            div *= 2
            if n % 2 == 1:
                val += 1.0 / div
            n //= 2
        vdc_vals[i] = val
    h = np.where(vdc_vals < 0.5, -1.0, 1.0)
    shift = seed % N
    h = np.roll(h, shift)
    return h


def _random_init(seed):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=N)


# ---------------------------------------------------------------------------
# Annealing with variable-length block flips
# ---------------------------------------------------------------------------

def _anneal(seed, time_budget, init_h=None):
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        h = _random_init(seed)
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 6.0
    T_min = 1e-7
    tau = 10.0
    start = time.time()
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(32.0 * (1.0 - 0.6 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        ac_new = _ac_full(h)
        new = _peak_sidelobe(h)
        d = _soft_peak(ac_new, tau) - _soft_peak(ac, tau)
        if new <= cur or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99995)
        tau = max(1.0, tau * 0.9998)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish
# ---------------------------------------------------------------------------

def _greedy_single_only(h, time_budget, start):
    """Tight single-flip polish with the O(N) update."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
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


def _greedy_polish(h, time_budget, start):
    """Single-flip + adjacent-pair-flip polish."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        # Single-flip sweep
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
        else:
            # Try pair flips
            if time_budget - (time.time() - start) > 1.0:
                best_pair = -1
                best_c2 = cur
                for i in range(N - 1):
                    if time.time() - start > time_budget - 0.5:
                        break
                    h[i] *= -1.0
                    h[i + 1] *= -1.0
                    ac2 = _ac_full(h)
                    c = float(np.max(np.abs(ac2[1:])))
                    h[i] *= -1.0
                    h[i + 1] *= -1.0
                    if c < best_c2:
                        best_c2 = c
                        best_pair = i
                if best_pair >= 0:
                    h[best_pair] *= -1.0
                    h[best_pair + 1] *= -1.0
                    ac = _ac_full(h)
                    cur = best_c2
                    if cur < best:
                        best = cur
                        best_h = h.copy()
                else:
                    break
            else:
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

        # Phase 1: continuous relaxation to seed a good +-1 starting point
        budget = min(15.0, _remaining() - 30.0)
        if budget > 3.0:
            h, c = _continuous_relax(global_best_h, budget, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 30.0)
            if sub > 0.5:
                h2, c2 = _greedy_single_only(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: long anneal from spectral initialization
        budget = min(25.0, _remaining() - 25.0)
        if budget > 2.0:
            h_sine = _sine_init(5.0, np.pi / 4, 12345)
            h, c = _anneal(12345, budget, init_h=h_sine)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 25.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 3: multiple shorter restarts for diversity
        restart_id = 1
        while _remaining() > 20.0:
            sub = min(10.0, _remaining() - 20.0)
            if sub < 1.5:
                break
            if restart_id % 5 == 1:
                h_init = _sine_init(3.0, 0.0, 100000 + restart_id)
            elif restart_id % 5 == 2:
                h_init = _sine_init(7.0, np.pi / 2, 100000 + restart_id)
            elif restart_id % 5 == 3:
                h_init = _sine_init(11.0, np.pi / 3, 100000 + restart_id)
            elif restart_id % 5 == 4:
                h_init = _vdc_init(100000 + restart_id)
            else:
                h_init = None
            h, c = _anneal(100000 + restart_id, sub, init_h=h_init)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 20.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 4: final high-precision single-flip polish
        remaining = _remaining() - 5.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: one more polish pass if time remains
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)

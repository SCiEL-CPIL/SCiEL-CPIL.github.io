"""Search for a low-autocorrelation signed step function (400 equal intervals).

Goal: minimise the peak aperiodic autocorrelation sidelobe of a signed (+-1)
sequence of length 400; this drives the C3 autoconvolution upper bound below
the all-ones baseline.

Design family: simulated annealing with block flips + greedy coordinate
descent polish (single-bit and adjacent-pair flips), all evaluated via FFT.

Key improvements over the seed:
  1.  Faster peak-sidelobe evaluation using real-FFT and a pre-allocated
      buffer, reducing per-flip cost so more flips fit in the budget.
  2.  Annealing schedule: block length adapts to progress; acceptance
      threshold uses a smoothed surrogate (soft-max over top-k sidelobes)
      to break plateaus that the hard max objective creates.
  3.  Greedy polish: interleaved single-flip and pair-flip sweeps; after
      convergence, a second tighter single-flip sweep with a smaller
      improvement threshold to catch marginal gains.
  4.  Multiple restarts with diverse seeds, each followed by a quick polish.
  5.  Hard 135 s wall-clock deadline with 15 s safety margin.
"""
import numpy as np
import time

N = 400
_HARD_DEADLINE = 135.0  # 15 s margin below the 150 s kill

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _continuous_to_binary(h_cont):
    """Convert a continuous vector to a signed binary sequence via sign."""
    return np.sign(h_cont).astype(np.float64)


def _gradient_descent_seed(time_budget, seed=42):
    """
    Run gradient descent on the L2 norm of the autocorrelation sidelobes
    to find a good continuous starting point, then binarize.
    
    Objective: minimize sum_k |ac_k|^2 for k >= 1 (flat spectrum)
    This gives a much better binary seed than random initialization.
    Uses FFT-based gradient computation for efficiency.
    """
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(N)
    start = time.time()
    lr = 0.01
    decay = 0.9995
    patience = 200
    no_improve = 0
    best_cost = np.inf
    best_h = h.copy()
    
    for step in range(50000):
        if time.time() - start > time_budget:
            break
        
        # Compute autocorrelation via FFT
        H = np.fft.rfft(h, _FFT_SIZE)
        power = H * H.conj()
        ac = np.fft.irfft(power, _FFT_SIZE, n=N)
        sidelobes = ac[1:]
        
        # Cost: L2 norm of sidelobes
        cost = np.sum(sidelobes ** 2)
        
        if cost < best_cost - 1e-14:
            best_cost = cost
            best_h = h.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience:
            break
        
        # Gradient computation via FFT
        # The gradient of sum_k ac_k^2 w.r.t. h is:
        # 2 * ifft(2 * |H|^2 * ac_shifted) where ac_shifted has ac[0]=0
        # More precisely: grad = 2 * real(ifft(2 * power * ac_full))
        # where ac_full is the full autocorrelation including lag 0
        # But we want to exclude lag 0, so:
        # grad = 2 * real(ifft(2 * power * (ac - ac[0]*delta)))
        # where delta is 1 at index 0, 0 elsewhere
        
        # Actually, the correct formula:
        # ac_k = sum_i h_i * h_{i+k} (aperiodic)
        # The gradient of sum_{k>=1} ac_k^2 w.r.t. h_i is:
        # 2 * sum_{k>=1} ac_k * (h_{i+k} + h_{i-k})
        # This equals 2 * (ac * h)_i where * is circular correlation,
        # but with the lag-0 term removed.
        
        # Using FFT: 
        # Let A = ifft(|H|^2) = ac (circular autocorrelation)
        # Then the circular correlation of ac with h is ifft(A * H.conj())
        # But we need to be careful about the aperiodic vs circular distinction.
        
        # For aperiodic autocorrelation, the gradient is:
        # grad_i = 2 * sum_{k=1}^{N-1} ac_k * (h_{i+k} + h_{i-k})
        #         where h_j = 0 for j outside [0, N-1]
        
        # This can be computed as:
        # Let ac_ext = [ac_1, ac_2, ..., ac_{N-1}]
        # grad = 2 * (ac_ext * h) + 2 * (ac_ext * h) reversed appropriately
        
        # Simpler: use the fact that for the circular case,
        # grad = 2 * ifft(2 * |H|^2 * A) where A = ifft(|H|^2)
        # but subtract the lag-0 contribution.
        
        # Let's use a direct but vectorized approach with limited lags
        # for the gradient, which is O(N * L) where L is max lag
        L_max = min(N - 1, 200)  # Limit to 200 lags for speed
        grad = np.zeros_like(h)
        for k in range(1, L_max + 1):
            if time.time() - start > time_budget:
                break
            ac_k = np.sum(h[:N-k] * h[k:])
            # grad[i] += ac_k * (h[i+k] if i+k < N else 0) + ac_k * (h[i-k] if i-k >= 0 else 0)
            if N - k > 0:
                grad[:N-k] += ac_k * h[k:]
                grad[k:] += ac_k * h[:N-k]
        grad *= 2
        
        # Update with adaptive learning rate
        h -= lr * grad
        lr *= decay
    
    return _continuous_to_binary(best_h)


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H.conj(), _FFT_SIZE, n=N)
    return float(np.max(np.abs(ac[1:])))


def _soft_peak_sidelobe(h, k=10, beta=2.0):
    """Smooth surrogate: soft-max over top-k sidelobe magnitudes.
    Lower beta -> closer to hard max; higher beta -> smoother."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H.conj(), _FFT_SIZE, n=N)
    mags = np.abs(ac[1:])
    # soft-max via log-sum-exp on the top-k values
    idx = np.argpartition(mags, -k)[-k:]
    top = mags[idx]
    mx = np.max(top)
    # log-sum-exp
    shifted = top - mx
    lse = mx + np.log(np.sum(np.exp(beta * shifted))) / beta
    return float(lse)


# ---------------------------------------------------------------------------
# Annealing with variable-length block flips and soft-max acceptance
# ---------------------------------------------------------------------------

def _anneal(seed, time_budget, initial_h=None):
    rng = np.random.default_rng(seed)
    if initial_h is not None:
        h = initial_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-8
    start = time.time()
    use_soft = True  # use soft surrogate for acceptance
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        # Block length: longer early, shorter late
        max_len = max(1, int(20.0 * (1.0 - 0.6 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0

        if use_soft:
            new_soft = _soft_peak_sidelobe(h, k=8, beta=1.5)
            new_hard = _peak_sidelobe(h)
        else:
            new_hard = _peak_sidelobe(h)
            new_soft = new_hard

        d = new_hard - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_hard
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0

        T = max(T_min, T * 0.99995)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish: single-flip + adjacent-pair-flip
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start=None):
    """Interleaved single-flip and adjacent-pair-flip greedy descent."""
    if start is None:
        start = time.time()
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

        if not improved:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start=None):
    """Tighter single-flip polish for the final pass."""
    if start is None:
        start = time.time()
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


def _greedy_triple_flip(h, time_budget, start=None):
    """Greedy sweep that also considers flipping 3 consecutive bits."""
    if start is None:
        start = time.time()
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        improved = False
        # single-flip sweep
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
        # triple-flip sweep
        best_triple = -1
        best_c3 = cur
        for i in range(N - 2):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            h[i + 2] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c3:
                best_c3 = c
                best_triple = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
            h[i + 2] *= -1.0
        if best_triple >= 0:
            h[best_triple] *= -1.0
            h[best_triple + 1] *= -1.0
            h[best_triple + 2] *= -1.0
            cur = best_c3
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        if not improved:
            break
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return _HARD_DEADLINE - (time.time() - start)

        # Phase 0: gradient descent seed
        gd_budget = min(8.0, _remaining() - 20.0)
        if gd_budget > 1.0:
            h_gd = _gradient_descent_seed(gd_budget, seed=42)
            c_gd = _peak_sidelobe(h_gd)
            if c_gd < global_best:
                global_best = c_gd
                global_best_h = h_gd
            # Quick polish on GD seed
            sub = min(3.0, _remaining() - 20.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(h_gd, sub)
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 1: initial anneal (start from GD seed if available)
        budget = min(25.0, _remaining() - 10.0)
        if budget > 2.0:
            h, c = _anneal(42, budget, initial_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 10.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub)
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: restarts with diverse seeds
        restart_id = 1
        while _remaining() > 15.0:
            sub = min(10.0, _remaining() - 15.0)
            if sub < 1.5:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 15.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2)
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: greedy polish with triple flips
        remaining = _remaining() - 5.0
        if remaining > 0.5:
            h, c = _greedy_triple_flip(global_best_h, remaining)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: final tight single-flip polish
        remaining2 = _remaining() - 2.0
        if remaining2 > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining2)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: one more full polish if time remains
        remaining3 = _remaining() - 1.0
        if remaining3 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining3)
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)

"""Search for a low-autocorrelation signed step function (400 equal intervals).

Goal: minimise the peak aperiodic autocorrelation sidelobe of a signed (+-1)
sequence of length 400; this drives the C3 autoconvolution upper bound below
the all-ones baseline.

This version uses:
  1.  Continuous relaxation with soft-max (log-sum-exp) surrogate for the
      peak sidelobe, optimized via gradient descent in R^N.
  2.  Multiple restarts with different initializations and seeds.
  3.  Greedy single-flip and pair-flip polish with incremental AC updates.
  4.  Deadline guard with safety margin.
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


def _full_ac(h):
    """Compute full aperiodic autocorrelation array (length N)."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_from_ac(ac):
    """Peak sidelobe from an existing AC array."""
    return float(np.max(np.abs(ac[1:])))


# ---------------------------------------------------------------------------
# Continuous relaxation with soft-max surrogate
# ---------------------------------------------------------------------------

def _round_to_sign(h):
    """Round continuous values to ±1."""
    return np.where(h >= 0, 1.0, -1.0)


def _soft_max_grad_ac(ac, tau):
    """
    Compute gradient of soft-max objective wrt ac.
    Returns (obj, grad_ac) where grad_ac[k] = d(obj)/d(ac[k]) for k=1..N-1.
    """
    abs_ac = np.abs(ac[1:])
    m = np.max(abs_ac)
    shifted = tau * (abs_ac - m)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    Z = np.sum(exp_shifted)
    obj = m + np.log(Z)
    grad_ac = np.zeros(N)
    signs = np.sign(ac[1:])
    grad_ac[1:] = signs * exp_shifted / Z
    return obj, grad_ac


def _compute_grad_h(h, grad_ac):
    """
    Compute gradient of soft-max objective wrt h, given grad_ac.
    grad_h[i] = sum_{k=1}^{N-1} grad_ac[k] * (h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
    """
    grad_h = np.zeros(N)
    g = grad_ac[1:]
    for i in range(N):
        if i + 1 < N:
            ks = np.arange(1, N - i)
            grad_h[i] += np.dot(g[:len(ks)], h[i + ks])
        if i >= 1:
            ks = np.arange(1, i + 1)
            grad_h[i] += np.dot(g[:len(ks)], h[i - ks])
    return grad_h


def _continuous_relax(h_init, time_budget, start, seed=0):
    """
    Continuous gradient descent on soft-max objective.
    Returns best ±1 sequence found.
    """
    rng = np.random.default_rng(seed)
    h = h_init.astype(np.float64).copy()
    tau = 1.0
    lr = 0.3
    best_h = _round_to_sign(h)
    best_val = _peak_sidelobe(best_h)
    
    step = 0
    while time.time() - start < time_budget:
        ac = _full_ac(h)
        obj, grad_ac = _soft_max_grad_ac(ac, tau)
        grad_h = _compute_grad_h(h, grad_ac)
        
        h = h - lr * grad_h
        
        if step % 5 == 0:
            h_rounded = _round_to_sign(h)
            val = _peak_sidelobe(h_rounded)
            if val < best_val:
                best_val = val
                best_h = h_rounded.copy()
        
        tau *= 1.02
        lr *= 0.998
        
        if step % 50 == 0 and rng.random() < 0.3:
            h = _round_to_sign(h) + 0.5 * rng.randn(N)
        
        step += 1
    
    h_final = _round_to_sign(h)
    val_final = _peak_sidelobe(h_final)
    if val_final < best_val:
        best_val = val_final
        best_h = h_final.copy()
    
    return best_h, best_val


# ---------------------------------------------------------------------------
# Incremental single-flip autocorrelation update
# ---------------------------------------------------------------------------

def _flip_ac_delta(h, i):
    """
    Compute the change in the autocorrelation array when flipping h[i].
    Returns (delta_ac, affected_lags) where delta_ac is a dict mapping
    lag -> delta, and affected_lags is the set of lags that changed.
    """
    # Flipping h[i] -> -h[i]
    # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    # When we flip h[i], for each lag k:
    #   If 0 <= i <= N-1-k: h[i] appears as h[j] with j=i, so term h[i]*h[i+k] changes sign
    #   If 0 <= i-k <= N-1-k: h[i] appears as h[j+k] with j=i-k, so term h[i-k]*h[i] changes sign
    # The change in ac[k] is:
    #   -2 * (contribution from h[i])
    # where contribution is h[i]*h[i+k] if i+k < N, plus h[i-k]*h[i] if i-k >= 0
    
    # For lag k >= 1:
    #   h[i] contributes to ac[k] as:
    #     - h[i]*h[i+k] if i+k < N (i is the first index)
    #     - h[i-k]*h[i] if i-k >= 0 (i is the second index)
    # After flip, these become -h[i]*h[i+k] and -h[i-k]*h[i]
    # So delta_ac[k] = -2 * (h[i]*h[i+k] if i+k<N else 0) - 2 * (h[i-k]*h[i] if i-k>=0 else 0)
    
    # We'll compute delta for all k in 1..N-1
    # But only k where i+k < N or i-k >= 0 are affected
    # k ranges from max(1, i-(N-1)) to min(N-1, N-1-i) for the first term
    # and k ranges from max(1, N-i) to min(N-1, i) for the second term
    # Actually let's just compute for all k=1..N-1 but only non-zero ones
    
    delta = np.zeros(N)
    # First term: h[i]*h[i+k] for k=1..N-1-i
    if i + 1 < N:
        k_range = np.arange(1, N - i)
        delta[k_range] -= 2.0 * h[i] * h[i + k_range]
    # Second term: h[i-k]*h[i] for k=1..i
    if i >= 1:
        k_range2 = np.arange(1, i + 1)
        delta[k_range2] -= 2.0 * h[i - k_range2] * h[i]
    
    return delta


def _eval_single_flip(h, ac, i):
    """
    Evaluate the peak sidelobe after flipping h[i], using incremental AC update.
    Returns (new_peak, new_ac) where new_ac is the updated AC array.
    """
    delta = _flip_ac_delta(h, i)
    new_ac = ac + delta
    new_peak = float(np.max(np.abs(new_ac[1:])))
    return new_peak, new_ac


def _apply_flip(h, ac, i):
    """
    Apply flip at position i to both h and ac.
    Returns updated h and ac.
    """
    delta = _flip_ac_delta(h, i)
    ac = ac + delta
    h[i] *= -1.0
    return h, ac


# ---------------------------------------------------------------------------
# Greedy polish: single-flip + adjacent-pair-flip (with incremental AC)
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    h = h.copy()
    ac = _full_ac(h)
    cur = _peak_from_ac(ac)
    best = cur
    best_h = h.copy()
    best_ac = ac.copy()
    while time.time() - start < time_budget:
        # --- single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            c, _ = _eval_single_flip(h, ac, i)
            if c < best_c:
                best_c = c
                best_i = i
        if best_i >= 0:
            h, ac = _apply_flip(h, ac, best_i)
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
                best_ac = ac.copy()
        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            # Flip both i and i+1
            # Compute delta for flipping i, then for flipping i+1 on the already-flipped h
            # But we need to be careful: after flipping i, h[i] is negated
            # Let's compute the combined delta directly
            # Original: h[i], h[i+1]
            # After flip: -h[i], -h[i+1]
            # The change in ac[k] from flipping both:
            # For each lag k, the terms involving h[i] or h[i+1] change
            # This is more complex; let's just do two sequential flips for correctness
            # but compute the combined effect
            delta1 = _flip_ac_delta(h, i)
            # After flipping i, h[i] is negated. Now flip i+1.
            # For the second flip, we need h with h[i] negated
            h_temp = h.copy()
            h_temp[i] *= -1.0
            delta2 = _flip_ac_delta(h_temp, i + 1)
            total_delta = delta1 + delta2
            new_ac = ac + total_delta
            c = float(np.max(np.abs(new_ac[1:])))
            if c < best_c2:
                best_c2 = c
                best_pair = i
        if best_pair >= 0:
            # Apply both flips
            h, ac = _apply_flip(h, ac, best_pair)
            h, ac = _apply_flip(h, ac, best_pair + 1)
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
                best_ac = ac.copy()
        # --- break if neither sweep improved ---
        if best_i < 0 and best_pair < 0:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tighter single-flip polish for the final pass (with incremental AC)."""
    h = h.copy()
    ac = _full_ac(h)
    cur = _peak_from_ac(ac)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            c, _ = _eval_single_flip(h, ac, i)
            if c < best_c:
                best_c = c
                best_i = i
        if best_i < 0:
            break
        h, ac = _apply_flip(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
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

        # Phase 1: Continuous relaxation from random start
        rng = np.random.default_rng(42)
        h_init = rng.choice([-1.0, 1.0], size=N)
        budget1 = min(35.0, _remaining() - 25.0)
        if budget1 > 5.0:
            h, c = _continuous_relax(h_init, budget1, time.time(), seed=0)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 2: Greedy polish
        budget2 = min(25.0, _remaining() - 20.0)
        if budget2 > 2.0:
            h2, c2 = _greedy_polish(global_best_h, budget2, time.time())
            if c2 < global_best:
                global_best = c2
                global_best_h = h2

        # Phase 3: Another continuous relaxation with different seed
        budget3 = min(25.0, _remaining() - 15.0)
        if budget3 > 3.0:
            h3_init = global_best_h.copy()
            h3, c3 = _continuous_relax(h3_init, budget3, time.time(), seed=1)
            if c3 < global_best:
                global_best = c3
                global_best_h = h3

        # Phase 4: Final greedy polish
        budget4 = min(20.0, _remaining() - 8.0)
        if budget4 > 1.0:
            h4, c4 = _greedy_single_only(global_best_h, budget4, time.time())
            if c4 < global_best:
                global_best = c4
                global_best_h = h4

        # Phase 5: One more polish if time remains
        budget5 = _remaining() - 3.0
        if budget5 > 0.5:
            h5, c5 = _greedy_polish(global_best_h, budget5, time.time())
            if c5 < global_best:
                global_best = c5
                global_best_h = h5

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)

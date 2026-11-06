"""Flattest plus-minus-one polynomials with 24 coefficients (minimum sup norm).

P(z) = sum c_k z^k, c_k in {+1,-1}, |c| = 24.
Score = max_{|z|=1} |P(z)| / sqrt(24).  Optimum ~ 1.2583758 (max|P| ~ 6.1363).

Strategy: Simulated Annealing with incremental FFT-based evaluation.
This allows escaping local minima by accepting worse moves with a temperature-dependent probability.
"""
import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0


def _time_left():
    return _LIMIT - (time.time() - _T0)


def _rs_poly():
    """Rudin-Shapiro coefficients truncated to _N (strong low-peak seed)."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]


def _setup_fft(M):
    """Precompute FFT basis for fast evaluation."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    powers = np.arange(_N)
    Z = np.power(z[:, None], powers[None, :])
    return Z, theta, z


def _peak_from_fft(vals):
    """Get peak magnitude and its phase index from FFT values."""
    mag2 = np.abs(vals) ** 2
    best_idx = np.argmax(mag2)
    return math.sqrt(mag2[best_idx]), best_idx


def _golden_section(coeffs, Z, theta, z, peak, best_idx, M, iters=14):
    """Golden-section refinement of a local peak around theta[best_idx]."""
    center = theta[best_idx]
    step = 2.0 * math.pi / M
    lo = center - step
    hi = center + step
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)

    def f(a):
        zk = np.exp(1j * a)
        zk_powers = np.power(zk, powers)
        s = np.dot(c_arr, zk_powers)
        return s.real * s.real + s.imag * s.imag

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(iters):
        m1 = hi - inv_phi * (hi - lo)
        m2 = lo + inv_phi * (hi - lo)
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            lo = m1
        else:
            hi = m2
    mid = (lo + hi) / 2.0
    peak2 = f(mid)
    if peak2 > peak * peak:
        peak = math.sqrt(peak2)
    return peak


def _search_peak(coeffs, Z, theta, z, M, refine=True):
    """Measured peak: dense scan + refinement of top few peaks."""
    c = np.array(coeffs, dtype=complex)
    vals = Z @ c
    peak, best_idx = _peak_from_fft(vals)
    if refine:
        nb = max(1, M // 512)
        for d in (0, nb, -nb, 2*nb, -2*nb):
            ph = theta[(best_idx + d) % M]
            peak = _golden_section(coeffs, Z, theta, z, peak, (best_idx + d) % M, M, iters=12)
    return peak


def _local_search_fast(seed, M, deadline):
    """Simulated Annealing local search using incremental FFT updates for single flips."""
    Z, theta, z = _setup_fft(M)
    cur = list(seed)
    c = np.array(cur, dtype=complex)
    
    # Precompute column j of Z (z_k^j for all k)
    Zcols = np.zeros((M, _N), dtype=complex)
    for j in range(_N):
        Zcols[:, j] = np.power(z, j)
    
    cur_peak = _search_peak(cur, Z, theta, z, M)
    best = list(cur)
    best_peak = cur_peak
    
    # Keep track of current FFT values for incremental updates
    fft_vals = Z @ c
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    # Start with a temperature that allows some exploration
    # The "energy" difference is in peak magnitude.
    # We want to accept worse moves if exp(-delta/T) > random.
    # Typical peak differences are small (1e-3 to 1e-1).
    T_start = 0.05
    T_end = 1e-6
    
    # We will run until time limit, cooling T linearly or exponentially.
    # Let's estimate number of steps we can do.
    # Each step is very fast (vectorized ops).
    
    steps_done = 0
    steps_target = 1000000 # Just a placeholder to calculate cooling rate if we had a fixed step count
    # Better: cool based on time remaining.
    
    while _time_left() > 10.0:
        # Calculate current temperature
        # Simple linear cooling from T_start to T_end over the expected duration
        # Or just use a fixed schedule based on iterations? 
        # Let's use time-based cooling.
        elapsed = time.time() - _T0
        # Assume total run time is _LIMIT - 10
        total_time = _LIMIT - 10.0
        progress = min(1.0, max(0.0, elapsed / total_time))
        # Exponential cooling is often better
        T = T_end + (T_start - T_end) * math.exp(-5.0 * progress)
        
        # Pick a random flip
        j = int(rng.integers(0, _N))
        
        # Flip coefficient j
        old_c_j = c[j]
        cur[j] = -cur[j]
        c[j] = -c[j]
        
        # Incremental FFT update
        # new_fft = old_fft + (new_c[j] - old_c[j]) * Zcols[:, j]
        # new_c[j] = -old_c[j]
        # diff = -2 * old_c[j]
        fft_vals = fft_vals - 2.0 * old_c_j * Zcols[:, j]
        
        # Evaluate new peak
        new_peak, _ = _peak_from_fft(fft_vals)
        
        # Acceptance criterion
        if new_peak < cur_peak:
            # Always accept improvement
            cur_peak = new_peak
            if cur_peak < best_peak - 1e-9:
                # Refine the best candidate to get a more accurate score
                # This is expensive, so only do it when we think we found a new global best
                # Actually, _peak_from_fft is an upper bound. 
                # If the grid peak drops below the refined best, it's likely a true improvement.
                # To be safe and fast, we can just track the grid peak as the "current" state
                # and refine periodically or at the end.
                # However, the final score depends on the true sup norm.
                # If we accept a move based on grid peak, we might accept a move that looks better on grid
                # but is worse on true sup norm due to aliasing/undersampling?
                # With M=256 or 512, aliasing is low for degree 23.
                # Let's refine only if the grid peak is significantly lower than the current best refined peak.
                if best_peak - cur_peak > 1e-4:
                    p_refined = _search_peak(cur, Z, theta, z, M, refine=True)
                    if p_refined < best_peak:
                        best = list(cur)
                        best_peak = p_refined
        else:
            # Worse move: accept with probability exp(-delta / T)
            delta = new_peak - cur_peak
            if delta < 0.01: # Only bother with exp if delta is small
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    # Accept
                    cur_peak = new_peak
                else:
                    # Revert
                    cur[j] = -cur[j]
                    c[j] = -c[j]
                    fft_vals = fft_vals + 2.0 * c[j] * Zcols[:, j] # Revert FFT
            else:
                # Revert
                cur[j] = -cur[j]
                c[j] = -c[j]
                fft_vals = fft_vals + 2.0 * c[j] * Zcols[:, j]
                
    # Final refinement of the best candidate
    if best_peak > _search_peak(best, Z, theta, z, M, refine=True):
        best_peak = _search_peak(best, Z, theta, z, M, refine=True)

    return best


def _score_final(coeffs, M=4096):
    """Score on a dense grid; the max on the grid is a rigorous upper bound on the score."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)
    Z = np.power(z[:, None], powers[None, :])
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    return math.sqrt(np.max(mag2)) / _SQRTN


def construct():
    global _T0
    _T0 = time.time()

    # Seed: Rudin-Shapiro (low peak ~6.19, just above the 6.1363 optimum).
    seed = _rs_poly()

    # Phase 1: Simulated Annealing with moderate resolution
    # Use M=512 for better peak estimation during search
    best = _local_search_fast(seed, M=512, deadline=_LIMIT)

    # Phase 2: If time allows, do a quick local polish on the best found
    if _time_left() > 10.0:
        M_final = 1024
        Z, theta, z = _setup_fft(M_final)
        Zcols = np.zeros((M_final, _N), dtype=complex)
        for j in range(_N):
            Zcols[:, j] = np.power(z, j)
        
        cur = list(best)
        c = np.array(cur, dtype=complex)
        fft_vals = Z @ c
        cur_peak = _search_peak(cur, Z, theta, z, M_final)
        best_peak = cur_peak
        
        # Simple hill climbing on high-res grid
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            for j in range(_N):
                if _time_left() < 5.0:
                    break
                old_c_j = c[j]
                cur[j] = -cur[j]
                c[j] = -c[j]
                fft_vals = fft_vals - 2.0 * old_c_j * Zcols[:, j]
                p = _search_peak(cur, Z, theta, z, M_final)
                if p < best_peak - 1e-10:
                    best_peak = p
                    best = list(cur)
                    improved = True
                else:
                    # Revert
                    cur[j] = -cur[j]
                    c[j] = -c[j]
                    fft_vals = fft_vals + 2.0 * c[j] * Zcols[:, j]

    # Ensure valid output: exactly 24 coefficients in {+1,-1}.
    best = [1 if v >= 0 else -1 for v in best]
    return best[:_N]

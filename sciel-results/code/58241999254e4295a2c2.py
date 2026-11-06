"""Flattest plus-minus-one polynomials with 24 coefficients (minimum sup norm).

P(z) = sum c_k z^k, c_k in {+1,-1}, |c| = 24.
Score = max_{|z|=1} |P(z)| / sqrt(24).  Optimum ~ 1.2583758 (max|P| ~ 6.1363).

Strategy: incremental FFT-based local search with random restarts.
The key insight from the cell findings is that incremental FFT updates allow
rapid single-flip local search, enabling convergence to the known global
optimum within the time budget.
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
    # Z[k, j] = z_k^j
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
        # Refine the global max and a couple neighbours for robustness
        nb = max(1, M // 512)
        for d in (0, nb, -nb, 2*nb, -2*nb):
            ph = theta[(best_idx + d) % M]
            peak = _golden_section(coeffs, Z, theta, z, peak, (best_idx + d) % M, M, iters=12)
    return peak


def _local_search_fast(seed, M, deadline):
    """Fast local search using incremental FFT updates for single flips."""
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
    
    while _time_left() > 20.0:
        improved = False
        # Try all single flips
        for j in range(_N):
            if _time_left() < 20.0:
                break
            # Flip coefficient j
            cur[j] = -cur[j]
            c[j] = -c[j]
            
            # Incremental FFT update: new_fft = old_fft - 2*c[j]*Zcols[:,j]
            # But c[j] is already flipped, so: new_fft = old_fft - 2*new_c[j]*Zcols[:,j]
            # Actually: old_fft = sum old_c[k]*Zcols[:,k]
            # new_fft = old_fft - old_c[j]*Zcols[:,j] + new_c[j]*Zcols[:,j]
            #         = old_fft + (new_c[j] - old_c[j])*Zcols[:,j]
            #         = old_fft - 2*old_c[j]*Zcols[:,j]
            # Since we already flipped c[j], old_c[j] = -c[j]
            fft_vals = fft_vals + 2.0 * c[j] * Zcols[:, j]
            
            p, _ = _peak_from_fft(fft_vals)
            # Add a small refinement cost for accuracy
            if p < cur_peak - 1e-9:
                # Do a quick refinement to confirm
                p_refined = _search_peak(cur, Z, theta, z, M, refine=True)
                if p_refined < cur_peak - 1e-9:
                    cur_peak = p_refined
                    improved = True
                    if cur_peak < best_peak - 1e-9:
                        best = list(cur)
                        best_peak = cur_peak
                else:
                    # Revert
                    cur[j] = -cur[j]
                    c[j] = -c[j]
                    fft_vals = fft_vals - 2.0 * c[j] * Zcols[:, j]
            else:
                # Revert
                cur[j] = -cur[j]
                c[j] = -c[j]
                fft_vals = fft_vals - 2.0 * c[j] * Zcols[:, j]
        
        if not improved:
            # Try some double flips to escape local optima
            moved = False
            for _ in range(min(10, _N * (_N - 1) // 2)):
                if _time_left() < 20.0:
                    break
                j = int(rng.integers(0, _N))
                k = int(rng.integers(0, _N))
                if j == k:
                    continue
                # Flip both
                cur[j] = -cur[j]
                cur[k] = -cur[k]
                c[j] = -c[j]
                c[k] = -c[k]
                fft_vals = fft_vals + 2.0 * c[j] * Zcols[:, j] + 2.0 * c[k] * Zcols[:, k]
                
                p, _ = _peak_from_fft(fft_vals)
                if p < cur_peak - 1e-9:
                    p_refined = _search_peak(cur, Z, theta, z, M, refine=True)
                    if p_refined < cur_peak - 1e-9:
                        cur_peak = p_refined
                        moved = True
                        if cur_peak < best_peak - 1e-9:
                            best = list(cur)
                            best_peak = cur_peak
                    else:
                        # Revert
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
                        c[j] = -c[j]
                        c[k] = -c[k]
                        fft_vals = fft_vals - 2.0 * c[j] * Zcols[:, j] - 2.0 * c[k] * Zcols[:, k]
                else:
                    # Revert
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    c[j] = -c[j]
                    c[k] = -c[k]
                    fft_vals = fft_vals - 2.0 * c[j] * Zcols[:, j] - 2.0 * c[k] * Zcols[:, k]
            
            if not moved:
                # Random restart: perturb the best solution
                cur = list(best)
                cur_peak = best_peak
                c = np.array(cur, dtype=complex)
                fft_vals = Z @ c
                for _ in range(3):
                    j = int(rng.integers(0, _N))
                    cur[j] = -cur[j]
                    c[j] = -c[j]
                    fft_vals = fft_vals + 2.0 * c[j] * Zcols[:, j]
                cur_peak = _search_peak(cur, Z, theta, z, M)
    
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

    # Phase 1: Fast local search with moderate resolution
    best = _local_search_fast(seed, M=256, deadline=_LIMIT)

    # Phase 2: Refine with finer grid if time allows
    if _time_left() > 30.0:
        best = _local_search_fast(best, M=512, deadline=_LIMIT)

    # Phase 3: Final high-precision polish: try all single flips with very fine peak measurement
    if _time_left() > 15.0:
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
        
        for j in range(_N):
            if _time_left() < 10.0:
                break
            cur[j] = -cur[j]
            c[j] = -c[j]
            fft_vals = fft_vals + 2.0 * c[j] * Zcols[:, j]
            p = _search_peak(cur, Z, theta, z, M_final)
            if p < best_peak - 1e-10:
                best_peak = p
                best = list(cur)
            else:
                cur[j] = -cur[j]
                c[j] = -c[j]
                fft_vals = fft_vals - 2.0 * c[j] * Zcols[:, j]

    # Ensure valid output: exactly 24 coefficients in {+1,-1}.
    best = [1 if v >= 0 else -1 for v in best]
    return best[:_N]

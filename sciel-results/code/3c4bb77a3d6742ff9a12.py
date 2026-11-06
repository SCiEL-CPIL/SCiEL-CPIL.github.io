"""Model-and-optimize for flattest plus-minus-one 24-coefficient polynomial.

P(z) = sum c_k z^k, c_k in {+1,-1}, |c| = 24.
Score = max_{|z|=1} |P(z)| / sqrt(24).  Optimum ~ 1.2583758 (max|P| ~ 6.1363).

Strategy: Simulated Annealing with FFT-based evaluation for speed.
1. Initialize with Rudin-Shapiro coefficients.
2. Run Simulated Annealing:
   - Flip random coefficient.
   - Evaluate peak using FFT (fast).
   - Accept based on Metropolis criterion.
3. Polish the best candidate with high-resolution local search.
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
    """Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]


def _peak_fft(coeffs, M):
    """Calculate peak magnitude using FFT on a grid of size M."""
    # coeffs are real (+1/-1)
    # FFT computes sum c_j * exp(-i 2 pi k j / M)
    # This is P(z_k) for z_k = exp(-i 2 pi k / M)
    # |P(z)| is invariant to conjugation of z for real coeffs, so magnitudes are correct.
    # We need M >= _N for unique representation? No, we are sampling the polynomial.
    # If M < _N, there is aliasing. We must have M >= _N.
    # Actually, to approximate the sup norm on the unit circle, M should be large.
    # FFT computes the DFT of the sequence padded with zeros to length M.
    # This corresponds to evaluating the polynomial P(z) = sum_{j=0}^{N-1} c_j z^j at z_k = exp(-2 pi i k / M).
    c = np.zeros(M)
    c[:_N] = coeffs
    vals = np.fft.fft(c)
    return np.max(np.abs(vals))


def _peak_refine(coeffs, M, phase, iters=20):
    """Golden section refinement of peak around phase."""
    lo = phase - math.pi / M
    hi = phase + math.pi / M
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)

    def f(a):
        z = np.exp(1j * a)
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return s.real * s.real + s.imag * s.imag

    f_lo = f(lo)
    f_hi = f(hi)
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            lo = m1
            f_lo = f1
        else:
            hi = m2
            f_hi = f2
    return math.sqrt(max(f_lo, f_hi))


def _score_final(coeffs, M=4096):
    """High precision score calculation."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    best_idx = np.argmax(mag2)
    peak = math.sqrt(mag2[best_idx])
    # Refine the peak
    peak = _peak_refine(coeffs, M, theta[best_idx], iters=30)
    return peak / _SQRTN


def _simulate_annealing(seed, M, T_start, T_end, steps):
    """Run simulated annealing."""
    cur = np.array(seed, dtype=np.int8)
    cur_peak = _peak_fft(cur, M)
    
    best = list(cur)
    best_peak = cur_peak
    
    rng = np.random.default_rng(42)
    
    # Cooling schedule: exponential
    # T_k = T_start * (T_end / T_start) ** (k / steps)
    ratio = T_end / T_start
    step_ratio = ratio ** (1.0 / steps)
    
    T = T_start
    
    for i in range(steps):
        if _time_left() < 10.0:
            break
            
        # Flip a random coefficient
        j = int(rng.integers(0, _N))
        cur[j] = -cur[j]
        
        new_peak = _peak_fft(cur, M)
        delta = new_peak - cur_peak
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best = list(cur)
        else:
            cur[j] = -cur[j] # revert
            
        T *= step_ratio
        
    return best, best_peak


def construct():
    global _T0
    _T0 = time.time()

    # Seed
    seed = _rs_poly()
    
    # Phase 1: Simulated Annealing
    # Use a reasonably fine grid for SA to balance accuracy and speed.
    # M=1024 is fast with FFT.
    M_sa = 1024
    T_start = 0.5
    T_end = 0.01
    
    # Estimate max steps based on time
    # Each step takes ~10-20us? Let's say 50us to be safe.
    # 100s / 50us = 2,000,000 steps.
    # We'll just run until time is up.
    
    best, best_peak = _simulate_annealing(seed, M_sa, T_start, T_end, 10**9)
    
    # Phase 2: Polish
    # Use high resolution local search on the best candidate.
    if _time_left() > 5.0:
        M_polish = 2048
        cur = np.array(best, dtype=np.int8)
        cur_peak = _peak_fft(cur, M_polish)
        
        # Single flip hill climbing
        improved = True
        while improved and _time_left() > 2.0:
            improved = False
            for j in range(_N):
                cur[j] = -cur[j]
                p = _peak_fft(cur, M_polish)
                if p < cur_peak:
                    cur_peak = p
                    improved = True
                else:
                    cur[j] = -cur[j]
        
        best = list(cur)
        
    # Phase 3: Final verification and potential double-flip escape if time permits
    if _time_left() > 5.0:
        # Try a few double flips from the best state with high res
        cur = np.array(best, dtype=np.int8)
        cur_peak = _peak_fft(cur, 2048)
        
        for _ in range(10): # Try 10 random double flips
            if _time_left() < 2.0:
                break
            j = int(np.random.randint(0, _N))
            k = int(np.random.randint(0, _N))
            if j == k: continue
            
            cur[j] = -cur[j]
            cur[k] = -cur[k]
            p = _peak_fft(cur, 2048)
            if p < cur_peak:
                cur_peak = p
                best = list(cur)
            else:
                cur[j] = -cur[j]
                cur[k] = -cur[k]
                
    # Final score calculation with high precision
    # The evaluator does this, but we return the coefficients.
    # We just need to return the best coefficients.
    
    return [1 if v > 0 else -1 for v in best]

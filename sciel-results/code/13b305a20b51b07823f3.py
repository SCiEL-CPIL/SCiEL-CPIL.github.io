import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 145.0  # Safety margin from 150s limit

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n):
    """Generate Rudin-Shapiro coefficients of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    
    R = [1]
    S = [1]
    while len(R) < n:
        new_R = R + S
        new_S = R + [-s for s in S]
        R = new_R
        S = new_S
    return R[:n]

def _calc_peak(coeffs, M=2048, refine_top=10, refine_iters=40):
    """
    Calculate the peak magnitude of P(z) on the unit circle.
    Uses dense sampling and golden-section refinement for top peaks.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks and refine
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers_arr = powers
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

# Cache for Z matrix to speed up _calc_peak_fast
_Z_CACHE = {}
def _get_Z(M):
    if M not in _Z_CACHE:
        k = np.arange(M)
        theta = 2.0 * np.pi * k / M
        powers = np.arange(_N)
        _Z_CACHE[M] = np.exp(1j * np.outer(theta, powers))
    return _Z_CACHE[M]

def _calc_peak_fast(coeffs, M=1024):
    """
    Very fast peak calculation for local search iterations.
    Uses only dense sampling without refinement for speed.
    """
    c = np.array(coeffs, dtype=complex)
    Z = _get_Z(M)
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    return math.sqrt(peak_val)

def _simulated_annealing(seed, max_time=100.0, M=1024):
    """
    Simulated Annealing to minimize the peak.
    """
    cur = list(seed)
    cur_peak = _calc_peak_fast(cur, M)
    best = list(cur)
    best_peak = _calc_peak(cur, M=4096, refine_top=10, refine_iters=40)
    
    start_time = time.time()
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < max_time:
        # Pick a random position to flip
        j = rng.integers(0, _N)
        cur[j] = -cur[j]
        
        new_peak = _calc_peak_fast(cur, M)
        delta = new_peak - cur_peak
        
        if delta < 0:
            # Accept improvement
            cur_peak = new_peak
            if new_peak < best_peak - 1e-10:
                # Verify with high accuracy
                hp = _calc_peak(cur, M=4096, refine_top=10, refine_iters=40)
                if hp < best_peak:
                    best = list(cur)
                    best_peak = hp
        else:
            # Accept worse solution with probability exp(-delta/T)
            if rng.random() < math.exp(-delta / T):
                cur_peak = new_peak
            else:
                # Revert
                cur[j] = -cur[j]
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best, best_peak

def construct():
    global _T0
    _T0 = time.time()
    
    # Known good seeds to try
    # Rudin-Shapiro
    seed_rs = _rs_poly(_N)
    
    # Try several known optimal or near-optimal polynomials
    # Based on Odlyzko's tables for n=24
    candidate_seeds = [
        seed_rs,
        [1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
        [1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1],
        [1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
    ]
    
    best_coeffs = seed_rs
    best_peak = _calc_peak(best_coeffs, M=4096, refine_top=10, refine_iters=40)
    
    # First, evaluate all candidates quickly
    for seed in candidate_seeds:
        if _time_left() < 10.0:
            break
        p = _calc_peak(seed, M=2048, refine_top=5, refine_iters=20)
        if p < best_peak - 1e-8:
            best_coeffs = list(seed)
            best_peak = p
    
    # Run Simulated Annealing on the best candidate
    if _time_left() > 30.0:
        sa_coeffs, sa_peak = _simulated_annealing(best_coeffs, max_time=min(80.0, _time_left() - 20.0))
        if sa_peak < best_peak:
            best_coeffs = sa_coeffs
            best_peak = sa_peak
    
    # Final high-precision polish
    if _time_left() > 5.0:
        final_peak = _calc_peak(best_coeffs, M=8192, refine_top=10, refine_iters=50)
        best_peak = final_peak
    
    # Ensure coefficients are +1 or -1
    best_coeffs = [1 if c >= 0 else -1 for c in best_coeffs]
    
    return best_coeffs

if __name__ == "__main__":
    # Self-test
    coeffs = construct()
    print(coeffs)
    print(len(coeffs))
    print(_calc_peak(coeffs, M=8192, refine_top=10, refine_iters=50))

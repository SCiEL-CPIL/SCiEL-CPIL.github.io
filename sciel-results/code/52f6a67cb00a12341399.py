import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 150.0

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
    
    # Recursive generation
    # R_{k+1} = R_k S_k
    # S_{k+1} = R_k (-S_k)
    # Start with R_1=[1], S_1=[1]
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
    z = np.exp(1j * theta)
    
    # Vectorized evaluation: vals = sum_j c_j * z^j
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks and refine
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        
        # Precompute powers for speed
        powers_arr = powers
        
        for idx in top_indices:
            # Refine around theta[idx]
            # Use a window that covers the full grid spacing
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            # Golden section search
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

def _calc_peak_fast(coeffs, M=2048, refine_top=5, refine_iters=20):
    """
    Faster peak calculation for local search iterations.
    Less accurate but faster.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    # Vectorized evaluation
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Light refinement for top few peaks
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

def _local_search(seed, M=2048, max_time=140.0):
    """
    Perform local search to minimize the peak.
    Uses fast peak calculation for speed, with final verification.
    """
    cur = list(seed)
    cur_peak = _calc_peak_fast(cur, M, refine_top=5, refine_iters=20)
    best = list(cur)
    best_peak = cur_peak
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Single flip pass
        for j in range(_N):
            if time.time() - start_time > max_time:
                break
            cur[j] = -cur[j]
            p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
            if p < cur_peak - 1e-12:
                cur_peak = p
                improved = True
                if cur_peak < best_peak - 1e-12:
                    best = list(cur)
                    best_peak = cur_peak
            else:
                cur[j] = -cur[j]
        
        if not improved:
            # Try double flips
            double_improved = False
            for j in range(_N):
                if time.time() - start_time > max_time:
                    break
                for k in range(j + 1, _N):
                    if time.time() - start_time > max_time:
                        break
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
                    if p < cur_peak - 1e-12:
                        cur_peak = p
                        double_improved = True
                        if cur_peak < best_peak - 1e-12:
                            best = list(cur)
                            best_peak = cur_peak
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
            
            if not double_improved:
                break
            else:
                improved = True
        
        # Periodic high-accuracy check
        if time.time() - start_time < max_time - 5.0:
            high_acc_peak = _calc_peak(best, M=4096, refine_top=10, refine_iters=40)
            if high_acc_peak < best_peak:
                best_peak = high_acc_peak
    
    # Final high-accuracy verification
    final_peak = _calc_peak(best, M=8192, refine_top=10, refine_iters=50)
    return best, final_peak

def construct():
    global _T0
    _T0 = time.time()
    
    # Primary Seed: Rudin-Shapiro
    seed_rs = _rs_poly(_N)
    
    best_coeffs = seed_rs
    best_peak = _calc_peak(best_coeffs, M=4096, refine_top=10, refine_iters=40)
    
    # Run local search on RS seed
    res_coeffs, res_peak = _local_search(seed_rs, M=2048, max_time=100.0)
    if res_peak < best_peak:
        best_coeffs = res_coeffs
        best_peak = res_peak
    
    # If time remains, try a few random restarts
    if _time_left() > 25.0:
        rng = np.random.default_rng(42)
        for _ in range(3):
            if _time_left() < 15.0:
                break
            # Random seed with RS-like structure
            seed_alt = list(seed_rs)
            for _ in range(2):
                idx = int(rng.integers(0, _N))
                seed_alt[idx] = -seed_alt[idx]
            
            r_coeffs, r_peak = _local_search(seed_alt, M=1024, max_time=min(15.0, _time_left() - 10.0))
            if r_peak < best_peak:
                v_peak = _calc_peak(r_coeffs, M=4096, refine_top=10, refine_iters=40)
                if v_peak < best_peak:
                    best_coeffs = r_coeffs
                    best_peak = v_peak

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

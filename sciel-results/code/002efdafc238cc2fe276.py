import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 148.0  # safety margin

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

def _eval_poly(c, theta):
    """Evaluate P(e^{i*theta}) using Horner's method. Fast for single point."""
    # c[0] + c[1]*z + c[2]*z^2 + ... + c[23]*z^23
    # Horner: ((c[23]*z + c[22])*z + c[21])*z + ... + c[0]
    z = np.exp(1j * theta)
    result = c[-1]
    for i in range(_N - 2, -1, -1):
        result = result * z + c[i]
    return result

def _eval_poly_mag2(c, theta):
    """Return |P(e^{i*theta})|^2 using Horner's method."""
    z = np.exp(1j * theta)
    result = c[-1]
    for i in range(_N - 2, -1, -1):
        result = result * z + c[i]
    return result.real * result.real + result.imag * result.imag

def _calc_peak(coeffs, M=2048, refine_top=10, refine_iters=40):
    """
    Calculate the peak magnitude of P(z) on the unit circle.
    Uses dense sampling and golden-section refinement for top peaks.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    # Vectorized evaluation via FFT-based approach or direct
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks and refine
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                return _eval_poly_mag2(c, a)

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

def _calc_peak_fast(coeffs, M=1024, refine_top=3, refine_iters=12):
    """
    Faster peak calculation for local search iterations.
    Uses fewer grid points and less refinement but is much faster.
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
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                return _eval_poly_mag2(c, a)

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

def _local_search(seed, M=1024, max_time=100.0):
    """
    Perform local search to minimize the peak.
    Uses fast peak calculation for speed, with final verification.
    """
    cur = list(seed)
    cur_peak = _calc_peak_fast(cur, M, refine_top=3, refine_iters=12)
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
            p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=10)
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
                    p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=10)
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
                # Try triple flips if no double flip worked
                triple_improved = False
                for j in range(_N):
                    if time.time() - start_time > max_time:
                        break
                    for k in range(j + 1, _N):
                        if time.time() - start_time > max_time:
                            break
                        for m in range(k + 1, _N):
                            if time.time() - start_time > max_time:
                                break
                            cur[j] = -cur[j]
                            cur[k] = -cur[k]
                            cur[m] = -cur[m]
                            p = _calc_peak_fast(cur, M, refine_top=2, refine_iters=8)
                            if p < cur_peak - 1e-12:
                                cur_peak = p
                                triple_improved = True
                                if cur_peak < best_peak - 1e-12:
                                    best = list(cur)
                                    best_peak = cur_peak
                            else:
                                cur[j] = -cur[j]
                                cur[k] = -cur[k]
                                cur[m] = -cur[m]
                
                if not triple_improved:
                    break
                else:
                    improved = True
            else:
                improved = True
        
        # Periodic high-accuracy check (less frequent to save time)
        if time.time() - start_time < max_time - 8.0 and (time.time() - start_time) % 20.0 < 0.5:
            high_acc_peak = _calc_peak(best, M=4096, refine_top=8, refine_iters=30)
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
    
    best_coeffs = list(seed_rs)
    best_peak = _calc_peak(best_coeffs, M=4096, refine_top=10, refine_iters=40)
    
    # Run local search on RS seed with generous time budget
    res_coeffs, res_peak = _local_search(seed_rs, M=1024, max_time=min(80.0, _time_left() - 30.0))
    if res_peak < best_peak:
        best_coeffs = res_coeffs
        best_peak = res_peak
    
    # If time remains, try a few random restarts
    if _time_left() > 30.0:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if _time_left() < 15.0:
                break
            # Random seed with RS-like structure
            seed_alt = list(seed_rs)
            n_flips = 1 + int(rng.integers(0, 3))  # 1 to 3 flips
            for _ in range(n_flips):
                idx = int(rng.integers(0, _N))
                seed_alt[idx] = -seed_alt[idx]
            
            max_t = min(12.0, _time_left() - 10.0)
            if max_t < 3.0:
                break
            r_coeffs, r_peak = _local_search(seed_alt, M=1024, max_time=max_t)
            if r_peak < best_peak:
                v_peak = _calc_peak(r_coeffs, M=4096, refine_top=10, refine_iters=40)
                if v_peak < best_peak:
                    best_coeffs = r_coeffs
                    best_peak = v_peak

    # Final high-precision polish with tight tolerance
    if _time_left() > 5.0:
        final_peak = _calc_peak(best_coeffs, M=16384, refine_top=15, refine_iters=60)
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

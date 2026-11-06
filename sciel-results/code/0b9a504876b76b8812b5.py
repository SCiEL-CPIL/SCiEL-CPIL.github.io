import time
import numpy as np
import math

_N = 24
_SQRN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 115.0 # Safety margin

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly():
    """Generate Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]

def _setup_eval(M=4096):
    """
    Precompute DFT matrix columns for fast incremental updates.
    Returns Z (M x N).
    """
    j = np.arange(M)
    k = np.arange(_N)
    J, K = np.meshgrid(j, k, indexing='ij')
    Z = np.exp(1j * 2 * np.pi * J * K / M)
    return Z

def _fast_local_search(coeffs, M=2048, max_iters=5000):
    """
    Local search using precomputed deltas with single and double flips.
    coeffs: list of 24 ints
    """
    Z = _setup_eval(M)
    coeffs_arr = np.array(coeffs, dtype=float)
    
    # Precompute single-flip deltas: delta_k = -2 * c_k * Z[:, k]
    # But c_k changes as we flip, so we recompute deltas when coefficients change
    # Instead, store Z[:, k] and compute delta on the fly based on current c_k
    
    # Initial evaluation
    vals = Z @ coeffs_arr
    mags = np.abs(vals)
    current_max = np.max(mags)
    
    best_coeffs = coeffs[:]
    best_max = current_max
    no_improve_count = 0
    
    # Precompute Z columns for fast access
    Z_cols = [Z[:, k] for k in range(_N)]
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        best_flip_idx = -1
        best_flip2_idx = -1
        best_new_max = current_max
        
        # Try single flips
        for k in range(_N):
            old_c = coeffs_arr[k]
            delta = -2.0 * old_c * Z_cols[k]
            new_vals = vals + delta
            new_max = np.max(np.abs(new_vals))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = k
        
        # Try double flips (only if no single flip improved)
        if best_flip_idx == -1:
            for k1 in range(_N):
                for k2 in range(k1 + 1, _N):
                    old_c1 = coeffs_arr[k1]
                    old_c2 = coeffs_arr[k2]
                    delta = -2.0 * old_c1 * Z_cols[k1] - 2.0 * old_c2 * Z_cols[k2]
                    new_vals = vals + delta
                    new_max = np.max(np.abs(new_vals))
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_flip_idx = k1
                        best_flip2_idx = k2
        
        if best_flip_idx != -1:
            # Apply best flip(s)
            k = best_flip_idx
            old_c = coeffs_arr[k]
            coeffs_arr[k] = -old_c
            vals += -2.0 * old_c * Z_cols[k]
            
            if best_flip2_idx != -1:
                k2 = best_flip2_idx
                old_c2 = coeffs_arr[k2]
                coeffs_arr[k2] = -old_c2
                vals += -2.0 * old_c2 * Z_cols[k2]
            
            current_max = best_new_max
            best_coeffs = coeffs_arr.astype(int).tolist()
            best_max = current_max
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                break
            
    return best_coeffs, best_max

def _eval_precise(coeffs, M=16384):
    """
    More precise evaluation for final comparison with golden-section refinement.
    """
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:_N] = coeffs
    vals = np.fft.ifft(c_padded) * M
    mags = np.abs(vals)
    
    # Find top 15 peaks
    top_idx = np.argsort(mags)[-15:]
    max_val = mags[top_idx[0]]
    
    # Precompute powers for faster evaluation
    k = np.arange(_N)
    c_arr = np.array(coeffs, dtype=complex)
    
    def P_mag(theta):
        zk = np.exp(1j * theta * k)
        return np.abs(np.dot(c_arr, zk))
        
    for idx in top_idx:
        theta0 = 2 * np.pi * idx / M
        a = theta0 - np.pi / M
        b = theta0 + np.pi / M
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        for _ in range(45):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = P_mag(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = P_mag(d)
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def construct():
    t_start = time.time()
    np.random.seed(42)  # For reproducibility
    
    # Initial candidate: Rudin-Shapiro
    best_coeffs = _rs_poly()
    best_val = _eval_precise(best_coeffs)
    
    # Strategy:
    # 1. Run fast local search from RS with high resolution.
    # 2. Run fast local search from random starts.
    
    # Local search 1: RS with higher resolution
    c1, v1 = _fast_local_search(best_coeffs, M=4096, max_iters=10000)
    if v1 < best_val:
        best_coeffs = c1
        best_val = v1
        
    # Random restarts with varied starting points
    restart_count = 0
    while _time_left() > 20.0:
        # Mix of random and perturbed RS starts
        if np.random.rand() < 0.4:
            # Perturb RS or best found
            c_rand = best_coeffs[:]
            num_flips = np.random.randint(1, 6)
            for _ in range(num_flips):
                c_rand[np.random.randint(_N)] *= -1
        else:
            c_rand = [1 if np.random.rand() > 0.5 else -1 for _ in range(_N)]
        
        # Use moderate resolution for speed
        c2, v2 = _fast_local_search(c_rand, M=2048, max_iters=5000)
        
        if v2 < best_val:
            # Verify with higher precision
            v2_precise = _eval_precise(c2)
            if v2_precise < best_val:
                best_coeffs = c2
                best_val = v2_precise
                # Do a follow-up local search from this new best
                if _time_left() > 15.0:
                    c2b, v2b = _fast_local_search(best_coeffs, M=4096, max_iters=3000)
                    v2b_precise = _eval_precise(c2b)
                    if v2b_precise < best_val:
                        best_coeffs = c2b
                        best_val = v2b_precise
        
        restart_count += 1
        
    # Final polish with higher resolution
    if _time_left() > 15.0:
        c3, v3 = _fast_local_search(best_coeffs, M=8192, max_iters=8000)
        if v3 < best_val:
             v3_precise = _eval_precise(c3)
             if v3_precise < best_val:
                 best_coeffs = c3
                 best_val = v3_precise
    
    # Final high-precision evaluation
    final_val = _eval_precise(best_coeffs, M=32768)
    
    return best_coeffs

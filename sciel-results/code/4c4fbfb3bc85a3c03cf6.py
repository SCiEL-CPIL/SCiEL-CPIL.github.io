import time
import numpy as np
import math
from itertools import combinations

_N = 24
_SQRN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0  # Safety margin for 150s wall-clock

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
    Precompute DFT matrix columns and all single/double/triple flip deltas.
    Returns Z, Z_cols, double_deltas, triple_deltas.
    """
    j = np.arange(M)
    k = np.arange(_N)
    J, K = np.meshgrid(j, k, indexing='ij')
    Z = np.exp(1j * 2 * np.pi * J * K / M)
    
    # Precompute Z columns as separate arrays for fast access
    Z_cols = [Z[:, k] for k in range(_N)]
    
    # Precompute all double-flip delta vectors (sum of Z columns)
    double_deltas = {}
    for k1, k2 in combinations(range(_N), 2):
        double_deltas[(k1, k2)] = Z_cols[k1] + Z_cols[k2]
    
    # Precompute all triple-flip delta vectors
    triple_deltas = {}
    for k1, k2, k3 in combinations(range(_N), 3):
        triple_deltas[(k1, k2, k3)] = Z_cols[k1] + Z_cols[k2] + Z_cols[k3]
    
    return Z, Z_cols, double_deltas, triple_deltas

def _fast_local_search(coeffs, M=2048, max_iters=5000):
    """
    Local search using incremental FFT updates with single, double, and triple flips.
    coeffs: list of 24 ints
    """
    Z, Z_cols, double_deltas, triple_deltas = _setup_eval(M)
    coeffs_arr = np.array(coeffs, dtype=float)
    
    # Initial evaluation
    vals = Z @ coeffs_arr
    mags = np.abs(vals)
    current_max = np.max(mags)
    
    best_coeffs = coeffs[:]
    best_max = current_max
    no_improve_count = 0
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        best_new_max = current_max
        best_move = None  # tuple of indices to flip
        
        # Try single flips
        for k in range(_N):
            old_c = coeffs_arr[k]
            delta = -2.0 * old_c * Z_cols[k]
            new_vals = vals + delta
            new_max = np.max(np.abs(new_vals))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_move = (k,)
        
        # Try double flips (compare with best single)
        if best_move is None or len(best_move) == 1:
            for (k1, k2) in double_deltas:
                old_c1 = coeffs_arr[k1]
                old_c2 = coeffs_arr[k2]
                delta = -2.0 * old_c1 * Z_cols[k1] - 2.0 * old_c2 * Z_cols[k2]
                new_vals = vals + delta
                new_max = np.max(np.abs(new_vals))
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = (k1, k2)
        
        # Try triple flips (compare with best single/double)
        if best_move is None or len(best_move) <= 2:
            for (k1, k2, k3) in triple_deltas:
                old_c1 = coeffs_arr[k1]
                old_c2 = coeffs_arr[k2]
                old_c3 = coeffs_arr[k3]
                delta = (-2.0 * old_c1 * Z_cols[k1] - 2.0 * old_c2 * Z_cols[k2] 
                       - 2.0 * old_c3 * Z_cols[k3])
                new_vals = vals + delta
                new_max = np.max(np.abs(new_vals))
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = (k1, k2, k3)
        
        if best_move is not None:
            # Apply best move
            for k in best_move:
                old_c = coeffs_arr[k]
                coeffs_arr[k] = -old_c
                vals += -2.0 * old_c * Z_cols[k]
            
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

def _eval_precise(coeffs, M=32768):
    """
    More precise evaluation for final comparison.
    """
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:_N] = coeffs
    vals = np.fft.ifft(c_padded) * M
    mags = np.abs(vals)
    
    # Find top 20 peaks
    top_idx = np.argsort(mags)[-20:]
    max_val = mags[top_idx[0]]
    
    # Precompute powers for faster evaluation
    k = np.arange(_N)
    
    def P_mag(theta):
        z = np.exp(1j * theta)
        zk = np.exp(1j * theta * k)
        return np.abs(np.sum(coeffs * zk))
        
    for idx in top_idx:
        theta0 = 2 * np.pi * idx / M
        a = theta0 - np.pi / M
        b = theta0 + np.pi / M
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        for _ in range(45):  # More iterations for better precision
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
    # 1. Run fast local search from RS.
    # 2. Run fast local search from random starts.
    
    # Local search 1: RS
    c1, v1 = _fast_local_search(best_coeffs, M=4096, max_iters=8000)
    if v1 < best_val:
        best_coeffs = c1
        best_val = v1
        
    # Random restarts with varied starting points
    while _time_left() > 15.0:
        # Mix of random and perturbed RS starts
        if np.random.rand() < 0.3:
            # Perturb RS
            c_rand = best_coeffs[:]
            for _ in range(np.random.randint(1, 5)):
                c_rand[np.random.randint(_N)] *= -1
        else:
            c_rand = [1 if np.random.rand() > 0.5 else -1 for _ in range(_N)]
        
        c2, v2 = _fast_local_search(c_rand, M=2048, max_iters=4000)
        
        if v2 < best_val:
            # Verify with higher precision
            v2_precise = _eval_precise(c2)
            if v2_precise < best_val:
                best_coeffs = c2
                best_val = v2_precise
                
    # Final polish with higher resolution
    if _time_left() > 10.0:
        c3, v3 = _fast_local_search(best_coeffs, M=8192, max_iters=5000)
        if v3 < best_val:
             v3_precise = _eval_precise(c3)
             if v3_precise < best_val:
                 best_coeffs = c3
                 best_val = v3_precise

    return best_coeffs

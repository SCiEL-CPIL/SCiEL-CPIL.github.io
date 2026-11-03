"""
Combined approach: Spectral subspace projection + Local sign search with FFT updates.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def _score(h):
    """Compute the raw score: max(|ac|) / (sum(h))^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv_full(h)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_basis(K):
    """
    Generate a low-dimensional cosine basis.
    h(x) = sum_{k=0}^{K-1} a_k * cos(pi * k * x / N)
    We return the matrix B where h = B @ a.
    """
    x = np.arange(N)
    B = np.zeros((N, K))
    for k in range(K):
        B[:, k] = np.cos(np.pi * k * x / N)
    return B

def _project_to_spectral(h, K):
    """
    Project h onto the spectral subspace and return the coefficients.
    """
    B = _spectral_basis(K)
    # Least squares projection
    a, _, _, _ = np.linalg.lstsq(B, h, rcond=None)
    return a

def _reconstruct_from_spectral(a, K):
    """
    Reconstruct h from spectral coefficients.
    """
    B = _spectral_basis(K)
    return B @ a

def _smooth_obj_spectral(a, K, beta=50.0):
    """
    Objective function for optimization in the spectral subspace.
    Uses soft-max approximation for the peak.
    """
    h = _reconstruct_from_spectral(a, K)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        S = 0.0
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        log_sum = np.log(np.sum(np.exp(scaled)))
        S = m + log_sum / beta
    
    return S / (s * s)

def _local_search_signs(h, deadline, max_passes=5):
    """
    Local search: flip signs of elements to minimize max(|ac|)/(sum)^2.
    Uses incremental FFT updates for speed.
    """
    h = h.copy()
    best_val = _score(h)
    H = np.fft.rfft(h, _FSIZE)
    
    # Precompute basis vectors for FFT updates
    freqs = np.arange(_FSIZE // 2 + 1)
    basis_cache = {}
    for i in range(N):
        basis_cache[i] = np.exp(-2j * np.pi * freqs * i / _FSIZE)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = h[i]
            h[i] = -old_val
            
            # Update FFT incrementally
            delta = -2 * old_val * basis_cache[i]
            H_new = H + delta
            
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            integral = np.sum(h)
            
            if abs(integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (integral * integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            break
    
    return h, best_val

def _local_search_blocks(h, deadline, block_sizes=[2, 3, 5]):
    """
    Local search: flip contiguous blocks of signs.
    """
    h = h.copy()
    best_val = _score(h)
    H = np.fft.rfft(h, _FSIZE)
    
    # Precompute basis vectors
    freqs = np.arange(_FSIZE // 2 + 1)
    basis_cache = {}
    for i in range(N):
        basis_cache[i] = np.exp(-2j * np.pi * freqs * i / _FSIZE)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, bs):
                if time.time() > deadline:
                    break
                
                # Flip block
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                
                # Update FFT
                delta = np.zeros(_FSIZE // 2 + 1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2 * block[j] * basis_cache[idx]
                
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                integral = np.sum(h)
                
                if abs(integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (integral * integral)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    improved = True
                else:
                    h[start:start+bs] = block
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin
    
    best_score = np.inf
    best_h = None
    
    def update_best(h):
        nonlocal best_score, best_h
        s = np.sum(h)
        if abs(s) < 1e-12:
            return
        val = _score(h)
        if val < best_score:
            best_score = val
            best_h = h.copy()
    
    # 1. Spectral Subspace Optimization
    # Try different dimensions K
    for K in [10, 20, 50]:
        if time.time() > deadline - 20:
            break
        
        # Initialize with random coefficients
        rng = np.random.default_rng(42 + K)
        a_init = rng.standard_normal(K)
        
        # Optimize in spectral subspace
        try:
            res = minimize(_smooth_obj_spectral, a_init, args=(K, 50.0),
                          method='L-BFGS-B', options={'maxiter': 100, 'disp': False})
            a_opt = res.x
            h_opt = _reconstruct_from_spectral(a_opt, K)
            
            # Normalize
            s = np.sum(h_opt)
            if abs(s) > 1e-9:
                h_opt = h_opt / s
                update_best(h_opt)
        except Exception:
            pass
    
    # 2. Local Search Refinement
    if best_h is not None:
        # Sign flipping
        if time.time() < deadline - 10:
            h_ref, val_ref = _local_search_signs(best_h, deadline - 10, max_passes=3)
            update_best(h_ref)
        
        # Block flipping
        if time.time() < deadline - 5:
            h_ref, val_ref = _local_search_blocks(best_h, deadline - 5, block_sizes=[2, 3])
            update_best(h_ref)
    
    # Fallback
    if best_h is None:
        return np.ones(N) / N
    
    return best_h

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _build_basis():
    """Precompute basis vectors for fast local search.
    
    BASIS[k, n] = 1 if n == k (for flip of h[k])
    For flipping h[k]: delta_ac[n] = -4*h[k]*h[n-k] + 4*delta(n,2k)
    We precompute the structure.
    """
    # For each k, the delta when flipping h[k] is:
    # delta_ac[n] = -4 * h[k] * h[n-k] for n in [k, k+N-1]
    #              + 4 * delta(n, 2k)
    # This depends on h, so we can't fully precompute.
    # Instead, we precompute the indices.
    
    # For block flips, we can use a different approach:
    # Precompute the FFT of basis vectors
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # BASIS_BINS[k] = fft of vector with 1 at position k, 0 elsewhere
    # Then autoconv of h can be computed as sum_k h[k] * (h convolved with delta_k)
    # But that's not quite right for autoconv.
    
    # Alternative: for local search, precompute the effect of flipping each bit
    # delta_ac_flip[k, n] = -4 * h[n-k] + 4*delta(n,2k) * (1/h[k] if h[k]!=0 else 0)
    # This still depends on h.
    
    # Better approach: for the local search, we'll use a hybrid:
    # 1. Single-bit flips: use incremental update (already fast)
    # 2. Block flips: precompute FFT of indicator vectors
    
    # Precompute FFT of indicator vectors for blocks
    # For a block of size bs starting at start, the flip delta is complex.
    # Instead, let's precompute the FFT of h once and update incrementally.
    
    return None  # Placeholder for now

def _fast_local_search(h, max_passes=500, deadline=None, block_sizes=(1, 2, 3, 4, 8)):
    h = h.copy()
    s = np.sum(h)
    A = _autoconv_fft(h)
    current_score = np.max(np.abs(A)) / (s * s)
    
    passes_since_recompute = 0
    rng = np.random.default_rng(12345)
    
    # Precompute for faster block flips
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Single-bit flips
        for k in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            h_k = h[k]
            A_new = A.copy()
            # A_new[n] -= 4 * h_k * h[n-k]
            # Valid for n in [k, k+N-1]
            A_new[k:k+N] -= 4 * h_k * h
            
            # A_new[n] += 4 * delta(n, 2k)
            if 2*k < 2*N - 1:
                A_new[2*k] += 4.0
            
            s_new = s - 2 * h_k
            
            if abs(s_new) < 1e-9:
                continue
                
            score_new = np.max(np.abs(A_new)) / (s_new * s_new)
            
            if score_new < current_score:
                h[k] = -h_k
                A = A_new
                s = s_new
                current_score = score_new
                improved = True
        
        # Block flips (try a few random blocks of various sizes)
        for _ in range(12):  # Increased from 8 to 12
            if deadline is not None and time.time() > deadline:
                break
            bs = block_sizes[rng.integers(0, len(block_sizes))]
            start = rng.integers(0, max(1, N - bs + 1))
            idx = np.arange(start, start + bs)
            
            # Use FFT recompute for blocks (simpler and correct)
            h_trial = h.copy()
            h_trial[idx] = -h_trial[idx]
            s_new = s - 2 * np.sum(h_trial[idx])
            
            if abs(s_new) < 1e-9:
                continue
                
            A_new = _autoconv_fft(h_trial)
            score_new = np.max(np.abs(A_new)) / (s_new * s_new)
            
            if score_new < current_score:
                h = h_trial
                A = A_new
                s = s_new
                current_score = score_new
                improved = True
        
        if not improved:
            break
            
        passes_since_recompute += 1
        if passes_since_recompute % 10 == 0:
            A = _autoconv_fft(h)
            passes_since_recompute = 0
            
    return h, current_score

def _high_precision_polish(h, deadline=None):
    """Tighter local search with more passes and smaller tolerance."""
    h = h.copy()
    s = np.sum(h)
    A = _autoconv_fft(h)
    current_score = np.max(np.abs(A)) / (s * s)
    
    # Use very tight improvement threshold
    eps = 1e-12
    max_passes = 2000
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for k in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            h_k = h[k]
            A_new = A.copy()
            A_new[k:k+N] -= 4 * h_k * h
            if 2*k < 2*N - 1:
                A_new[2*k] += 4.0
            
            s_new = s - 2 * h_k
            
            if abs(s_new) < 1e-9:
                continue
                
            score_new = np.max(np.abs(A_new)) / (s_new * s_new)
            
            if score_new < current_score - eps:
                h[k] = -h_k
                A = A_new
                s = s_new
                current_score = score_new
                improved = True
        
        if not improved:
            break
    
    return h, current_score

def _smooth_obj_with_grad(x, beta):
    """Smooth surrogate objective with gradient using log-sum-exp."""
    ac = _autoconv_fft(x)
    s = np.sum(x)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(x)
    
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        w = np.zeros_like(ac)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        log_sum = np.log(np.sum(exp_scaled))
        S = m + log_sum / beta
        w = exp_scaled / np.sum(exp_scaled)
    
    v = w * np.sign(ac)
    fsize = 1
    while fsize < 2 * (2 * N - 1):
        fsize *= 2
    V = np.fft.rfft(v, fsize)
    H = np.fft.rfft(x, fsize)
    C = np.fft.irfft(V * np.conj(H), fsize)
    dS_dh = 2 * C[:N]
    
    grad = (dS_dh * s - 2 * S) / (s ** 3)
    return S / (s * s), grad

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_score = np.inf
    best_h = np.ones(N) / N

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    inits = []
    x = np.arange(N)
    inits.append(np.ones(N))
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    phase = np.pi * x**2 / N
    inits.append(np.cos(phase))
    inits.append(np.array([1.0 if i % 4 in [0, 2] else -1.0 for i in range(N)]))
    
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    for i in range(5):
        inits.append(rng.choice([1.0, -1.0], N))

    # Optimization loop using smooth surrogate
    for h_init in inits:
        if time.time() > deadline - 35:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            # Use more beta values for better coverage
            for beta in [10, 20, 50, 100]:
                if time.time() > deadline - 25:
                    break
                
                def obj(x):
                    return _smooth_obj_with_grad(x, beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', jac=True,
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                
                s_opt = np.sum(h_opt)
                if abs(s_opt) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                    
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = _score(h_bin)
                if val_bin < best_score:
                    best_score = val_bin
                    best_h = h_bin.copy()
                    
        except Exception:
            pass

    # Final local search refinement with block moves
    if time.time() < deadline - 15:
        h_work = best_h.copy()
        h_refined, val_refined = _fast_local_search(h_work, max_passes=400, deadline=deadline-10)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # High-precision polish pass
    if time.time() < deadline - 5:
        h_work = best_h.copy()
        h_polished, val_polished = _high_precision_polish(h_work, deadline=deadline-2)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END

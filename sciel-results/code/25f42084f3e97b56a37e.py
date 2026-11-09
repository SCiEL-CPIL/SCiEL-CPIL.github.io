"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _compute_ratio(h):
    """Compute R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 < 1e-12 or l2_sq < 1e-12:
        return 0.0
    # Autocorrelation via FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    ac_inf = np.max(np.abs(ac))
    if ac_inf < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_inf)

def _normalize_l2(h):
    """Normalize h to have L2 norm = 1 (scale invariance)."""
    n = np.linalg.norm(h)
    if n < 1e-12:
        return h
    return h / n

def _coordinate_descent(h, max_iters=2000, tol=1e-10, deadline=None):
    """Coordinate descent with 1D line search on each coordinate.
    Works on shape (L2=1 fixed), maximizing R."""
    h = _normalize_l2(h.copy())
    best_val = _compute_ratio(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try varying h[i] while keeping others fixed
            # After changing h[i], renormalize to L2=1
            old_hi = h[i]
            # Line search: try a few candidate values for h[i]
            # h[i] >= 0, and we want to explore around current value
            candidates = []
            if old_hi > 1e-6:
                for frac in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                    candidates.append(old_hi * frac)
            else:
                # If near zero, try small positive values
                norm_others = np.sqrt(max(0, 1.0 - old_hi**2))
                for frac in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    candidates.append(norm_others * frac)
            # Also try setting to 0
            candidates.append(0.0)
            
            best_candidate_val = best_val
            best_candidate_h = None
            
            for new_hi in candidates:
                if new_hi < 0:
                    continue
                h[i] = new_hi
                h_new = _normalize_l2(h)
                val = _compute_ratio(h_new)
                if val > best_candidate_val + 1e-12:
                    best_candidate_val = val
                    best_candidate_h = h_new.copy()
            
            if best_candidate_h is not None:
                h = best_candidate_h
                best_val = best_candidate_val
                improved = True
                if best_val - _compute_ratio(_normalize_l2(h)) > 1e-12:
                    # recompute to be safe
                    pass
        
        if not improved:
            break
    
    return h, best_val

def _gen_initializations():
    """Generate diverse initializations for the optimization."""
    inits = []
    
    # 1. Uniform (baseline, R=1)
    inits.append(np.ones(N))
    
    # 2. Raised cosine window
    x = np.linspace(0, np.pi, N)
    inits.append(0.5 * (1 + np.cos(x)))
    
    # 3. Sinc-like (low-pass)
    n = np.arange(N)
    center = (N - 1) / 2.0
    inits.append(np.abs(np.sinc((n - center) / (N / 4.0))))
    
    # 4. Gaussian window
    n = np.arange(N)
    center = (N - 1) / 2.0
    sigma = N / 4.0
    inits.append(np.exp(-0.5 * ((n - center) / sigma) ** 2))
    
    # 5. Two-peak (split energy)
    h = np.zeros(N)
    h[:N//2] = 1.0
    h[N//2:] = 0.5
    inits.append(h)
    
    # 6. Alternating pattern (high frequency)
    h = np.zeros(N)
    h[::2] = 1.0
    inits.append(h)
    
    # 7. Smooth sinusoidal modulation
    n = np.arange(N)
    inits.append(0.5 + 0.5 * np.cos(2 * np.pi * n / N))
    
    # 8. Triangle wave
    n = np.arange(N)
    inits.append(np.abs(1.0 - 2.0 * (n / N - 0.5)))
    
    # 9. Random smooth
    rng = np.random.default_rng(42)
    h = rng.random(N)
    # Smooth it
    kernel = np.ones(5) / 5
    h = np.convolve(h, kernel, mode='same')
    inits.append(h)
    
    # 10. Another random
    rng2 = np.random.default_rng(123)
    h = rng2.random(N)
    kernel = np.ones(3) / 3
    h = np.convolve(h, kernel, mode='same')
    inits.append(h)
    
    return inits

def construct():
    """Construct 50 nonnegative heights to maximize the autocorrelation ratio."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = 0.0
    
    inits = _gen_initializations()
    
    for init in inits:
        if time.time() > deadline - 5.0:
            break
        
        h_norm = _normalize_l2(init)
        val = _compute_ratio(h_norm)
        
        if val > best_val:
            best_val = val
            best_h = h_norm.copy()
        
        # Run coordinate descent from this initialization
        h_opt, val_opt = _coordinate_descent(
            h_norm, 
            max_iters=1500, 
            tol=1e-10,
            deadline=deadline - 3.0
        )
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # If we still have time, do a few more random restarts
    if best_h is None:
        best_h = np.ones(N)
    
    remaining = deadline - time.time()
    if remaining > 10.0:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            h_rand = rng.random(N)
            # Smooth
            kernel = np.ones(3) / 3
            h_rand = np.convolve(h_rand, kernel, mode='same')
            h_rand = np.maximum(h_rand, 0)
            
            h_opt, val_opt = _coordinate_descent(
                h_rand,
                max_iters=1000,
                tol=1e-10,
                deadline=deadline - 3.0
            )
            
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0.0)
    # Final normalization check
    best_h = _normalize_l2(best_h)
    
    return list(best_h)

# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy import optimize

N = 600

def _autoconv_peak(h):
    """Compute the peak of the autoconvolution of h, normalized by (sum h)^2."""
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Full autoconvolution via FFT for speed
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2*n-1]
    peak = np.max(ac)
    return peak / (s * s)

def _soft_autoconv_peak(x, T=0.005):
    """
    Smooth surrogate: soft-max of autoconvolution values, normalized.
    Uses FFT for the convolution.
    """
    h = np.maximum(x, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2*n-1]
    
    # Normalize ac by s^2 for the objective scaling
    ac_norm = ac / (s * s)
    
    # Soft max
    ac_max = np.max(ac_norm)
    if ac_max < 1e-15:
        return 0.0
    shifted = (ac_norm - ac_max) / T
    shifted = np.clip(shifted, -100, 100)
    soft_max = ac_max + T * np.log(np.sum(np.exp(shifted)))
    return soft_max

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _autoconv_peak(best_h)
    
    # Diverse initializations
    np.random.seed(42)
    initializations = []
    
    # 0: Uniform
    initializations.append(np.ones(N))
    
    # 1: Cosine perturbation
    t = np.linspace(0, 1, N, endpoint=False)
    initializations.append(np.maximum(1.0 + 0.2 * np.cos(2 * np.pi * t), 0.01))
    
    # 2: Sine perturbation
    initializations.append(np.maximum(1.0 + 0.2 * np.sin(2 * np.pi * t), 0.01))
    
    # 3: Two-frequency
    initializations.append(np.maximum(1.0 + 0.1 * np.cos(2 * np.pi * t) + 0.1 * np.cos(4 * np.pi * t), 0.01))
    
    # 4: Random small
    initializations.append(np.maximum(np.ones(N) + 0.1 * np.random.randn(N), 0.01))
    
    # 5: Edge enhanced
    initializations.append(np.maximum(1.0 + 0.3 * np.abs(np.linspace(-1, 1, N)), 0.01))
    
    # 6: Center enhanced
    t_c = np.linspace(-1, 1, N)
    initializations.append(np.maximum(1.0 + 0.3 * np.exp(-t_c**2 * 5), 0.01))
    
    # 7: Parabolic
    initializations.append(np.maximum(1.0 + 0.4 * (1 - t_c**2), 0.01))
    
    # 8: Random (different seed)
    np.random.seed(123)
    h_rand = np.random.rand(N) + 0.1
    h_rand = h_rand / np.mean(h_rand)
    initializations.append(h_rand)
    
    # 9: Higher frequency
    initializations.append(np.maximum(1.0 + 0.08 * np.cos(6 * np.pi * t) + 0.08 * np.cos(10 * np.pi * t), 0.01))
    
    # 10: Random (another seed)
    np.random.seed(456)
    h_rand2 = np.random.rand(N) + 0.05
    h_rand2 = h_rand2 / np.mean(h_rand2)
    initializations.append(h_rand2)
    
    # 11: Mixed
    initializations.append(np.maximum(1.0 + 0.1 * np.sin(4 * np.pi * t) + 0.05 * np.cos(8 * np.pi * t), 0.01))
    
    # Optimization loop over initializations
    # We allocate time budget per initialization
    num_inits = len(initializations)
    time_per_init = (deadline - start_time) / num_inits
    
    for i, h0 in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        h0 = np.asarray(h0, dtype=float)
        
        # Stage 1: Coarse optimization with higher T
        # Use L-BFGS-B on the soft objective
        try:
            # Set a tighter deadline for this stage
            stage1_deadline = time.time() + time_per_init * 0.6
            if time.time() > stage1_deadline:
                stage1_deadline = time.time() + 5.0
            
            res1 = optimize.minimize(
                lambda x, T=0.05: _soft_autoconv_peak(x, T),
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 50}
            )
            
            if time.time() > deadline - 10:
                break
            
            # Stage 2: Fine optimization with lower T
            if res1.fun < best_score * 1.05: # Only continue if promising
                stage2_deadline = time.time() + time_per_init * 0.4
                if time.time() > stage2_deadline:
                    stage2_deadline = time.time() + 5.0
                
                res2 = optimize.minimize(
                    lambda x, T=0.005: _soft_autoconv_peak(x, T),
                    res1.x,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-11, 'gtol': 1e-9, 'maxls': 50}
                )
                
                # Evaluate candidates
                candidates = [res1.x, res2.x]
                for cand in candidates:
                    h_cand = np.maximum(cand, 0.0)
                    score_cand = _autoconv_peak(h_cand)
                    if score_cand < best_score:
                        best_score = score_cand
                        best_h = h_cand.copy()
                        
        except Exception:
            pass

    # Final polish: Local search around the best solution
    if time.time() < deadline - 5:
        # Try a few random perturbations + L-BFGS polish
        np.random.seed(999)
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            h_pert = best_h + 0.005 * np.random.randn(N)
            h_pert = np.maximum(h_pert, 0)
            
            try:
                res = optimize.minimize(
                    lambda x, T=0.002: _soft_autoconv_peak(x, T),
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_new = np.maximum(res.x, 0)
                score_new = _autoconv_peak(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
            except Exception:
                pass

    return [float(x) for x in best_h]

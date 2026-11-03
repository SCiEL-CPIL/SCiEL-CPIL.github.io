import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _soft_max(v, tau=0.1):
    """Soft maximum of vector v."""
    v_max = np.max(v)
    # Prevent overflow
    if v_max > 700:
        return v_max
    return v_max + tau * np.log(np.sum(np.exp((v - v_max) / tau)))

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric search for good initializations
    # Uniform
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussians
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        # Add some noise
        h_rand += np.random.rand(N) * 0.1
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using SLSQP on log-scale variables with Soft-Max surrogate
    # We maximize Score(h) = sum(h^2) / (sum(h) * DELTA * SoftMax(conv(h,h)))
    # Since we normalize h to sum=1, we maximize sum(h^2) / SoftMax(conv(h,h))
    # Let h = exp(x) / sum(exp(x))
    
    def neg_obj_soft(x):
        h = np.exp(x)
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        # Use soft max to smooth the kink
        sm = _soft_max(h_conv, tau=0.01)
        if sm == 0:
            return 0.0
        return -(l2_sq / sm)

    # Constraints: h >= 0 is handled by exp. Sum h = 1 is handled by normalization in objective.
    # But SLSQP needs explicit constraints or we can just optimize x freely.
    # Since h = exp(x)/sum(exp(x)) is always valid, we can optimize x without constraints.
    
    # Start from best_h
    h_start = best_h.copy()
    h_start[h_start < 1e-8] = 1e-8
    x0 = np.log(h_start)
    
    # Optimization
    try:
        res = minimize(neg_obj_soft, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
        if res.success:
            h_candidate = np.exp(res.x)
            h_candidate /= np.sum(h_candidate)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    except Exception:
        pass

    # Try multiple restarts
    inits = [best_h.copy()]
    
    # Add more diverse inits
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    for h_init in inits:
        if time.time() > deadline: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        h_init[h_init < 1e-8] = 1e-8
        x_init = np.log(h_init)
        
        try:
            res = minimize(neg_obj_soft, x_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Final polish with tighter tolerance on the exact (non-soft) objective using SLSQP
    # This corrects any bias from the soft-max approximation
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        h_start[h_start < 1e-8] = 1e-8
        x0 = np.log(h_start)
        
        def neg_obj_exact(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0.0
            return -(l2_sq / max_conv)

        try:
            res = minimize(neg_obj_exact, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-END

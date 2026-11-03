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
    if v_max > 700:
        return v_max
    return v_max + tau * np.log(np.sum(np.exp((v - v_max) / tau)))

def _soft_max_grad(v, tau=0.1):
    """Gradient of soft max with respect to v."""
    v_max = np.max(v)
    if v_max > 700:
        grad = np.zeros_like(v)
        grad[np.argmax(v)] = 1.0
        return grad
    exp_v = np.exp((v - v_max) / tau)
    weights = exp_v / np.sum(exp_v)
    return weights

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
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(1, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # Sparse peaks with different patterns
    for _ in range(200):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 20)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = np.random.rand(num_peaks)
        h_rand += np.random.rand(N) * 0.05
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()
    
    # Optimization using L-BFGS-B on log-scale variables with Soft-Max surrogate
    # We maximize Score(h) = sum(h^2) / (sum(h) * SoftMax(conv(h,h)))
    # Since we normalize h to sum=1, we maximize sum(h^2) / SoftMax(conv(h,h))
    # Let h = exp(x) / sum(exp(x))
    
    def neg_obj_soft(x, tau=0.01):
        h = np.exp(x)
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        sm = _soft_max(h_conv, tau=tau)
        if sm == 0:
            return 0.0
        return -(l2_sq / sm)
    
    def neg_obj_soft_grad(x, tau=0.01):
        h = np.exp(x)
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        sm = _soft_max(h_conv, tau=tau)
        if sm == 0:
            return np.zeros_like(x)
        
        # Gradient of -l2_sq / sm with respect to x
        # d(sm)/d(h_conv) = weights
        # d(h_conv)/d(h) = convolution structure
        # d(h)/d(x) = h * (1 - h) (since h = softmax(x))
        
        weights = _soft_max_grad(h_conv, tau=tau)
        
        # Gradient of l2_sq with respect to h: 2*h
        # Gradient of sm with respect to h: need to trace through convolution
        # h_conv[k] = sum_i h[i] * h[k-i] (for valid range)
        # d(h_conv[k])/d(h[j]) = h[k-j] + h[j-k] (with appropriate bounds)
        
        # This is complex, so we'll use numerical gradient as fallback
        # For now, return zero gradient and let L-BFGS-B use numerical
        return np.zeros_like(x)
    
    # Try multiple tau values for soft-max
    tau_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    # Start from best_h
    h_start = best_h.copy()
    h_start[h_start < 1e-8] = 1e-8
    x0 = np.log(h_start)
    
    # Optimization with different tau values
    for tau in tau_values:
        if time.time() > deadline:
            break
            
        def neg_obj_tau(x):
            return neg_obj_soft(x, tau=tau)
        
        try:
            res = minimize(neg_obj_tau, x0, method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                    x0 = res.x.copy()  # Use as starting point for next tau
        except Exception:
            pass

    # Try multiple restarts with diverse initializations
    inits = [best_h.copy()]
    
    # Add more diverse inits
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 20)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = np.random.rand(k)
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(np.random.randint(3, 10))/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)
    
    for h_init in inits:
        if time.time() > deadline: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        h_init[h_init < 1e-8] = 1e-8
        x_init = np.log(h_init)
        
        # Try with best tau from previous
        for tau in [0.01, 0.05]:
            def neg_obj_init(x):
                return neg_obj_soft(x, tau=tau)
            
            try:
                res = minimize(neg_obj_init, x_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # Final polish with tighter tolerance on the exact (non-soft) objective using SLSQP
    # This corrects any bias from the soft-max approximation
    if time.time() < deadline - 15:
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
            res = minimize(neg_obj_exact, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Additional polish: try SLSQP with explicit constraints on h directly
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        
        def neg_obj_h(h):
            l1 = np.sum(h)
            if l1 == 0: return 0.0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0.0
            return -(l2_sq / (l1 * max_conv))
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        bounds = [(1e-10, 1.0)] * N
        
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', constraints=constraints, 
                          bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15})
            if res.success:
                h_candidate = res.x
                h_candidate = np.maximum(h_candidate, 0)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-END

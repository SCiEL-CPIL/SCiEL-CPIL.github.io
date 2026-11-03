import numpy as np
import time
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

def _smooth_max_conv(h, beta=50.0):
    """Smooth approximation of max(conv(h,h)) using log-sum-exp."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta

def _objective(x):
    """Minimize negative score (maximize score)."""
    h = np.exp(x)  # Ensure positivity
    s = _score_raw(h)
    if s <= 0:
        return 0.0
    return -s

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    # Parametric search for good initializations
    x_idx = np.arange(N)
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Sparse peaks (Sidon-like intuition)
    # Try to place peaks at positions that minimize autoconvolution overlap
    # Greedy Sidon set
    def find_sidon_set(n, k):
        # Greedy: pick smallest possible
        # Or just try random sparse sets
        pass
        
    # Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        # Add some noise
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using SLSQP with smooth max via log-sum-exp
    # Maximize L2^2 / max_conv subject to sum(h)=1, h>=0
    
    def neg_obj_smooth(h, beta=50.0):
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        smax = _smooth_max_conv(h, beta)
        if smax <= 0:
            return 0.0
        return -(l2_sq / smax)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N
    
    h_start = best_h / np.sum(best_h)
    
    # Try different beta values for smooth max
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline: break
        def neg_obj_beta(h):
            return neg_obj_smooth(h, beta)
        try:
            res = minimize(neg_obj_beta, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
                    h_start = h_candidate.copy()
        except Exception:
            pass
    
    # Generate diverse initializations
    inits = [best_h / np.sum(best_h), np.ones(N) / N]
    
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimize from each init with smooth objective
    for h_init in inits:
        if time.time() > deadline: break
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        for beta in [50.0, 100.0]:
            if time.time() > deadline: break
            def neg_obj_b(h):
                return neg_obj_smooth(h, beta)
            try:
                res = minimize(neg_obj_b, h_init, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
                if res.success:
                    h_candidate = res.x
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
            except Exception:
                pass

    # Final polish with exact (non-smooth) objective
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        
        def neg_obj_exact(h):
            l1 = np.sum(h)
            if l1 == 0: return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0
            return -(l2_sq / max_conv)
        
        try:
            res = minimize(neg_obj_exact, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
        
        if time.time() < deadline - 2:
            h_pert = best_h * (1 + 0.01 * np.random.randn(N))
            h_pert = np.maximum(0, h_pert)
            if np.sum(h_pert) > 0:
                h_pert /= np.sum(h_pert)
                try:
                    res = minimize(neg_obj_exact, h_pert, method='SLSQP', bounds=bounds,
                                  constraints=constraints, options={'maxiter': 500, 'ftol': 1e-14})
                    if res.success:
                        h_candidate = res.x
                        s_candidate = _score_raw(h_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_h = h_candidate
                except Exception:
                    pass

    return list(best_h)

# EVOLVE-BLOCK-END

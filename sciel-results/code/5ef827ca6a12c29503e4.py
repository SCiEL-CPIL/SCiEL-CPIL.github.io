import numpy as np
import time
import sys
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * Sup(AutoConv))"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    # Continuous autoconvolution of step function f(t) = sum h_i * 1_{[x_i, x_{i+1})}(t)
    # The peak is proportional to the max of the discrete convolution of h with itself.
    # Specifically, max(f*f) = DELTA * max(conv(h,h))
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max <= 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # We want to maximize: L2^2 / (L1 * Sup(AutoConv))
    # This is scale invariant if we normalize L1=1, then it becomes L2^2 / Sup(AutoConv)
    # But actually, let's work with the normalized version: assume L1=1
    # Then score = L2^2 / (1 * DELTA * max_conv) = L2^2 / (DELTA * max_conv)
    # So maximizing score is equivalent to maximizing L2^2 / max_conv (since DELTA is constant)
    
    # Strategy: 
    # 1. Try various structured initializations
    # 2. Use SLSQP optimization with L1=1 constraint
    # 3. Use random restarts and local search
    # 4. Try to exploit structure: sparse peaks, symmetric patterns, etc.
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussian peaks
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(1.0, N/3, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 0:
                h_g /= np.sum(h_g)
                inits.append(h_g)
    
    # 3. Triangular
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(1.0, N/2, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
    
    # 4. Sparse peaks (Sidon-like intuition: minimize autoconvolution overlap)
    # Try to place peaks at positions that minimize pairwise sum collisions
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 12)
        indices = np.random.choice(N, k, replace=False)
        h_s[indices] = 1.0
        # Add small noise to break symmetry
        h_s += np.random.rand(N) * 0.05
        h_s /= np.sum(h_s)
        inits.append(h_s)
    
    # 5. Alternating patterns
    for period in [2, 3, 4, 5, 7]:
        h_alt = np.zeros(N)
        h_alt[::period] = 1.0
        h_alt += np.random.rand(N) * 0.05
        h_alt /= np.sum(h_alt)
        inits.append(h_alt)
    
    # 6. Random smooth
    for _ in range(30):
        h_r = np.random.rand(N)
        # Smooth it
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)
    
    # 7. Cosine patterns
    for freq in np.linspace(0.1, 2.0, 10):
        h_cos = np.abs(np.cos(2 * np.pi * freq * x_idx / N))
        h_cos += np.random.rand(N) * 0.05
        h_cos /= np.sum(h_cos)
        inits.append(h_cos)
    
    # 8. Power-law decay from center
    for alpha in np.linspace(0.5, 3.0, 10):
        center = N / 2
        h_pow = 1.0 / (1.0 + np.abs(x_idx - center) ** alpha)
        h_pow += np.random.rand(N) * 0.05
        h_pow /= np.sum(h_pow)
        inits.append(h_pow)
    
    # Evaluate all initializations
    for h_init in inits:
        if time.time() > deadline:
            break
        h_init = np.asarray(h_init, dtype=np.float64)
        h_init = np.maximum(0, h_init)
        if np.sum(h_init) <= 0:
            continue
        h_init /= np.sum(h_init)
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
    
    # Optimization using SLSQP
    # Variables: h (non-negative, sum=1)
    # Objective: Maximize L2^2 / max_conv(h,h)
    # Equivalent to: Minimize -L2^2 / max_conv(h,h)
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 <= 0:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv <= 0:
            return 0.0
        return -(l2_sq / max_conv)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}  # h >= 0
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Optimize from best initialization
    h_start = best_h.copy()
    try:
        res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            h_candidate = np.maximum(0, res.x)
            if np.sum(h_candidate) > 0:
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
    except Exception:
        pass
    
    # Try optimization from multiple diverse initializations
    # Pick top 10 initializations by score
    scored_inits = []
    for h_init in inits:
        h_init = np.asarray(h_init, dtype=np.float64)
        h_init = np.maximum(0, h_init)
        if np.sum(h_init) <= 0:
            continue
        h_init /= np.sum(h_init)
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    for s_init, h_init in scored_inits[:10]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # Local search: perturb best solution and try to improve
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Random perturbation
            h_perturbed = best_h.copy()
            noise_scale = 0.05
            h_perturbed += np.random.normal(0, noise_scale, N)
            h_perturbed = np.maximum(0, h_perturbed)
            if np.sum(h_perturbed) <= 0:
                continue
            h_perturbed /= np.sum(h_perturbed)
            
            try:
                res = minimize(neg_obj_h, h_perturbed, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 100, 'ftol': 1e-10})
                if res.success:
                    h_candidate = np.maximum(0, res.x)
                    if np.sum(h_candidate) > 0:
                        h_candidate /= np.sum(h_candidate)
                        s_candidate = _score_raw(h_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_h = h_candidate
            except Exception:
                pass
    
    return list(best_h)

# EVOLVE-BLOCK-END

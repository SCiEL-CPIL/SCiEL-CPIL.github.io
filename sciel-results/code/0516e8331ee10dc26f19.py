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

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    
    x_idx = np.arange(N)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussians
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(1, N/2, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 3. Triangles
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(1, N, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 4. Sparse sets (Sidon-like)
    for k in range(5, 16):
        for seed in range(5):
            rng = np.random.RandomState(seed * 100 + k)
            idx = rng.choice(N, k, replace=False)
            h_s = np.zeros(N)
            h_s[idx] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s)
            
    # 5. Smooth random
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    best_h = inits[0].copy()
    best_score = _score_raw(best_h)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    def neg_obj_smooth(h, temp=0.05):
        """Smooth objective using log-sum-exp for soft max of autoconvolution."""
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        # Soft max via log-sum-exp
        shifted = h_conv / temp
        max_shifted = np.max(shifted)
        logsumexp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_max = temp * logsumexp
        if soft_max < 1e-15:
            return 0.0
        return -(l2_sq / soft_max)
    
    def neg_obj_exact(h):
        """Exact objective with hard max."""
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return -(l2_sq / max_conv)
    
    # Phase 1: Optimize with smooth objective from multiple inits
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        if np.sum(h_init) < 1e-15:
            continue
        h_init = h_init / np.sum(h_init)
        try:
            res = minimize(lambda h: neg_obj_smooth(h, 0.05), h_init, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with exact objective
    if time.time() < deadline - 15:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_exact, h_start, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
        
        # Perturbation restarts from best
        for _ in range(15):
            if time.time() > deadline - 5:
                break
            h_pert = best_h + np.random.randn(N) * 0.03
            h_pert = np.maximum(0, h_pert)
            if np.sum(h_pert) > 1e-15:
                h_pert /= np.sum(h_pert)
                try:
                    res = minimize(neg_obj_exact, h_pert, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'maxiter': 300, 'ftol': 1e-14})
                    if res.success:
                        h_candidate = res.x
                        s_candidate = _score_raw(h_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_h = h_candidate.copy()
                except Exception:
                    pass

    return list(best_h)

# EVOLVE-BLOCK-END

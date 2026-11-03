import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    deadline = 110.0
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Parametric Gaussian basis approach
    def gaussian_model(params, m):
        """params: [a1, c1, s1, a2, c2, s2, ..., am, cm, sm]"""
        h = np.zeros(N)
        for j in range(m):
            a = params[3*j]
            c = params[3*j+1]
            s = params[3*j+2]
            h += a * np.exp(-0.5 * ((x_idx - c) / s)**2)
        return np.maximum(h, 0)
    
    def neg_score_gaussian(params, m):
        h = gaussian_model(params, m)
        if np.sum(h) == 0:
            return 0.0
        return -_score_raw(h)
    
    # Try different numbers of Gaussians
    for m in [1, 2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 30:
            break
        
        # Bounds: a in [0, 10], c in [0, N-1], s in [0.5, N]
        bounds = [(0, 10), (0, N-1), (0.5, N)] * m
        
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random initial parameters
            params0 = np.array([
                np.random.uniform(0, 5) for _ in range(m)  # amplitudes
            ] + [
                np.random.uniform(0, N-1) for _ in range(m)  # centers
            ] + [
                np.random.uniform(1, N/2) for _ in range(m)  # widths
            ])
            
            try:
                res = minimize(
                    neg_score_gaussian, params0,
                    args=(m,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                h_candidate = gaussian_model(res.x, m)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    # Direct optimization on h with L1=1 constraint
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0
        return -(l2_sq / (l1 * DELTA * max_conv))
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Generate diverse initializations
    inits = []
    
    # Best so far
    if np.sum(best_h) > 0:
        inits.append(best_h / np.sum(best_h))
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # Sparse peaks with Sidon-like spacing
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 15)
        # Try to space peaks apart
        positions = np.random.choice(N, k, replace=False)
        positions.sort()
        # Check if well-spaced
        gaps = np.diff(positions)
        if np.min(gaps) >= 2:
            h_s[positions] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s)
    
    # Smooth random
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(7) / 7
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(h_r, 0)
        h_r /= np.sum(h_r)
        inits.append(h_r)
    
    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 5:
            break
        
        if np.sum(h_init) == 0:
            continue
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(
                neg_obj_h, h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Differential evolution for global search
    if time.time() < deadline - 20:
        def de_obj(h):
            return -_score_raw(h)
        
        try:
            res = differential_evolution(
                de_obj,
                bounds=[(0, 1)] * N,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False
            )
            h_candidate = res.x
            # Normalize to L1=1
            if np.sum(h_candidate) > 0:
                h_candidate = h_candidate / np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        if np.sum(best_h) > 0:
            h_start = best_h / np.sum(best_h)
            try:
                res = minimize(
                    neg_obj_h, h_start,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                if res.success:
                    h_candidate = res.x
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
            except Exception:
                pass
    
    return list(best_h)

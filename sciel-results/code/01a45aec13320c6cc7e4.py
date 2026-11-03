import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """
    Computes the raw score for a sequence of heights h.
    Score = L2_sq / (L1 * Delta * max_conv)
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    # Compute discrete autoconvolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    
    if max_conv <= 1e-15:
        return 0.0
        
    return l2_sq / (l1 * DELTA * max_conv)

def _objective_gaussian(params):
    """
    Objective function for optimization using Gaussian mixture.
    params = [A1, C1, S1, A2, C2, S2, ...]
    """
    M = len(params) // 3
    A = params[0:M]
    C = params[M:2*M]
    S = params[2*M:3*M]
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA/2
    
    h = np.zeros(N)
    for i in range(M):
        sigma = abs(S[i]) + 1e-6
        h += A[i] * np.exp(-0.5 * ((x - C[i]) / sigma) ** 2)
        
    h = np.clip(h, 0, None)
    
    if np.sum(h) < 1e-15:
        return 0.0
        
    h_norm = h / np.sum(h)
    s = _score_raw(h_norm)
    return -s

def _objective_direct(h):
    """
    Objective function for direct optimization of heights.
    h is the vector of heights.
    """
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-15:
        return 0.0
    h_norm = h / np.sum(h)
    s = _score_raw(h_norm)
    return -s

def _bounds_gaussian(M):
    bounds = []
    for i in range(M):
        bounds.append((0.0, 10.0))   # A
    for i in range(M):
        bounds.append((-0.25, 0.25)) # C
    for i in range(M):
        bounds.append((0.005, 0.25)) # S
    return bounds

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Initial best: Uniform
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # 1. Global Search with Differential Evolution on Gaussian Mixture
    M = 8 # Increased number of Gaussians for more complex shapes
    param_bounds = _bounds_gaussian(M)
    
    try:
        if time.time() < deadline - 60:
            de_res = differential_evolution(
                _objective_gaussian, 
                param_bounds, 
                seed=42, 
                maxiter=200, 
                popsize=20,
                tol=0.005,
                update_interval=10,
                polish=False # We will polish later
            )
            
            if de_res.success:
                # Convert best params back to h
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA/2
                A = de_res.x[0:M]
                C = de_res.x[M:2*M]
                S = de_res.x[2*M:3*M]
                
                h = np.zeros(N)
                for i in range(M):
                    sigma = abs(S[i]) + 1e-6
                    h += A[i] * np.exp(-0.5 * ((x - C[i]) / sigma) ** 2)
                h = np.clip(h, 0, None)
                
                if np.sum(h) > 1e-15:
                    h_norm = h / np.sum(h)
                    s = _score_raw(h_norm)
                    if s > best_score:
                        best_score = s
                        best_h = h_norm
    except Exception:
        pass

    # 2. Local Refinement using L-BFGS-B on Direct Heights
    # This allows the solution to escape the Gaussian manifold
    try:
        if time.time() < deadline - 10:
            # Start from the best Gaussian mixture result
            # Also try a few random starts
            starts = [best_h.copy()]
            
            # Add random sparse-like starts
            for _ in range(3):
                h_rand = np.random.exponential(1.0, N)
                if np.sum(h_rand) > 0:
                    starts.append(h_rand / np.sum(h_rand))
            
            # Add a "comb" start
            h_comb = np.zeros(N)
            h_comb[::2] = 1.0
            if np.sum(h_comb) > 0:
                starts.append(h_comb / np.sum(h_comb))
            
            for p_start in starts:
                if time.time() > deadline - 5:
                    break
                
                res = minimize(
                    _objective_direct, 
                    p_start, 
                    method='L-BFGS-B', 
                    bounds=[(0.0, 1.0)] * N, # Heights are non-negative, normalized later
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                
                if res.success:
                    h_opt = np.clip(res.x, 0, None)
                    if np.sum(h_opt) > 1e-15:
                        h_norm = h_opt / np.sum(h_opt)
                        s = _score_raw(h_norm)
                        if s > best_score:
                            best_score = s
                            best_h = h_norm
                            
    except Exception:
        pass

    return list(best_h)

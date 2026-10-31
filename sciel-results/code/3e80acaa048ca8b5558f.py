import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _calculate_ratio(h):
    """
    Calculate the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    h is a numpy array of nonnegative heights.
    """
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
        
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    # np.convolve with mode='full' gives length 2N-1
    conv = np.convolve(h, h, mode='full')
    
    max_conv = np.max(conv)
    
    if max_conv == 0:
        return 0.0
        
    return l2_sq / (l1 * max_conv)

def _objective_smooth(h):
    """
    Smooth objective function to maximize the ratio.
    Minimizes -log(ratio) = -log(sum(h^2)) + log(sum(h)) + log(max(conv))
    Using soft-max for the max term.
    """
    l1 = np.sum(h)
    if l1 < 1e-10:
        return 1e10
        
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-10:
        return 1e10
        
    conv = np.convolve(h, h, mode='full')
    
    # Soft max approximation
    # Temperature tau
    tau = 0.01 
    soft_max = tau * np.log(np.sum(np.exp(conv / tau)))
    
    return -np.log(l2_sq) + np.log(l1) + soft_max

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 0.0
    best_h = None
    
    # Candidate 1: Spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    score_spike = _calculate_ratio(h_spike)
    if score_spike > best_score:
        best_score = score_spike
        best_h = h_spike.copy()
        
    # Candidate 2: Uniform
    h_uniform = np.ones(N)
    score_uniform = _calculate_ratio(h_uniform)
    if score_uniform > best_score:
        best_score = score_uniform
        best_h = h_uniform.copy()
        
    # Candidate 3: Sidon Set (Mian-Chowla)
    # This structure minimizes the autoconvolution peak for a given support size.
    h_sidon = np.zeros(N)
    sidon_indices = [0, 1, 3, 7, 12, 20, 31, 45]
    for idx in sidon_indices:
        h_sidon[idx] = 1.0
        
    score_sidon = _calculate_ratio(h_sidon)
    if score_sidon > best_score:
        best_score = score_sidon
        best_h = h_sidon.copy()
        
    # Candidate 4: Local Optimization from Sidon and Random Sparse Starts
    np.random.seed(42)
    
    # List of starting points for local optimization
    starts = [best_h.copy()]
    
    # Add a few random sparse binary vectors
    for i in range(5):
        if time.time() > deadline:
            break
        h_rand = np.zeros(N)
        # Pick 8 random indices
        idxs = np.random.choice(N, 8, replace=False)
        h_rand[idxs] = 1.0
        starts.append(h_rand)
        
    bounds = [(0, None)] * N
    
    for h_init in starts:
        if time.time() > deadline:
            break
            
        # Normalize to sum 1 for numerical stability
        s = np.sum(h_init)
        if s > 0:
            h_init = h_init / s
            
        try:
            res = minimize(
                _objective_smooth, 
                h_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            # Ensure non-negative
            h_opt = np.maximum(h_opt, 0)
            
            score_opt = _calculate_ratio(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            continue
            
    # Final Polish
    if best_h is not None and time.time() < deadline - 10:
        h_final = best_h.copy()
        s = np.sum(h_final)
        if s > 0:
            h_final = h_final / s
            
        try:
            res = minimize(
                _objective_smooth, 
                h_final, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = _calculate_ratio(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    if best_h is None:
        return [1.0] + [0.0] * (N - 1)
        
    return list(best_h)

import numpy as np
from scipy.optimize import minimize
import time

def _score_from_heights(h):
    # h is a numpy array of nonnegative heights
    dt = 0.02
    l1 = np.sum(h) * dt
    if l1 == 0:
        return 0.0
    
    h_norm = h / np.sum(h) if np.sum(h) > 0 else h
    
    l2_sq = np.sum(h_norm ** 2) * dt
    l1_norm = np.sum(h_norm) * dt
    
    ac = np.convolve(h_norm, h_norm, mode='full')
    ac_vals = ac * dt
    peak = np.max(ac_vals)
    
    if peak == 0:
        return 0.0
        
    return l2_sq / (l1_norm * peak)

def _smooth_score_from_heights(h, beta=10.0):
    # Smooth surrogate for optimization
    dt = 0.02
    sum_h = np.sum(h)
    if sum_h == 0:
        return 0.0
    h_norm = h / sum_h
    
    l2_sq = np.sum(h_norm ** 2) * dt
    l1_norm = np.sum(h_norm) * dt
    
    ac = np.convolve(h_norm, h_norm, mode='full')
    ac_vals = ac * dt
    
    # Soft max
    # Shift to avoid overflow
    ac_max = np.max(ac_vals)
    log_sum_exp = np.log(np.sum(np.exp(beta * (ac_vals - ac_max)))) + beta * ac_max
    peak_soft = log_sum_exp / beta
    
    if peak_soft == 0:
        return 0.0
        
    return l2_sq / (l1_norm * peak_soft)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 50
    best_score = -np.inf
    best_thetas = np.zeros(N) # Start with spike (flat phase)
    
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            thetas = np.zeros(N)
        else:
            thetas = np.random.uniform(0, 2*np.pi, N)
            
        def objective(thetas_flat):
            z = np.exp(1j * thetas_flat)
            h = np.abs(np.fft.ifft(z))
            return -_smooth_score_from_heights(h, beta=20.0)
        
        res = minimize(objective, thetas, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 5000})
        
        # Evaluate with exact score
        z = np.exp(1j * res.x)
        h = np.abs(np.fft.ifft(z))
        exact_score = _score_from_heights(h)
        
        if exact_score > best_score:
            best_score = exact_score
            best_thetas = res.x
            
        if time.time() > deadline:
            break
            
    z = np.exp(1j * best_thetas)
    h = np.abs(np.fft.ifft(z))
    
    sum_h = np.sum(h)
    if sum_h > 0:
        h = h / sum_h
        
    return list(h)

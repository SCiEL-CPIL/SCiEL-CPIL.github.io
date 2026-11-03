import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 135

def compute_score(f):
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def objective_g(g, beta=10.0):
    f = np.exp(g)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    f_norm = f / s
    conv = np.convolve(f_norm, f_norm, mode='full')
    return smooth_max(conv, beta)

def local_refinement(f, start_time, max_time):
    best_f = f.copy()
    best_score = compute_score(f)
    current_f = f.copy()
    alpha = 0.01
    
    while time.time() - start_time < max_time:
        conv = np.convolve(current_f, current_f, mode='full')
        max_c = np.max(conv)
        current_score = max_c / (np.sum(current_f) ** 2)
        
        if current_score < best_score:
            best_score = current_score
            best_f = current_f.copy()
            
        threshold = max_c * 0.9
        peak_mask = conv >= threshold
        
        grad_f = np.convolve(peak_mask.astype(float), current_f, mode='same')
        
        new_f = current_f - alpha * grad_f
        new_f = np.maximum(new_f, 0)
        new_f = new_f / np.sum(new_f)
        
        new_score = compute_score(new_f)
        
        if new_score < current_score:
            current_f = new_f
            alpha *= 1.1
        else:
            alpha *= 0.8
            if alpha < 1e-6:
                break
                
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial guess: Gaussian
    x = np.linspace(-1, 1, N)
    f0 = np.exp(-5 * x**2)
    f0 = f0 / np.sum(f0)
    g0 = np.log(f0 + 1e-10)
    
    best_score = compute_score(f0)
    best_f = f0.copy()
    
    betas = [5.0, 10.0, 20.0]
    
    for beta in betas:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        try:
            result = minimize(
                objective_g, g0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < np.inf:
                f_opt = np.exp(result.x)
                f_opt = f_opt / np.sum(f_opt)
                score_opt = compute_score(f_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_f = f_opt.copy()
                    g0 = result.x
        except Exception:
            pass
            
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_f, refined_score = local_refinement(best_f, start_time, TIME_LIMIT - 2)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
            
    return [float(x) for x in best_f]

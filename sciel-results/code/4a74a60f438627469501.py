import numpy as np
import time
from scipy.optimize import minimize

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    def entropy_from_values(values, weights):
        min_v = np.min(values)
        shifted = values - min_v
        counts = np.bincount(shifted, weights=weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    d_vals = x_flat - y_flat
    D = entropy_from_values(d_vals, w)
    
    h0_vals = x_flat
    h1_vals = x_flat + y_flat
    h2_vals = x_flat + 2 * y_flat
    
    H0 = entropy_from_values(h0_vals, w)
    H1 = entropy_from_values(h1_vals, w)
    H2 = entropy_from_values(h2_vals, w)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _compute_ratio_smooth(P):
    P = np.clip(P, 1e-15, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    def entropy_from_values(values, weights):
        min_v = np.min(values)
        shifted = values - min_v
        counts = np.bincount(shifted, weights=weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    d_vals = x_flat - y_flat
    D = entropy_from_values(d_vals, w)
    
    h0_vals = x_flat
    h1_vals = x_flat + y_flat
    h2_vals = x_flat + 2 * y_flat
    
    H0 = entropy_from_values(h0_vals, w)
    H1 = entropy_from_values(h1_vals, w)
    H2 = entropy_from_values(h2_vals, w)
    
    # Use smooth max via log-sum-exp
    log_H = np.array([np.log(H0 + 1e-15), np.log(H1 + 1e-15), np.log(H2 + 1e-15)])
    log_max = np.max(log_H)
    M_smooth = np.exp(log_max)
    
    if M_smooth <= 1e-9:
        return 0.0
        
    return D / M_smooth

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    start_time = time.time()
    timeout = 600
    safety_margin = 10
    
    best_score = _compute_ratio(P)
    best_P = P.copy()
    
    np.random.seed(42)
    
    # Multi-start local search
    num_restarts = 50
    time_per_restart = (timeout - safety_margin) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Initialize current P
        if r == 0:
            current_P = P.copy()
        else:
            # Perturb best found or uniform
            current_P = best_P + np.random.normal(0, 0.1, (n, n))
            current_P = np.clip(current_P, 0, None)
            s = current_P.sum()
            if s > 0:
                current_P = current_P / s
            else:
                current_P = P.copy()
                
        current_score = _compute_ratio(current_P)
        
        # Hill climbing
        step_size = 0.2
        local_time_limit = time_per_restart * 0.7  # Use 70% for hill climbing, 30% for L-BFGS-B
        local_start = time.time()
        
        while time.time() - local_start < local_time_limit:
            i, j = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_size, step_size)
            
            new_P = current_P.copy()
            new_P[i, j] += delta
            
            if np.any(new_P < -1e-9):
                continue
                
            new_P = np.clip(new_P, 0, None)
            s = new_P.sum()
            if s <= 1e-9:
                continue
            new_P = new_P / s
            
            new_score = _compute_ratio(new_P)
            
            if new_score > current_score:
                current_P = new_P
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_P = current_P.copy()
                step_size *= 0.99
            else:
                pass
        
        # L-BFGS-B optimization on log-softmax parameterization
        remaining_time = timeout - safety_margin - (time.time() - start_time)
        if remaining_time > 5:
            # Parameterize P as exp(z) / sum(exp(z))
            z = np.log(current_P.flatten() + 1e-15)
            
            def objective(z_flat):
                if time.time() - start_time > timeout - safety_margin:
                    return 0.0
                z_reshaped = z_flat.reshape((n, n))
                P_candidate = np.exp(z_reshaped)
                P_candidate = P_candidate / P_candidate.sum()
                return -_compute_ratio_smooth(P_candidate)
            
            try:
                result = minimize(
                    objective,
                    z,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-6}
                )
                if result.success or result.fun < 0:
                    z_opt = result.x
                    P_opt = np.exp(z_opt.reshape((n, n)))
                    P_opt = P_opt / P_opt.sum()
                    opt_score = _compute_ratio(P_opt)
                    if opt_score > current_score:
                        current_P = P_opt
                        current_score = opt_score
                        if opt_score > best_score:
                            best_score = opt_score
                            best_P = current_P.copy()
            except Exception:
                pass
                
    return best_P

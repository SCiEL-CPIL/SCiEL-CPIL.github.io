import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial candidates
    h_uniform = np.ones(N)
    h_hann = np.hanning(N)
    h_hann[h_hann == 0] = 1e-6 # Avoid zero sum issues if needed, but hann sum is > 0
    
    candidates = [h_uniform, h_hann]
    
    # Add a few random perturbations of uniform
    np.random.seed(42)
    for _ in range(5):
        noise = np.random.normal(0, 0.1, N)
        h_rand = h_uniform + noise
        candidates.append(h_rand)
        
    best_h = None
    best_score = float('inf')
    
    def score(h):
        # Scale invariant, but for numerical stability in optimization, we might fix sum
        # Here we compute the raw ratio
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        c = np.convolve(h, h, mode='full')
        max_c = np.max(np.abs(c))
        return max_c / (s * s)
    
    # Evaluate initial candidates
    for h in candidates:
        s_val = score(h)
        if s_val < best_score:
            best_score = s_val
            best_h = h.copy()
            
    # Optimization using smooth surrogate
    # We fix sum(h) = 1 to remove scale invariance
    # Objective: soft_max(|conv(h, h)|)
    
    def soft_max_abs_conv(h):
        c = np.convolve(h, h, mode='full')
        beta = 50.0
        # Use log-sum-exp for stability
        # max_c = np.max(c)
        # shifted = c - max_c
        # But we need abs(c).
        # abs(c) is not smooth at 0, but log-sum-exp of exp(beta*|c|) works.
        # To avoid overflow, subtract max(abs(c))
        abs_c = np.abs(c)
        max_abs_c = np.max(abs_c)
        if max_abs_c == 0:
            return 0.0
        shifted = abs_c - max_abs_c
        val = max_abs_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        return val

    def objective_wrapper(h):
        # h is the variable
        # We need to enforce sum(h) = 1?
        # L-BFGS-B doesn't handle equality constraints well without 'SLSQP' or 'Trust-Constr'.
        # Let's use SLSQP.
        return soft_max_abs_conv(h)

    def constraint_sum(h):
        return np.sum(h) - 1.0

    # Try optimizing the best candidate
    # Normalize best_h to sum=1
    h_opt = best_h / np.sum(best_h)
    
    try:
        res = minimize(
            objective_wrapper, 
            h_opt, 
            method='SLSQP', 
            constraints=[{'type': 'eq', 'fun': constraint_sum}],
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        
        if time.time() < deadline:
            h_res = res.x
            s_res = score(h_res)
            if s_res < best_score:
                best_score = s_res
                best_h = h_res.copy()
                
    except Exception as e:
        pass
        
    # If time remains, try a few more random restarts with optimization
    while time.time() < deadline - 5.0:
        h_rand = np.random.uniform(0.5, 1.5, N) # Positive to keep sum large
        h_rand = h_rand / np.sum(h_rand)
        
        try:
            res = minimize(
                objective_wrapper, 
                h_rand, 
                method='SLSQP', 
                constraints=[{'type': 'eq', 'fun': constraint_sum}],
                options={'maxiter': 50, 'ftol': 1e-6, 'disp': False}
            )
            h_res = res.x
            s_res = score(h_res)
            if s_res < best_score:
                best_score = s_res
                best_h = h_res.copy()
        except:
            pass
            
    return best_h

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _exact_c3(h):
    """Compute exact C3 upper bound: max|conv| / (sum|h|)^2"""
    conv = np.convolve(h, h, mode='full')
    max_abs = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    if sum_abs < 1e-12:
        return np.inf
    return max_abs / (sum_abs ** 2)


def _discrete_local_search(h, max_rounds=200, time_limit=None):
    """
    Local search by flipping individual signs to reduce C3.
    """
    h = h.copy()
    current_c3 = _exact_c3(h)
    
    for round_idx in range(max_rounds):
        if time_limit is not None and time.time() > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Try flipping sign of h[i]
            h[i] = -h[i]
            new_c3 = _exact_c3(h)
            if new_c3 < current_c3:
                current_c3 = new_c3
                improved = True
            else:
                h[i] = -h[i]  # revert
        
        if not improved:
            break
    
    return h, current_c3


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    
    The metric is: upper_bound = max_k |convolve(h, h)[k]| / (sum_i |h_i|)^2
    
    We minimize this by optimizing the continuous variable x where h = tanh(x),
    then refining with discrete sign-flipping local search.
    """
    def objective(x):
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        abs_conv = np.abs(conv)
        
        # Use exact max for more accurate gradient signal
        max_abs = np.max(abs_conv)
        
        sum_abs_h = np.sum(np.abs(h))
        if sum_abs_h < 1e-12:
            return 1e6
        value = max_abs / (sum_abs_h ** 2)
        return value
    
    # Deadline guard: 140s out of 150s
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_val = np.inf
    best_x = None
    best_h = None
    
    # Multiple structured initializations
    initializations = []
    
    # 1. All positive (large x)
    initializations.append(np.full(N, 5.0))
    
    # 2. Alternating signs
    x_alt = np.full(N, 3.0)
    x_alt[::2] = -3.0
    initializations.append(x_alt)
    
    # 3. Random with moderate magnitude
    initializations.append(np.random.normal(0, 2, N))
    
    # 4. Block pattern: + - + - in blocks of 10
    x_block = np.zeros(N)
    for block in range(N // 10):
        sign = 3.0 if block % 2 == 0 else -3.0
        x_block[block*10:(block+1)*10] = sign
    initializations.append(x_block)
    
    # 5. Random sparse large values
    x_sparse = np.zeros(N)
    idx = np.random.choice(N, size=N//4, replace=False)
    x_sparse[idx] = np.random.choice([-4, 4], size=N//4)
    initializations.append(x_sparse)
    
    # Optimize from each initialization
    for init_idx, x0 in enumerate(initializations):
        if time.time() > deadline:
            break
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7})
        
        h_candidate = np.tanh(res.x)
        c3_candidate = _exact_c3(h_candidate)
        
        if c3_candidate < best_val:
            best_val = c3_candidate
            best_x = res.x.copy()
            best_h = h_candidate.copy()
    
    # If we have time, do random restarts from best solution
    if best_x is not None and time.time() < deadline - 30:
        for i in range(8):
            if time.time() > deadline - 10:
                break
            
            # Perturb best solution
            x0 = best_x + np.random.normal(0, 1.0, N)
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-7})
            
            h_candidate = np.tanh(res.x)
            c3_candidate = _exact_c3(h_candidate)
            
            if c3_candidate < best_val:
                best_val = c3_candidate
                best_x = res.x.copy()
                best_h = h_candidate.copy()
    
    # Final polish with tighter tolerance if time permits
    if best_x is not None and time.time() < deadline - 15:
        res = minimize(objective, best_x, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-9})
        h_candidate = np.tanh(res.x)
        c3_candidate = _exact_c3(h_candidate)
        if c3_candidate < best_val:
            best_val = c3_candidate
            best_x = res.x.copy()
            best_h = h_candidate.copy()
    
    # Discrete local search to refine signs
    if best_h is not None and time.time() < deadline - 5:
        h_refined, c3_refined = _discrete_local_search(
            best_h, max_rounds=100, time_limit=deadline - 3
        )
        if c3_refined < best_val:
            best_val = c3_refined
            best_h = h_refined
    
    return best_h


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END

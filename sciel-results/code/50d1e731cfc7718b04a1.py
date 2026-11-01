import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length N (lags 0 to N-1).
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac[:N]

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes (lags 1 to N-1).
    """
    if len(ac) <= 1:
        return 0.0
    return np.max(np.abs(ac[1:]))

def _local_search_fast(h, max_iterations=100000, time_limit=5.0):
    """
    Fast local search using O(N) incremental updates.
    """
    start_time = time.time()
    h = h.copy()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    delta_ac = np.zeros(N)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        new_h_i = -old_h_i
        
        # Calculate change in autocorrelation
        delta_ac[:] = 0.0
        
        max_k_plus = N - 1 - i
        if max_k_plus >= 1:
            delta_ac[1 : max_k_plus+1] += -2 * old_h_i * h[i+1 : i+max_k_plus+1]
            
        max_k_minus = i
        if max_k_minus >= 1:
            delta_ac[1 : max_k_minus+1] += -2 * old_h_i * h[i-1 : i-max_k_minus-1 : -1]

        ac_new = ac + delta_ac
        new_val = np.max(np.abs(ac_new[1:]))
        
        if new_val < current_val:
            h[i] = new_h_i
            ac = ac_new
            current_val = new_val
            
    return h, current_val

def _simulated_annealing_fast(h, start_temp=0.5, end_temp=0.01, iterations=20000, time_limit=10.0):
    """
    Simulated annealing with O(N) incremental updates.
    """
    start_time = time.time()
    h = h.copy()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_h = h.copy()
    best_val = current_val
    
    delta_ac = np.zeros(N)
    
    for i_iter in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        progress = i_iter / iterations
        temp = start_temp * (end_temp / start_temp) ** progress
        
        idx = np.random.randint(N)
        old_h_i = h[idx]
        new_h_i = -old_h_i
        
        delta_ac[:] = 0.0
        
        max_k_plus = N - 1 - idx
        if max_k_plus >= 1:
            delta_ac[1 : max_k_plus+1] += -2 * old_h_i * h[idx+1 : idx+max_k_plus+1]
            
        max_k_minus = idx
        if max_k_minus >= 1:
            delta_ac[1 : max_k_minus+1] += -2 * old_h_i * h[idx-1 : idx-max_k_minus-1 : -1]

        ac_new = ac + delta_ac
        new_val = np.max(np.abs(ac_new[1:]))
        
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            h[idx] = new_h_i
            ac = ac_new
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
            
    return best_h, best_val

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe.
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k == 0:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation (Limited time)
    # We will try 2-3 quick continuous optimizations
    # If this takes too long, we skip to discrete
    num_continuous_starts = 3
    betas = [50.0, 100.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 80.0:  # Reserve 80s for discrete search
            break
            
        x0 = np.random.choice([-1.0, 1.0], size=N)
        beta = betas[i % len(betas)]
        
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
        except Exception:
            continue
        
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
    # If no good start from continuous, use random
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
        best_val = _max_sidelobe(_compute_autocorr(best_h))
        
    # Phase 2: Discrete Search Loop
    # We will run multiple rounds of (Local Search -> SA -> Local Search)
    # until time runs out.
    
    while time.time() < deadline - 5.0:
        # 1. Local Search
        time_for_ls = min(5.0, (deadline - time.time()) / 3.0)
        if time_for_ls < 1.0: break
        h_ls, val_ls = _local_search_fast(best_h, max_iterations=10000, time_limit=time_for_ls)
        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls
            
        # 2. Simulated Annealing
        time_for_sa = min(10.0, (deadline - time.time()) / 3.0)
        if time_for_sa < 1.0: break
        h_sa, val_sa = _simulated_annealing_fast(best_h, start_temp=0.3, end_temp=0.005, 
                                                 iterations=10000, time_limit=time_for_sa)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa
             
        # 3. Final Local Search Polish
        time_for_polish = min(5.0, (deadline - time.time()) - 2.0)
        if time_for_polish < 1.0: break
        h_polish, val_polish = _local_search_fast(best_h, max_iterations=10000, time_limit=time_for_polish)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
            
    # Final check
    if best_h is None:
        return np.ones(N)
        
    return best_h

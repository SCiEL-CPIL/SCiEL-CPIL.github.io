"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective_hard(h):
    """Compute the hard objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # The task says: abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta):
    """
    Soft-max approximation of the objective.
    Minimizes 2*N*soft_max(conv) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    # Numerically stable log-sum-exp
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _param_to_h(theta, K):
    """
    Convert parameter vector theta to height vector h.
    theta = [c, a_1, ..., a_K, p_1, ..., p_K]
    c: background level
    a_k: amplitude of spike k
    p_k: position of spike k (integer)
    """
    c = theta[0]
    h = np.full(N, c)
    # Spikes
    for k in range(K):
        a_k = theta[1 + k]
        p_k = int(round(theta[1 + K + k]))
        # Clamp position to valid range
        p_k = np.clip(p_k, 0, N - 1)
        h[p_k] += a_k
    return h

def _sparse_obj(theta, beta, K):
    """
    Objective function for the sparse parameterization.
    """
    h = _param_to_h(theta, K)
    return _soft_max_obj(h, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Number of spikes to try
    K = 15
    
    # Generate diverse initializations for the sparse parameters
    # theta = [c, a_1...a_K, p_1...p_K]
    initializations = []
    
    # 1. Random spikes
    for _ in range(20):
        c = rng.uniform(-0.5, 0.5)
        amplitudes = rng.standard_normal(K) * 2.0
        # Random distinct positions
        positions = rng.choice(N, size=K, replace=False)
        theta0 = np.concatenate(([c], amplitudes, positions))
        initializations.append(theta0)
        
    # 2. Symmetric-ish spikes (pairs)
    for _ in range(10):
        c = rng.uniform(-0.2, 0.2)
        amplitudes = np.zeros(K)
        positions = np.zeros(K, dtype=int)
        # Create pairs
        num_pairs = K // 2
        for i in range(num_pairs):
            pos = rng.integers(0, N)
            amp = rng.uniform(1.0, 5.0)
            pos1 = pos
            pos2 = (pos + rng.integers(10, 100)) % N
            idx1 = i * 2
            idx2 = i * 2 + 1
            amplitudes[idx1] = amp
            amplitudes[idx2] = -amp # Anti-symmetric to cancel DC? No, keep sum non-zero.
            # Actually, let's just use random signs
            amplitudes[idx2] = amp * rng.choice([-1, 1])
            positions[idx1] = pos1
            positions[idx2] = pos2
        theta0 = np.concatenate(([c], amplitudes, positions))
        initializations.append(theta0)

    # 3. Based on seed (ones)
    # h = ones. c=1, amplitudes=0, positions=0
    theta_seed = np.concatenate(([1.0], np.zeros(K), np.zeros(K, dtype=int)))
    initializations.insert(0, theta_seed)

    # Optimization using scipy's minimize with BFGS on the sparse parameters
    betas = [10.0, 50.0, 200.0, 500.0]
    
    for init_idx, theta0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        theta_curr = theta0.copy()
        
        for beta in betas:
            if time.time() > deadline - 30.0:
                break
            
            def obj_func(theta):
                return _sparse_obj(theta, beta, K)
            
            try:
                # L-BFGS-B is good for this
                res = minimize(obj_func, theta_curr, 
                              method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                theta_new = res.x
            except Exception:
                theta_new = theta_curr
            
            h_new = _param_to_h(theta_new, K)
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                theta_curr = theta_new.copy()
            else:
                # Keep theta_curr if it's better than the new one in soft objective? 
                # Or just stick to the best hard value found so far.
                # Let's update theta_curr to the new one if the soft objective improved, 
                # to allow further refinement in the same direction.
                val_curr_soft = _soft_max_obj(_param_to_h(theta_curr, K), beta)
                val_new_soft = _soft_max_obj(h_new, beta)
                if val_new_soft < val_curr_soft:
                    theta_curr = theta_new.copy()

    # Final high-precision polish on the best found h (full vector)
    # This allows the background and non-spike elements to adjust to the fixed spikes
    if time.time() < deadline - 15.0:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 10.0:
                break
            
            def obj_func_full(h):
                return _soft_max_obj(h, beta)
            
            try:
                res = minimize(obj_func_full, best_h,
                              method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                h_polished = res.x
                val_polished = _objective_hard(h_polished)
                if val_polished < best_val:
                    best_val = val_polished
                    best_h = h_polished.copy()
            except Exception:
                pass

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END

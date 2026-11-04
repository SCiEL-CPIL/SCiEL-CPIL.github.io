import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=100.0):
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Soft-max for differentiability
    shifted = beta * (conv - max_val)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted))
    soft_max = max_val + log_sum_exp / beta
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max

def _ratio_neg(h, beta=100.0):
    return -_ratio(h, beta)

def _is_sidon(set_indices):
    """Check if a set of indices forms a Sidon set (all pairwise differences distinct)."""
    diffs = set()
    n = len(set_indices)
    for i in range(n):
        for j in range(i + 1, n):
            d = set_indices[j] - set_indices[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _find_sidon_set(max_size=10, time_limit=10.0):
    """Search for a large Sidon set in Z_N."""
    best_set = []
    start_time = time.time()
    
    # Greedy construction
    # Try to add elements one by one
    current_set = [0]
    
    # Randomized greedy with restarts
    rng = np.random.RandomState(42)
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        current_set = [rng.randint(N)]
        for candidate in range(N):
            if candidate in current_set:
                continue
            trial_set = current_set + [candidate]
            if _is_sidon(trial_set):
                current_set = trial_set
                if len(current_set) > len(best_set):
                    best_set = current_set.copy()
                    if len(best_set) >= max_size:
                        return best_set
                        
    return best_set

def construct():
    start_time = time.time()
    
    # 1. Find a good Sidon set
    sidon_set = _find_sidon_set(max_size=12, time_limit=5.0)
    
    # Initialize h with uniform weights on the Sidon set
    h_best = np.zeros(N)
    if len(sidon_set) > 0:
        for idx in sidon_set:
            h_best[idx] = 1.0 / len(sidon_set)
    
    best_score = _ratio(h_best)
    
    # 2. Refine weights using optimization
    # We keep the support fixed (Sidon set) but optimize the weights
    # This might yield a better ratio than uniform weights
    
    def objective_weights(w):
        h = np.zeros(N)
        for i, idx in enumerate(sidon_set):
            h[idx] = w[i]
        return _ratio_neg(h)
    
    w0 = np.ones(len(sidon_set)) / len(sidon_set)
    
    try:
        if _time_left(start_time) > 10.0:
            result = minimize(objective_weights, w0, method='L-BFGS-B',
                              bounds=[(0, None)] * len(sidon_set),
                              options={'maxiter': 500, 'gtol': 1e-9})
            w_opt = result.x
            h_opt = np.zeros(N)
            for i, idx in enumerate(sidon_set):
                h_opt[idx] = w_opt[i]
            
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                h_best = h_opt
    except Exception:
        pass
    
    # 3. Try a few other Sidon sets if time permits
    if _time_left(start_time) > 20.0:
        # Find a different Sidon set with a different seed
        rng2 = np.random.RandomState(123)
        current_set = [rng2.randint(N)]
        for candidate in range(N):
            if candidate in current_set:
                continue
            trial_set = current_set + [candidate]
            if _is_sidon(trial_set):
                current_set = trial_set
                if len(current_set) > 5:
                    break
        
        if len(current_set) > len(sidon_set):
            sidon_set = current_set
            h_new = np.zeros(N)
            for idx in sidon_set:
                h_new[idx] = 1.0 / len(sidon_set)
            
            s_new = _ratio(h_new)
            if s_new > best_score:
                best_score = s_new
                h_best = h_new
                
            # Optimize weights for this new set
            def objective_weights2(w):
                h = np.zeros(N)
                for i, idx in enumerate(sidon_set):
                    h[idx] = w[i]
                return _ratio_neg(h)
            
            w0_2 = np.ones(len(sidon_set)) / len(sidon_set)
            try:
                if _time_left(start_time) > 10.0:
                    result2 = minimize(objective_weights2, w0_2, method='L-BFGS-B',
                                       bounds=[(0, None)] * len(sidon_set),
                                       options={'maxiter': 300, 'gtol': 1e-9})
                    w_opt2 = result2.x
                    h_opt2 = np.zeros(N)
                    for i, idx in enumerate(sidon_set):
                        h_opt2[idx] = w_opt2[i]
                    
                    s_opt2 = _ratio(h_opt2)
                    if s_opt2 > best_score:
                        best_score = s_opt2
                        h_best = h_opt2
            except Exception:
                pass

    return h_best.tolist()

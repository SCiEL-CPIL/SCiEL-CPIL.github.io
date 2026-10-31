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

def _find_sidon_set_greedy(n_elements, max_index, seed=42):
    """
    Find a greedy Sidon-like set: indices where all pairwise sums are distinct.
    This minimizes autoconvolution collisions.
    """
    rng = np.random.RandomState(seed)
    best_set = None
    best_size = 0
    
    for attempt in range(50):
        used_sums = set()
        selected = []
        
        # Shuffle candidate indices
        candidates = list(range(max_index))
        rng.shuffle(candidates)
        
        for idx in candidates:
            if len(selected) >= n_elements:
                break
            # Check if adding idx creates any sum collision
            new_sums = set()
            for j in selected:
                s = idx + j
                if s in used_sums or s in new_sums:
                    break
                new_sums.add(s)
            else:
                # Also check self-sum
                s_self = idx + idx
                if s_self in used_sums or s_self in new_sums:
                    break
                new_sums.add(s_self)
                selected.append(idx)
                used_sums.update(new_sums)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = sorted(selected)
    
    return best_set if best_set else [0]

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
        
    # Candidate 3: Sidon-set based sparse vectors
    # Try different Sidon set sizes
    for n_sidon in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        if time.time() > deadline:
            break
        sidon = _find_sidon_set_greedy(n_sidon, N, seed=n_sidon)
        if len(sidon) < 2:
            continue
        h_sidon = np.zeros(N)
        for idx in sidon:
            h_sidon[idx] = 1.0
        score_sidon = _calculate_ratio(h_sidon)
        if score_sidon > best_score:
            best_score = score_sidon
            best_h = h_sidon.copy()
    
    # Candidate 4: Smooth optimization with Sidon-inspired initializations
    def objective_smooth(h):
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 1e10
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-10:
            return 1e10
        conv = np.convolve(h, h, mode='full')
        tau = 0.01
        # Clip to avoid overflow in exp
        conv_clipped = np.clip(conv, -50, 50)
        soft_max = tau * np.log(np.sum(np.exp(conv_clipped / tau)))
        return -np.log(l2_sq) + np.log(l1) + soft_max
    
    np.random.seed(42)
    
    # Phase 1: Random restarts
    for i in range(15):
        if time.time() > deadline:
            break
        h_init = np.random.exponential(1.0, N)
        h_init = h_init / np.sum(h_init)
        bounds = [(0, None)] * N
        try:
            res = minimize(
                objective_smooth, 
                h_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = _calculate_ratio(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Sidon-initialized restarts
    for n_sidon in [4, 5, 6, 8, 10]:
        if time.time() > deadline:
            break
        sidon = _find_sidon_set_greedy(n_sidon, N, seed=100 + n_sidon)
        if len(sidon) < 2:
            continue
        h_init = np.zeros(N)
        for idx in sidon:
            h_init[idx] = 1.0
        h_init = h_init / np.sum(h_init)
        
        # Add small random perturbation
        h_init = h_init + 0.1 * np.random.exponential(1.0, N)
        h_init = np.maximum(h_init, 0)
        h_init = h_init / np.sum(h_init)
        
        bounds = [(0, None)] * N
        try:
            res = minimize(
                objective_smooth, 
                h_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = _calculate_ratio(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Phase 3: Polish the best solution with tighter tolerance
    if best_h is not None and time.time() < deadline - 10:
        h_polish = best_h.copy()
        bounds = [(0, None)] * N
        try:
            res = minimize(
                objective_smooth, 
                h_polish, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = _calculate_ratio(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    if best_h is None:
        return [1.0] + [0.0] * (N - 1)
        
    return list(best_h)

# EVOLVE-BLOCK-END

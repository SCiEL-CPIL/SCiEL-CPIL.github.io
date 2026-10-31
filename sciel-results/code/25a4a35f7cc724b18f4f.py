import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def compute_ratio_logsumexp(h, beta=50.0):
    """Smooth surrogate for compute_ratio using log-sum-exp for the max."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    # Use log-sum-exp for smooth max
    conv_max_raw = np.max(conv)
    if conv_max_raw < 1e-15:
        return 0.0
    # Smooth max via log-sum-exp: log(sum(exp(beta*x))) / beta approximates max
    shifted = conv - conv_max_raw
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) / beta + conv_max_raw
    if log_sum_exp < 1e-15:
        return 0.0
    return h2_sq / log_sum_exp

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def local_search_swap(h, indices, max_swaps=100, use_sidon_constraint=True):
    """Try swapping elements in the support to reduce max convolution."""
    best_h = h.copy()
    best_score = compute_ratio(h)
    current_indices = list(indices)
    
    for swap_iter in range(max_swaps):
        improved = False
        for i in range(len(current_indices)):
            for j in range(N):
                if j in current_indices:
                    continue
                new_indices = current_indices.copy()
                new_indices[i] = j
                new_indices = sorted(new_indices)
                
                if use_sidon_constraint and not is_sidon_set(new_indices):
                    continue
                
                new_h = np.zeros(N)
                for idx in new_indices:
                    new_h[idx] = 1.0
                sc = compute_ratio(new_h)
                if sc > best_score + 1e-12:
                    best_h = new_h
                    best_score = sc
                    current_indices = new_indices
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h, best_score

def optimize_weights_for_support(indices, h_init, deadline, use_logsumexp=False, beta=50.0):
    """Optimize weights on a fixed support using smooth objective."""
    if len(indices) == 0:
        return h_init.copy(), compute_ratio(h_init)
    
    indices = np.array(indices)
    h_sparse_init = h_init[indices].copy()
    
    def objective_smooth(x):
        x = np.maximum(x, 0.0)
        s = np.sum(x)
        if s < 1e-15:
            return 1e10
        x_norm = x / s
        h = np.zeros(N)
        h[indices] = x_norm
        if use_logsumexp:
            return -compute_ratio_logsumexp(h, beta=beta)
        else:
            return -compute_ratio(h)
    
    def objective_epigraph(x):
        """Epigraph formulation: minimize t subject to ratio >= 1/t"""
        # This is tricky, so we use a penalty approach
        x = np.maximum(x, 0.0)
        s = np.sum(x)
        if s < 1e-15:
            return 1e10
        x_norm = x / s
        h = np.zeros(N)
        h[indices] = x_norm
        h2_sq = np.sum(x_norm**2)
        conv = fftconvolve(x_norm, x_norm, mode='full')
        conv_inf = np.max(conv)
        if conv_inf < 1e-15:
            return 1e10
        ratio = h2_sq / conv_inf
        # Penalize small ratios
        return -ratio + 1.0 / (ratio + 1e-10)
    
    # Try multiple optimization strategies
    best_x = h_sparse_init.copy()
    best_score = compute_ratio(h_init)
    
    # Strategy 1: Standard Nelder-Mead with exact objective
    try:
        res1 = minimize(objective_smooth, h_sparse_init, 
                       method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        h_opt1_sparse = np.maximum(res1.x, 0.0)
        h_opt1 = np.zeros(N)
        h_opt1[indices] = h_opt1_sparse
        s1 = np.sum(h_opt1)
        if s1 > 1e-15:
            h_opt1 = h_opt1 / s1
            sc1 = compute_ratio(h_opt1)
            if sc1 > best_score:
                best_score = sc1
                best_x = h_opt1_sparse
    except Exception:
        pass
    
    # Strategy 2: With log-sum-exp smoothing
    if use_logsumexp:
        try:
            def objective_lse(x):
                x = np.maximum(x, 0.0)
                s = np.sum(x)
                if s < 1e-15:
                    return 1e10
                x_norm = x / s
                h = np.zeros(N)
                h[indices] = x_norm
                return -compute_ratio_logsumexp(h, beta=beta)
            
            res2 = minimize(objective_lse, best_x, 
                           method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            h_opt2_sparse = np.maximum(res2.x, 0.0)
            h_opt2 = np.zeros(N)
            h_opt2[indices] = h_opt2_sparse
            s2 = np.sum(h_opt2)
            if s2 > 1e-15:
                h_opt2 = h_opt2 / s2
                sc2 = compute_ratio(h_opt2)
                if sc2 > best_score:
                    best_score = sc2
                    best_x = h_opt2_sparse
        except Exception:
            pass
    
    # Final polish with tight tolerances
    try:
        res_final = minimize(objective_smooth, best_x, 
                            method='Nelder-Mead', 
                            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
        h_final_sparse = np.maximum(res_final.x, 0.0)
        h_final = np.zeros(N)
        h_final[indices] = h_final_sparse
        s_final = np.sum(h_final)
        if s_final > 1e-15:
            h_final = h_final / s_final
            sc_final = compute_ratio(h_final)
            if sc_final > best_score:
                best_score = sc_final
                best_x = h_final_sparse
    except Exception:
        pass
    
    h_result = np.zeros(N)
    h_result[indices] = best_x
    s_res = np.sum(h_result)
    if s_res > 1e-15:
        h_result = h_result / s_res
    
    return h_result, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))

    # Phase 1: Generate and evaluate Sidon sets
    sidon_sets = generate_sidon_sets()
    
    for s in sidon_sets:
        if time.time() > deadline - 60.0:
            break
        h = np.zeros(N)
        for idx in s:
            h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 1b: Random sparse search with various sparsities
    np.random.seed(42)
    num_sparse_tries = 2000
    for _ in range(num_sparse_tries):
        if time.time() > deadline - 50.0:
            break
        # Try different sparsity levels
        if np.random.rand() < 0.3:
            k = np.random.randint(3, 8)  # Very sparse
        elif np.random.rand() < 0.6:
            k = np.random.randint(8, 15)  # Medium sparse
        else:
            k = np.random.randint(15, 25)  # Denser
        
        indices = np.random.choice(N, k, replace=False)
        
        # Try different weight distributions
        r = np.random.rand()
        if r < 0.3:
            weights = np.random.rand(k)  # Uniform random
        elif r < 0.6:
            weights = np.ones(k)  # Equal weights
        else:
            weights = np.random.exponential(1.0, k)  # Exponential
        
        h = np.zeros(N)
        h[indices] = weights
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 1c: Try structured patterns
    patterns = [
        # Geometric progression
        [int(N * (i / 8.0)) for i in range(8)],
        # Prime indices
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        # Fibonacci-like
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
        # Power of 2
        [0, 1, 2, 4, 8, 16, 32],
        # Mixed
        [0, 1, 3, 7, 15, 31],
    ]
    for pat in patterns:
        if time.time() > deadline - 45.0:
            break
        valid_pat = [x for x in pat if 0 <= x < N]
        if len(valid_pat) >= 3:
            h = np.zeros(N)
            for idx in valid_pat:
                h[idx] = 1.0
            sc = compute_ratio(h)
            update_best(h, sc)
    
    # Sort candidates by score and keep top K for local refinement
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:50]
    
    # Phase 2: Local optimization from Top Candidates
    for i, (sc, h_start) in enumerate(top_k):
        if time.time() > deadline - 20.0:
            break
        
        indices = np.where(h_start > 1e-10)[0]
        if len(indices) == 0:
            continue
        
        # Use the improved weight optimizer
        h_opt, sc_opt = optimize_weights_for_support(indices, h_start, deadline, use_logsumexp=True, beta=50.0)
        update_best(h_opt, sc_opt)
        
        # Try local search with swaps on the best few
        if i < 10 and time.time() < deadline - 15.0:
            h_swapped, sc_swapped = local_search_swap(h_opt, indices, max_swaps=50, use_sidon_constraint=False)
            if sc_swapped > best_score:
                update_best(h_swapped, sc_swapped)
                new_indices = np.where(h_swapped > 1e-10)[0]
                if len(new_indices) > 0:
                    h_opt2, sc_opt2 = optimize_weights_for_support(new_indices, h_swapped, deadline, use_logsumexp=True, beta=50.0)
                    update_best(h_opt2, sc_opt2)
    
    # Phase 3: Final high-precision polish on best solution
    if time.time() < deadline - 5.0:
        best_indices = np.where(best_h > 1e-10)[0]
        if len(best_indices) > 0:
            h_final, sc_final = optimize_weights_for_support(best_indices, best_h, deadline, use_logsumexp=True, beta=100.0)
            update_best(h_final, sc_final)
        
        # One more round of swap search
        if time.time() < deadline - 3.0:
            best_indices = np.where(best_h > 1e-10)[0]
            if len(best_indices) > 0:
                h_swapped, sc_swapped = local_search_swap(best_h, best_indices, max_swaps=30, use_sidon_constraint=False)
                if sc_swapped > best_score:
                    update_best(h_swapped, sc_swapped)
                    new_indices = np.where(h_swapped > 1e-10)[0]
                    if len(new_indices) > 0 and time.time() < deadline - 1.0:
                        h_final2, sc_final2 = optimize_weights_for_support(new_indices, h_swapped, deadline, use_logsumexp=True, beta=100.0)
                        update_best(h_final2, sc_final2)

    return list(best_h)

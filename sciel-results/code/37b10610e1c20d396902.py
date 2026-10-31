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

def is_sidon_set(indices):
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

def optimize_weights_for_support(support, beta=100.0, max_iter=1000):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    def objective_and_grad(x):
        # x are weights for support, sum to 1
        h = np.zeros(N)
        h[support] = x
        
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Smooth max surrogate
        obj_val = max_c + np.log(sum_exp) / beta
        
        # Gradient: d/dx_j = 2 * sum_m weights[m] * h[m-j] for m in support
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                # For each j in support, grad_h[j] += 2 * hm * weights[m+j]
                for j in support:
                    idx = m + j
                    if 0 <= idx < len(weights):
                        grad_h[j] += 2.0 * hm * weights[idx]
        
        grad_x = grad_h[support]
        return obj_val, grad_x
    
    # Start with uniform weights
    x0 = np.ones(k) / k
    
    best_x = x0.copy()
    best_obj = objective_and_grad(x0)[0]
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    # Normalize to ensure sum = 1
    s = np.sum(best_x)
    if s > 1e-15:
        best_x = best_x / s
    
    h_full = np.zeros(N)
    h_full[support] = best_x
    sc = compute_ratio(h_full)
    
    return h_full, sc

def polish_weights(h, support, beta=500.0, max_iter=500):
    """Final high-precision polish"""
    k = len(support)
    x0 = h[support].copy()
    
    def objective_and_grad(x):
        h_local = np.zeros(N)
        h_local[support] = x
        
        conv = fftconvolve(h_local, h_local, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        obj_val = max_c + np.log(sum_exp) / beta
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h_local[m]
            if hm > 1e-15:
                for j in support:
                    idx = m + j
                    if 0 <= idx < len(weights):
                        grad_h[j] += 2.0 * hm * weights[idx]
        
        grad_x = grad_h[support]
        return obj_val, grad_x
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
        if res.success:
            x_opt = res.x
            s = np.sum(x_opt)
            if s > 1e-15:
                x_opt = x_opt / s
            h_full = np.zeros(N)
            h_full[support] = x_opt
            sc = compute_ratio(h_full)
            return h_full, sc
    except Exception:
        pass
    
    return h, compute_ratio(h)

def local_search_swap(h, support, max_iters=50, deadline=None):
    k = len(support)
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    best_support = list(support)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try swapping one index
        for old_idx in best_support:
            if deadline is not None and time.time() > deadline:
                break
            for new_idx in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                if new_idx in best_support:
                    continue
                new_support = [x for x in best_support if x != old_idx]
                new_support.append(new_idx)
                new_support.sort()
                
                h_new, sc_new = optimize_weights_for_support(new_support, beta=80.0, max_iter=500)
                if sc_new > best_score + 1e-6:
                    best_h = h_new
                    best_score = sc_new
                    best_support = new_support
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_h, best_score, best_support

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    sidon_sets = generate_sidon_sets()
    sidon_sets.sort(key=lambda x: -len(x))
    
    for s in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        
        h_opt, sc = optimize_weights_for_support(s, beta=100.0, max_iter=1000)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
            best_support = list(s)
    
    # Local search
    if time.time() < deadline - 25.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=50, deadline=deadline - 15.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    # Final polish
    if time.time() < deadline - 5.0:
        h_polish, sc_polish = polish_weights(best_h, best_support, beta=500.0, max_iter=500)
        if sc_polish > best_score:
            best_score = sc_polish
            best_h = h_polish.copy()
    
    return list(best_h)

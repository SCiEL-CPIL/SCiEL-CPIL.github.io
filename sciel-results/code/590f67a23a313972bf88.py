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
    
    np.random.seed(42)
    for _ in range(15000):
        k = np.random.randint(5, 13)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 120:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, maxiter=3000, ftol=1e-14, time_limit=None):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    t0 = time.time()
    deadline = t0 + time_limit if time_limit else float('inf')
    
    x0 = np.zeros(k)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        if time.time() > deadline:
            raise ValueError("timeout")
        h = softmax(x)
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        obj_val = max_c + (1.0 / beta) * np.log(sum_exp)
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h_full[m]
            if hm > 1e-15:
                end = min(m + N, 2 * N - 1)
                grad_h[:N] += 2.0 * hm * weights[m:end]
        
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-12})
        x_opt = res.x
        h_opt_local = softmax(x_opt)
        
        h_full = np.zeros(N)
        h_full[support] = h_opt_local
        sc = compute_ratio(h_full)
        return h_full, sc
    except Exception:
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        sc = compute_ratio(h_full)
        return h_full, sc

def gradient_polish(h_full, support, maxiter=2000, ftol=1e-15, time_limit=None):
    """
    Direct gradient-based polish on the simplex using L-BFGS-B with a smooth surrogate.
    """
    k = len(support)
    if k == 0:
        return h_full, compute_ratio(h_full)
    
    t0 = time.time()
    deadline = t0 + time_limit if time_limit else float('inf')
    
    h_local = h_full[support].copy()
    h_local = np.maximum(h_local, 1e-10)
    h_local = h_local / np.sum(h_local)
    
    x0 = np.log(h_local)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        if time.time() > deadline:
            raise ValueError("timeout")
        h = softmax(x)
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        # Use a high beta for sharpness
        beta = 500.0
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        obj_val = max_c + (1.0 / beta) * np.log(sum_exp)
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h_full[m]
            if hm > 1e-15:
                end = min(m + N, 2 * N - 1)
                grad_h[:N] += 2.0 * hm * weights[m:end]
        
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-12})
        x_opt = res.x
        h_opt_local = softmax(x_opt)
        
        h_full_new = np.zeros(N)
        h_full_new[support] = h_opt_local
        sc = compute_ratio(h_full_new)
        return h_full_new, sc
    except Exception:
        return h_full, compute_ratio(h_full)

def local_search_polish(h_full, support, time_limit=None):
    """
    Local search with swaps and weight adjustments.
    """
    t0 = time.time()
    deadline = t0 + time_limit if time_limit else float('inf')
    
    best_h = h_full.copy()
    best_score = compute_ratio(best_h)
    best_support = list(support)
    
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        
        # Try swapping one element
        for old_idx in best_support:
            if time.time() > deadline - 1.0:
                break
            for new_idx in range(N):
                if new_idx in best_support:
                    continue
                new_support = [x for x in best_support if x != old_idx]
                new_support.append(new_idx)
                new_support.sort()
                
                if not is_sidon_set(new_support):
                    continue
                
                # Quick check: uniform weights
                h_uniform = np.zeros(N)
                h_uniform[new_support] = 1.0 / len(new_support)
                sc_uniform = compute_ratio(h_uniform)
                
                # If uniform is good, try optimization
                if sc_uniform > best_score * 0.9:
                    h_new, sc_new = optimize_weights_for_support(new_support, beta=80.0, maxiter=300, time_limit=0.5)
                    if sc_new > best_score + 1e-8:
                        best_h = h_new
                        best_score = sc_new
                        best_support = new_support
                        improved = True
                        break
            
            if improved:
                break
        
        # Try adjusting weights via gradient polish
        if time.time() < deadline - 1.0:
            h_polished, sc_polished = gradient_polish(best_h, best_support, maxiter=1000, ftol=1e-15, time_limit=1.0)
            if sc_polished > best_score + 1e-8:
                best_h = h_polished
                best_score = sc_polished
                improved = True
    
    return best_h, best_score, best_support

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    sidon_sets = generate_sidon_sets()
    sidon_sets.sort(key=lambda x: -len(x))
    
    # Phase 1: Try Sidon sets with L-BFGS optimization
    for s in sidon_sets:
        if time.time() > deadline - 40.0:
            break
        
        # Try multiple betas
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 40.0:
                break
            h_opt, sc = optimize_weights_for_support(s, beta=beta, maxiter=2000, time_limit=2.0)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    # Phase 2: Local search with swaps
    if time.time() < deadline - 20.0:
        best_h, best_score, best_support = local_search_polish(best_h, best_support, time_limit=deadline - time.time() - 15.0)
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 10.0:
        # Gradient polish with high precision
        h_polish, sc_polish = gradient_polish(best_h, best_support, maxiter=5000, ftol=1e-15, time_limit=deadline - time.time() - 5.0)
        if sc_polish > best_score:
            best_score = sc_polish
            best_h = h_polish.copy()
        
        # Final L-BFGS polish with high beta
        if time.time() < deadline - 5.0:
            h_final, sc_final = optimize_weights_for_support(best_support, beta=500.0, maxiter=5000, ftol=1e-15, time_limit=deadline - time.time() - 2.0)
            if sc_final > best_score:
                best_score = sc_final
                best_h = h_final.copy()
    
    return list(best_h)

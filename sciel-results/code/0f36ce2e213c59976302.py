import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50
WALL_CLOCK_LIMIT = 150.0
SAFETY_MARGIN = 10.0

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

def greedy_sidon_search(indices, deadline):
    """Greedy heuristic with random restarts to find a large Sidon subset of indices.
    Returns the best Sidon set found."""
    best_sidon = set()
    for restart in range(50):
        if time.time() > deadline - 20.0:
            break
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        current = set()
        for idx in shuffled:
            test_set = list(current) + [idx]
            if is_sidon_set(test_set):
                current.add(idx)
        if len(current) > len(best_sidon):
            best_sidon = set(current)
    return list(best_sidon)

def smooth_max_surrogate(conv, temperature=0.1):
    """Log-sum-exp approximation to max for smooth optimization."""
    scaled = conv / temperature
    max_val = np.max(scaled)
    if max_val > 700:
        max_val = 700
    return temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))

def smooth_max_surrogate_grad(conv, temperature=0.1):
    """Log-sum-exp approximation to max, returning (value, gradient w.r.t. conv)."""
    scaled = conv / temperature
    max_val = np.max(scaled)
    if max_val > 700:
        max_val = 700
    shifted = scaled - max_val
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    value = temperature * (max_val + log_sum)
    # gradient of value w.r.t. conv: exp_shifted / sum(exp_shifted)
    grad = exp_shifted / np.sum(exp_shifted)
    return value, grad

def optimize_weights_logspace(indices, h_start, deadline, use_smooth=False, temperature=0.05):
    """Optimize weights on fixed support using log-space parametrization."""
    k = len(indices)
    if k == 0:
        return np.zeros(N)
    
    h_init = h_start[indices].copy()
    h_init = np.maximum(h_init, 1e-10)
    x_init = np.log(h_init)
    
    def objective(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        h = exp_x / np.sum(exp_x)
        
        h2_sq = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        
        if use_smooth:
            conv_inf = smooth_max_surrogate(conv, temperature)
        else:
            conv_inf = np.max(conv)
        
        if conv_inf < 1e-15:
            return 1e10
        return -h2_sq / conv_inf
    
    res = minimize(objective, x_init, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    x_opt = res.x
    x_shifted = x_opt - np.max(x_opt)
    exp_x = np.exp(x_shifted)
    h_opt = exp_x / np.sum(exp_x)
    
    h_full = np.zeros(N)
    h_full[indices] = h_opt
    return h_full

def gradient_polish(h, indices, deadline, max_iter=200):
    """L-BFGS-B polish with log-sum-exp smooth surrogate for the max constraint.
    Uses epigraph-style formulation: minimize -h2_sq subject to conv_inf <= t,
    but here we directly optimize the smooth ratio."""
    k = len(indices)
    if k < 2:
        return h.copy()
    
    h_init = h[indices].copy()
    h_init = np.maximum(h_init, 1e-12)
    
    # Normalize to sum=1 for optimization
    s = np.sum(h_init)
    if s < 1e-15:
        return h.copy()
    h_init = h_init / s
    
    # Log-space parametrization: x_i = log(h_i)
    x_init = np.log(h_init)
    
    def smooth_objective(x):
        """Minimize -h2_sq / smooth_max(conv) where h = softmax(x)."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        h = exp_x / np.sum(exp_x)
        
        h2_sq = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        
        # Anneal temperature
        temp = 0.02
        conv_inf, _ = smooth_max_surrogate_grad(conv, temp)
        
        if conv_inf < 1e-15:
            return 1e10
        return -h2_sq / conv_inf
    
    def smooth_gradient(x):
        """Numerical gradient (finite differences) for L-BFGS-B."""
        eps = 1e-7
        grad = np.zeros(k)
        f0 = smooth_objective(x)
        for i in range(k):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (smooth_objective(x_plus) - f0) / eps
        return grad
    
    if time.time() > deadline - 15.0:
        return h.copy()
    
    res = minimize(smooth_objective, x_init, method='L-BFGS-B',
                   jac=smooth_gradient,
                   options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-12,
                            'maxls': 50})
    
    x_opt = res.x
    x_shifted = x_opt - np.max(x_opt)
    exp_x = np.exp(x_shifted)
    h_opt = exp_x / np.sum(exp_x)
    
    h_full = h.copy()
    h_full[indices] = h_opt
    return h_full

def local_search_polish(h, deadline):
    """Enhanced local search with patience and more perturbations."""
    h = h.copy()
    improved = True
    max_iter = 200
    patience = 0
    
    for _ in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        if not improved and patience > 5:
            break
        improved = False
        current_score = compute_ratio(h)
        
        nonzero_idx = set(np.where(h > 1e-10)[0])
        zero_idx = [i for i in range(N) if i not in nonzero_idx]
        
        for idx in zero_idx[:10]:
            if time.time() > deadline - 10.0:
                break
            for w in [0.01, 0.05, 0.1, 0.2, 0.5]:
                h_test = h.copy()
                h_test[idx] = w
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-10:
                    h = h_test
                    current_score = sc
                    improved = True
                    patience = 0
                    break
            if improved:
                break
        
        if not improved:
            nonzero_list = list(nonzero_idx)
            for idx in nonzero_list:
                if time.time() > deadline - 10.0:
                    break
                h_test = h.copy()
                h_test[idx] = 0.0
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-10:
                    h = h_test
                    current_score = sc
                    improved = True
                    patience = 0
                    break
        
        if not improved:
            nonzero_list = list(nonzero_idx)
            for idx in nonzero_list:
                if time.time() > deadline - 10.0:
                    break
                for factor in [0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
                    h_test = h.copy()
                    h_test[idx] *= factor
                    sc = compute_ratio(h_test)
                    if sc > current_score + 1e-10:
                        h = h_test
                        current_score = sc
                        improved = True
                        patience = 0
                        break
                if improved:
                    break
        
        if not improved:
            patience += 1
    
    return h

def high_precision_polish(h, deadline):
    """Final high-precision polish with tight tolerances."""
    h = h.copy()
    indices = np.where(h > 1e-10)[0]
    if len(indices) == 0:
        return h
    
    # Multiple passes with decreasing step sizes
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    factor_sizes = [[0.97, 0.99, 1.01, 1.03],
                    [0.985, 0.995, 1.005, 1.015],
                    [0.99, 0.998, 1.002, 1.01],
                    [0.995, 0.999, 1.001, 1.005],
                    [0.998, 0.9995, 1.0005, 1.002]]
    
    for pass_idx, (step, factors) in enumerate(zip(step_sizes, factor_sizes)):
        if time.time() > deadline - 5.0:
            break
        current_score = compute_ratio(h)
        
        for _ in range(3):  # Multiple sweeps per step size
            if time.time() > deadline - 5.0:
                break
            
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                for delta in [-step, -step/2, step/2, step]:
                    h_test = h.copy()
                    h_test[idx] = max(0, h_test[idx] + delta)
                    sc = compute_ratio(h_test)
                    if sc > current_score + 1e-13:
                        h = h_test
                        current_score = sc
                        break
            
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                for factor in factors:
                    h_test = h.copy()
                    h_test[idx] *= factor
                    sc = compute_ratio(h_test)
                    if sc > current_score + 1e-13:
                        h = h_test
                        current_score = sc
                        break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + WALL_CLOCK_LIMIT - SAFETY_MARGIN

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

    np.random.seed(42)

    all_indices = list(range(N))
    
    # Phase 1: Find good Sidon sets (quick, ~20s budget)
    sidon_deadline = t0 + 20.0
    best_sidon = []
    for restart in range(20):
        if time.time() > sidon_deadline:
            break
        sidon_set = greedy_sidon_search(all_indices, sidon_deadline)
        if len(sidon_set) > len(best_sidon):
            best_sidon = sidon_set
    
    known_sidon_patterns = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 5, 11, 23],
        [0, 1, 4, 9, 16, 25, 36, 49],
        [0, 1, 5, 11, 25, 26, 31, 37, 43, 49],
        [0, 2, 3, 7, 12, 18, 25, 31, 38, 44, 49],
    ]
    
    all_sidon_candidates = [best_sidon] + known_sidon_patterns
    
    for sidon_set in all_sidon_candidates:
        if time.time() > deadline - 90.0:
            break
        if len(sidon_set) < 2:
            continue
        h = np.zeros(N)
        for idx in sidon_set:
            if 0 <= idx < N:
                h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 2: Diverse random initializations (~30s budget)
    random_deadline = deadline - 60.0
    num_sparse_tries = 600
    for _ in range(num_sparse_tries):
        if time.time() > random_deadline:
            break
        
        k = np.random.randint(3, 15)
        indices = np.random.choice(N, k, replace=False)
        
        # Mix of uniform and structured weights
        wtype = np.random.random()
        if wtype < 0.3:
            weights = np.random.rand(k)
        elif wtype < 0.6:
            weights = np.random.rand(k) ** np.random.uniform(0.5, 2.0)
        elif wtype < 0.8:
            # Geometric spacing
            ratio = np.random.uniform(0.5, 1.5)
            weights = np.array([ratio**i for i in range(k)])
        else:
            # Bimodal
            weights = np.concatenate([np.random.rand(k//2)*0.3, np.random.rand(k - k//2)])
        
        h = np.zeros(N)
        h[indices] = weights
        
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 3: Nelder-Mead optimization on top candidates (~30s budget)
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:15]
    
    nm_deadline = deadline - 30.0
    for i, (sc, h_start) in enumerate(top_k):
        if time.time() > nm_deadline:
            break
        
        indices = np.where(h_start > 1e-10)[0]
        if len(indices) == 0:
            continue
            
        for use_smooth in [False, True]:
            if time.time() > nm_deadline:
                break
            
            h_opt = optimize_weights_logspace(indices, h_start, nm_deadline,
                                              use_smooth=use_smooth, temperature=0.05)
            
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc_opt = compute_ratio(h_opt)
                update_best(h_opt, sc_opt)
    
    # Phase 4: Gradient polish with smooth surrogate (~20s budget)
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_grad = candidates[:10]
    
    grad_deadline = deadline - 20.0
    for i, (sc, h_start) in enumerate(top_grad):
        if time.time() > grad_deadline:
            break
        
        indices = np.where(h_start > 1e-10)[0]
        if len(indices) < 2:
            continue
        
        h_grad = gradient_polish(h_start, indices, grad_deadline, max_iter=100)
        sc_grad = compute_ratio(h_grad)
        update_best(h_grad, sc_grad)
    
    # Phase 5: Local search polish (~10s budget)
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_polish = candidates[:6]
    
    polish_deadline = deadline - 8.0
    for i, (sc, h_start) in enumerate(top_polish):
        if time.time() > polish_deadline:
            break
        
        h_polished = local_search_polish(h_start, polish_deadline)
        sc_polished = compute_ratio(h_polished)
        update_best(h_polished, sc_polished)
    
    # Phase 6: Final high-precision polish
    if time.time() < deadline - 5.0:
        h_final = high_precision_polish(best_h, deadline)
        sc_final = compute_ratio(h_final)
        update_best(h_final, sc_final)

    # Safety: ensure we return before wall-clock limit
    if time.time() > WALL_CLOCK_LIMIT - 3.0:
        return list(best_h)
    
    return list(best_h)

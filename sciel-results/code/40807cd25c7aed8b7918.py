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
    # Exclude lag 0 (index N-1)
    conv_offpeak = np.delete(conv, N-1)
    conv_inf = np.max(conv_offpeak)
    if conv_inf < 1e-15:
        return h2_sq / 1e-15
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

def optimize_weights_for_support(support, beta=100.0):
    """
    Optimize weights h_i for i in support to maximize ratio ||h||_2^2 / max_offpeak(h*h).
    Uses smooth max surrogate for the off-peak convolution terms.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    x0 = np.zeros(k)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        h = softmax(x)
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        # Exclude lag 0 (index N-1)
        conv_offpeak = np.delete(conv, N-1)
        
        max_c = np.max(conv_offpeak)
        if max_c < 1e-15:
            # If off-peak is zero, the ratio is very high.
            # We want to minimize the log-objective.
            # log(max) - log(h2). If max->0, log(max)->-inf.
            # So objective -> -inf.
            # This is good for minimization.
            # But gradient might be unstable.
            # Let's return a large negative value and zero gradient?
            # Or just use a small epsilon.
            max_c = 1e-15
        
        shifted = conv_offpeak - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Objective: (1/beta) * log(sum(exp(beta * conv_k))) - log(||h||_2^2)
        h2_sq = np.sum(h_full**2)
        if h2_sq < 1e-15:
            h2_sq = 1e-15
            
        obj_val = (1.0 / beta) * np.log(sum_exp) + max_c - np.log(h2_sq)
        
        # Gradient
        # d obj / d h_j = sum_{k != 0} weights_k * (h_{j+k} + h_{j-k}) - 2*h_j / h2_sq
        
        # weights corresponds to conv_offpeak indices.
        # We need to map back to original conv indices to handle lags correctly.
        # conv_offpeak is conv with index N-1 removed.
        # Let's create a full weights array for conv.
        w_full = np.zeros(len(conv))
        # Map weights back to conv indices
        # Indices in conv_offpeak: 0..N-2, N..2N-2
        # Indices in conv: 0..N-2, N-1, N..2N-2
        w_full[:N-1] = weights[:N-1]
        w_full[N:] = weights[N-1:]
        # w_full[N-1] is 0 (excluded)
        
        grad_h = np.zeros(N)
        for j in range(N):
            # Sum over all lags m
            # Term: w_full[m] * (h_{j + (m - (N-1))} + h_{j - (m - (N-1))})
            # Let lag = m - (N-1)
            # We iterate over m where w_full[m] != 0
            # This is O(N^2) which is 2500 ops. Fine.
            
            # Optimization: precompute non-zero weights
            pass
        
        # Efficient gradient calculation
        # grad_h[j] = sum_m w_full[m] * (h_{j + lag_m} + h_{j - lag_m})
        # where lag_m = m - (N-1)
        
        # We can use convolution/correlation tricks, but N=50 is small.
        # Let's just loop.
        
        # Precompute lags and weights
        m_indices = np.nonzero(w_full)[0]
        lags = m_indices - (N - 1)
        w_vals = w_full[m_indices]
        
        for j in range(N):
            # h_{j + lag}
            idx_plus = j + lags
            valid_plus = (idx_plus >= 0) & (idx_plus < N)
            term_plus = np.sum(w_vals[valid_plus] * h_full[idx_plus[valid_plus]])
            
            # h_{j - lag}
            idx_minus = j - lags
            valid_minus = (idx_minus >= 0) & (idx_minus < N)
            term_minus = np.sum(w_vals[valid_minus] * h_full[idx_minus[valid_minus]])
            
            grad_h[j] = term_plus + term_minus - 2.0 * h_full[j] / h2_sq
        
        # Chain rule for softmax
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-11})
        x_opt = res.x
        h_opt_local = softmax(x_opt)
        
        h_full = np.zeros(N)
        h_full[support] = h_opt_local
        sc = compute_ratio(h_full)
        return h_full, sc
    except Exception as e:
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        sc = compute_ratio(h_full)
        return h_full, sc

def local_search_swap(h, support, max_iters=100, deadline=None):
    """
    Local search: try swapping elements in support to improve the ratio.
    """
    k = len(support)
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    best_support = list(support)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try adding a new element
        for new_idx in range(N):
            if deadline is not None and time.time() > deadline:
                break
            if new_idx in best_support:
                continue
            new_support = best_support + [new_idx]
            new_support.sort()
            if len(new_support) > 15:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element
        for remove_idx in list(best_support):
            if deadline is not None and time.time() > deadline:
                break
            new_support = [x for x in best_support if x != remove_idx]
            if len(new_support) < 3:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping an element
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
                
                h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0)
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
        
        for beta in [30.0, 50.0, 80.0, 120.0]:
            if time.time() > deadline - 30.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    if time.time() < deadline - 25.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=80, deadline=deadline - 15.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    if time.time() < deadline - 5.0:
        for beta in [100.0, 200.0, 400.0, 800.0]:
            if time.time() > deadline - 5.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
    
    return list(best_h)

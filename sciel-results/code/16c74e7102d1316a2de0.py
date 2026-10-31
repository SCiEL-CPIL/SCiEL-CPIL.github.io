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

def optimize_weights_for_support(support, beta=100.0, maxiter=300):
    """
    Optimize weights h_i for i in support to maximize ratio = sum(h^2) / max(conv).
    Uses log-sum-exp smooth max surrogate with proper gradient.
    Parameterization: h = exp(x) / sum(exp(x)) ensures positivity and sum=1.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights in log-space
    x0 = np.zeros(k)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        h_support = softmax(x)
        h = np.zeros(N)
        h[support] = h_support
        
        conv = fftconvolve(h, h, mode='full')
        
        # Smooth max via log-sum-exp
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = max_c + np.log(sum_exp)
        smooth_max = log_sum_exp / beta
        
        # h2_sq = sum(h^2)
        h2_sq = np.sum(h_support ** 2)
        
        # We want to maximize h2_sq / smooth_max
        # Equivalent to minimize -log(h2_sq) + log(smooth_max)
        # Or simply minimize smooth_max / h2_sq
        # Let's minimize smooth_max - alpha * h2_sq for some alpha, or use ratio directly
        
        # Actually, let's minimize: smooth_max / h2_sq
        # This is equivalent to minimizing log(smooth_max) - log(h2_sq)
        
        if h2_sq < 1e-15:
            return 1e10, np.zeros(k)
        
        obj = np.log(smooth_max) - np.log(h2_sq)
        
        # Gradient computation
        # d(obj)/dx = d(log(smooth_max))/dx - d(log(h2_sq))/dx
        # = (1/smooth_max) * d(smooth_max)/dx - (1/h2_sq) * d(h2_sq)/dx
        
        # d(smooth_max)/dx = (1/beta) * d(log_sum_exp)/dx
        # d(log_sum_exp)/dx_j = sum_k weights_k * d(conv_k)/dx_j
        # where weights_k = exp(beta*(conv_k - max_c)) / sum_exp
        
        # d(conv_k)/dx_j = d(conv_k)/dh_j * dh_j/dx_j
        # h_j = exp(x_j) / sum(exp(x)) = softmax(x)_j
        # dh_j/dx_i = h_j * (delta_{ij} - h_i)
        
        # d(conv_k)/dh_j = 2 * h_{k-j} (as derived before, with h_m=0 for m out of range)
        
        # So d(conv_k)/dx_i = 2 * h_{k-i} * h_i * (delta_{ij} - h_j) ... this is getting complex.
        
        # Let's compute gradient numerically for reliability, or use a simpler approach.
        
        # Alternative: use the fact that we're on the simplex and use projected gradient.
        # For now, let's use a simpler objective: minimize smooth_max subject to sum(h)=1, h>=0
        # and then check the ratio.
        
        # Actually, let's just minimize smooth_max directly (since h2_sq is roughly constant 
        # for a given support size when h is spread out, and we want to reduce max conv).
        
        # Recompute with simpler objective: minimize smooth_max
        obj = smooth_max
        
        # Gradient of smooth_max w.r.t x
        # d(smooth_max)/dx_i = (1/beta) * sum_k weights_k * d(conv_k)/dx_i
        # d(conv_k)/dx_i = sum_j d(conv_k)/dh_j * dh_j/dx_i
        # = sum_j 2*h_{k-j} * h_j * (delta_{ji} - h_i)
        # = 2*h_{k-i}*h_i - 2*h_i * sum_j h_{k-j}*h_j
        # = 2*h_{k-i}*h_i - 2*h_i*conv_k
        
        # So d(smooth_max)/dx_i = (1/beta) * sum_k weights_k * (2*h_{k-i}*h_i - 2*h_i*conv_k)
        # = (2*h_i/beta) * sum_k weights_k * (h_{k-i} - conv_k)
        
        weights_k = exp_shifted / sum_exp
        
        # For each i in support, compute:
        # grad_x[i] = (2*h_i/beta) * sum_k weights_k * (h_{k-i} - conv_k)
        
        grad_x = np.zeros(k)
        for idx_i, i in enumerate(support):
            hi = h_support[idx_i]
            if hi < 1e-15:
                continue
            # sum_k weights_k * (h_{k-i} - conv_k)
            # h_{k-i} is h at index (k-i) if in range, else 0
            # We need to sum over k in [0, 2N-2]
            s = 0.0
            for k in range(2 * N - 1):
                wk = weights_k[k]
                if wk < 1e-15:
                    continue
                # h_{k-i}: index k-i, valid if 0 <= k-i < N
                h_ki = h[k - i] if 0 <= k - i < N else 0.0
                s += wk * (h_ki - conv[k])
            grad_x[idx_i] = (2.0 * hi / beta) * s
        
        return obj, grad_x
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        x_opt = res.x
        h_support = softmax(x_opt)
        
        h_opt = np.zeros(N)
        h_opt[support] = h_support
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception as e:
        # Fallback to uniform
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        sc = compute_ratio(h_opt)
        return h_opt, sc

def local_search_swap(h, indices, max_swaps=50, deadline=None):
    """Try swapping elements in the support to improve ratio."""
    best_h = h.copy()
    best_score = compute_ratio(h)
    current_indices = list(indices)
    support_set = set(current_indices)
    
    for swap_iter in range(max_swaps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(len(current_indices)):
            for j in range(N):
                if j in support_set:
                    continue
                new_indices = current_indices.copy()
                new_indices[i] = j
                new_indices = sorted(new_indices)
                
                if not is_sidon_set(new_indices):
                    continue
                
                new_h = np.zeros(N)
                for idx in new_indices:
                    new_h[idx] = 1.0
                sc = compute_ratio(new_h)
                if sc > best_score + 1e-12:
                    best_score = sc
                    best_h = new_h
                    current_indices = new_indices
                    support_set = set(current_indices)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin (evaluator kills at 150s)

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending - larger Sidon sets can have better ratios
    sidon_sets.sort(key=lambda s: len(s), reverse=True)
    
    for s in sidon_sets:
        if time.time() > deadline - 15.0:
            break
        
        # Try optimizing weights for this Sidon set
        h_opt, sc = optimize_weights_for_support(s, beta=50.0, maxiter=200)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
        
        # Try with higher beta for sharper max approximation
        if time.time() > deadline - 10.0:
            break
        h_opt2, sc2 = optimize_weights_for_support(s, beta=100.0, maxiter=200)
        if sc2 > best_score:
            best_score = sc2
            best_h = h_opt2.copy()
        
        # Local search: try swapping support elements
        if time.time() > deadline - 5.0:
            break
        indices = [i for i in range(N) if best_h[i] > 1e-10]
        if indices:
            h_ls, sc_ls = local_search_swap(best_h, indices, max_swaps=30, deadline=deadline - 2.0)
            if sc_ls > best_score:
                best_score = sc_ls
                best_h = h_ls.copy()

    return list(best_h)

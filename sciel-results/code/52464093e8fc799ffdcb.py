import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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

def optimize_weights_for_support(support, beta=100.0):
    """
    Optimize weights h_i for i in support to minimize max convolution coefficient.
    Uses smooth max surrogate with proper gradient handling on the simplex.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights
    x0 = np.ones(k) / k
    
    def objective_and_grad(x):
        """Compute smooth max of convolution and its gradient w.r.t x (on simplex)."""
        h = np.zeros(N)
        h[support] = x
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        weights = exp_shifted / sum_exp
        
        # Gradient w.r.t h: grad_h[j] = 2 * sum_{m in support} h[m] * weights[m+j]
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                grad_h += 2.0 * hm * weights[m : m+N]
        
        # Since we're on the simplex (sum(x) = 1), the gradient w.r.t x is just grad_h[support]
        # projected to the tangent space, but for SLSQP with equality constraint, we can return the raw gradient
        grad_x = grad_h[support]
        return obj, grad_x

    def objective(x):
        obj, _ = objective_and_grad(x)
        return obj

    def jac(x):
        _, grad = objective_and_grad(x)
        return grad

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
    ]
    bounds = [(0.0, 1.0) for _ in range(k)]
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try multiple initializations
    initializations = [x0.copy()]
    # Concentrated on first few elements
    if k >= 3:
        init2 = np.zeros(k)
        init2[0] = 0.5
        init2[1] = 0.3
        init2[2] = 0.2
        initializations.append(init2)
    # Geometric decay
    if k >= 4:
        init3 = np.array([np.exp(-0.5 * i) for i in range(k)])
        init3 = init3 / init3.sum()
        initializations.append(init3)
    
    for x_init in initializations:
        try:
            res = minimize(objective, x_init, jac=jac, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
            x_cand = res.x
            x_cand = np.maximum(x_cand, 0.0)
            s = np.sum(x_cand)
            if s > 1e-15:
                x_cand = x_cand / s
            obj_cand = objective(x_cand)
            if obj_cand < best_obj:
                best_obj = obj_cand
                best_x = x_cand.copy()
        except Exception:
            pass
    
    h_opt = np.zeros(N)
    h_opt[support] = best_x
    sc = compute_ratio(h_opt)
    return h_opt, sc

def local_search_polish(h, support, max_iter=200, tol=1e-12):
    """Local search: try small perturbations to improve the ratio."""
    k = len(support)
    current_h = h.copy()
    current_score = compute_ratio(current_h)
    
    for _ in range(max_iter):
        improved = False
        # Try perturbing pairs of weights
        for i in range(k):
            for j in range(i+1, k):
                # Try moving weight from i to j and vice versa
                for eps in [0.01, 0.005, 0.001]:
                    h_test = current_h.copy()
                    h_test[support[i]] = current_h[support[i]] - eps
                    h_test[support[j]] = current_h[support[j]] + eps
                    if h_test[support[i]] >= 0 and h_test[support[j]] >= 0:
                        sc = compute_ratio(h_test)
                        if sc > current_score + tol:
                            current_h = h_test
                            current_score = sc
                            improved = True
                            break
                    # Try moving weight from j to i
                    h_test = current_h.copy()
                    h_test[support[i]] = current_h[support[i]] + eps
                    h_test[support[j]] = current_h[support[j]] - eps
                    if h_test[support[i]] >= 0 and h_test[support[j]] >= 0:
                        sc = compute_ratio(h_test)
                        if sc > current_score + tol:
                            current_h = h_test
                            current_score = sc
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return current_h, current_score

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending? Larger sets might have better ratios?
    # Actually, for a fixed support, the ratio is roughly k / (max conv).
    # Uniform weights on Sidon set of size k: max conv is 1/k (if no collisions, which is true for Sidon).
    # So ratio = (1/k) / (1/k) = 1?
    # Wait. If h is uniform on Sidon set S of size k:
    # h_i = 1/k.
    # h2_sq = sum (1/k)^2 = k * (1/k^2) = 1/k.
    # conv_k = sum_i h_i h_{k-i}.
    # For Sidon set, each pair sum is unique.
    # The maximum convolution coefficient is 1/k (for the sums that appear once).
    # Wait, if all sums are distinct, then each conv_k is either 0 or 1/k.
    # So max conv = 1/k.
    # Ratio = (1/k) / (1/k) = 1.
    # So uniform weights on Sidon set give ratio 1.
    # We need to optimize weights to get ratio > 1.
    # This means we want to make the convolution more uniform (lower max) or make h2_sq larger.
    # Making h2_sq larger means concentrating weights.
    # But concentrating weights increases max conv.
    # There is a trade-off.
    
    # Let's try optimizing weights for each Sidon set.
    for s in sidon_sets:
        if time.time() > deadline - 15.0:
            break
        
        h_opt, sc = optimize_weights_for_support(s, beta=50.0)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
            
        # Try with different beta
        if time.time() > deadline - 15.0:
            break
        h_opt2, sc2 = optimize_weights_for_support(s, beta=100.0)
        if sc2 > best_score:
            best_score = sc2
            best_h = h_opt2.copy()
        
        # Local search polish
        if time.time() > deadline - 15.0:
            break
        h_polished, sc_polished = local_search_polish(best_h, s, max_iter=100, tol=1e-13)
        if sc_polished > best_score:
            best_score = sc_polished
            best_h = h_polished.copy()

    return list(best_h)

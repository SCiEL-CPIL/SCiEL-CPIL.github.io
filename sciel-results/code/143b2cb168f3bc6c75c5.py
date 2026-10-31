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
    for _ in range(8000):
        k = np.random.randint(4, 12)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 80:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0):
    """
    Optimize weights h_i for i in support to maximize ratio ||h||_2^2 / max_offpeak(h*h).
    Uses smooth max surrogate for the off-peak convolution terms with analytic gradient.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad_z(z):
        """Objective and gradient w.r.t. z (log-space parameterization)."""
        x = softmax(z)
        h = np.zeros(N)
        h[support] = x
        
        conv = fftconvolve(h, h, mode='full')
        # Exclude lag 0 (index N-1)
        conv_offpeak = np.delete(conv, N-1)
        
        max_c = np.max(conv_offpeak)
        if max_c < 1e-15:
            max_c = 1e-15
        
        shifted = conv_offpeak - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        h2_sq = np.sum(h**2)
        if h2_sq < 1e-15:
            h2_sq = 1e-15
            
        # Objective: smooth_max / h2_sq (we want to minimize this)
        # log(smooth_max) - log(h2_sq)
        obj_val = np.log(max_c) + np.log(sum_exp)/beta - np.log(h2_sq)
        
        # Gradient w.r.t. h:
        # d/dh [smooth_max] = sum_m weights_m * (h_{j+m-lag0} + h_{j-m+lag0}) for appropriate indices
        # d/dh [-log(h2_sq)] = -2*h / h2_sq
        
        # Build full weight array for conv (with lag 0 excluded)
        w_full = np.zeros(len(conv))
        # conv indices: 0..N-2 correspond to lags -(N-1)..-1
        # conv index N-1 is lag 0 (excluded)
        # conv indices N..2N-2 correspond to lags 1..N-1
        # conv_offpeak: first N-1 elements are conv[0..N-2], last N-1 are conv[N..2N-2]
        w_full[:N-1] = weights[:N-1]
        w_full[N:] = weights[N-1:]
        # w_full[N-1] = 0 (excluded)
        
        # grad_h[j] = sum_m w_full[m] * (h[j + (m-(N-1))] + h[j - (m-(N-1))]) - 2*h[j]/h2_sq
        # Only sum over m where w_full[m] != 0
        nonzero_m = np.where(w_full > 1e-15)[0]
        
        grad_h = np.zeros(N)
        for m in nonzero_m:
            lag = m - (N - 1)
            w_m = w_full[m]
            for j in range(N):
                j_plus = j + lag
                j_minus = j - lag
                if 0 <= j_plus < N:
                    grad_h[j] += w_m * h[j_plus]
                if 0 <= j_minus < N:
                    grad_h[j] += w_m * h[j_minus]
        
        grad_h -= 2.0 * h / h2_sq
        
        # Gradient w.r.t. z: grad_z[j] = x[j] * (grad_h[support[j]] - sum_i x[i]*grad_h[support[i]])
        gh_support = grad_h[support]
        sum_x_gh = np.sum(x * gh_support)
        grad_z = x * (gh_support - sum_x_gh)
        
        return obj_val, grad_z
    
    # Try multiple initializations
    best_h = None
    best_score = -np.inf
    
    n_init = 5
    np.random.seed(42 + k * 7)
    for init_i in range(n_init):
        if init_i == 0:
            z0 = np.zeros(k)  # uniform
        else:
            z0 = np.random.randn(k) * 0.5
        
        try:
            res = minimize(objective_and_grad_z, z0, jac=True, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            z_opt = res.x
            x_opt = softmax(z_opt)
            x_opt = np.maximum(x_opt, 0.0)
            x_opt = x_opt / np.sum(x_opt)
            
            h_opt = np.zeros(N)
            h_opt[support] = x_opt
            sc = compute_ratio(h_opt)
            
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
        except:
            pass
    
    if best_h is None:
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        best_score = compute_ratio(h_opt)
        best_h = h_opt
    
    return best_h, best_score

def local_search_polish(h, support, max_iter=200, tol=1e-10):
    """Local search: try small perturbations to improve the ratio."""
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    
    best_score = compute_ratio(h)
    
    for it in range(max_iter):
        improved = False
        for i in range(k):
            for j in range(i+1, k):
                for delta in [0.01, 0.005, 0.002, 0.001, -0.01, -0.005, -0.002, -0.001]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    hj = h[support[j]]
                    new_hi = hi + delta
                    new_hj = hj - delta
                    if new_hi >= 0 and new_hj >= 0:
                        h_new[support[i]] = new_hi
                        h_new[support[j]] = new_hj
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            for i in range(k):
                for delta in [0.01, 0.005, 0.002, 0.001, -0.01, -0.005, -0.002, -0.001]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    new_hi = hi + delta
                    if new_hi >= 0:
                        h_new[support[i]] = new_hi
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            
            if not improved:
                break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending
    sidon_sets.sort(key=lambda s: len(s), reverse=True)
    
    # Try optimizing weights for each Sidon set
    for s in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        
        # Try multiple beta values
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 30.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
        
        # Local polish
        if time.time() < deadline - 15.0:
            h_polished = local_search_polish(best_h, s, max_iter=100)
            sc_polished = compute_ratio(h_polished)
            if sc_polished > best_score:
                best_score = sc_polished
                best_h = h_polished

    # Final polish on the best solution
    if time.time() < deadline - 10.0:
        support = [i for i in range(N) if best_h[i] > 1e-10]
        if len(support) >= 2:
            best_h = local_search_polish(best_h, support, max_iter=300, tol=1e-12)
            best_score = compute_ratio(best_h)
        
        # Additional refinement with tighter local search
        if time.time() < deadline - 5.0:
            support = [i for i in range(N) if best_h[i] > 1e-10]
            if len(support) >= 2:
                h_refined = local_search_polish(best_h, support, max_iter=500, tol=1e-13)
                sc_refined = compute_ratio(h_refined)
                if sc_refined > best_score:
                    best_score = sc_refined
                    best_h = h_refined

    return list(best_h)

import numpy as np
import time
from scipy.optimize import minimize

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _optimize_epigraph(h_init, max_iter=500, ftol=1e-8):
    """
    Optimize h to minimize the peak of the autoconvolution, 
    subject to sum(h)=1 and h>=0.
    Variables: x = [h_0, ..., h_{N-1}, M]
    Objective: Minimize M
    Constraints:
      1. sum(h) - 1 = 0
      2. M - (h*h)_k >= 0 for all k
    """
    def obj(x):
        return x[N]

    def obj_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0

    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        # Clip h to 0 to handle numerical noise during optimization
        h_clip = np.clip(h, 0, None)
        c = np.convolve(h_clip, h_clip, mode="full")
        return M - c

    def constraints_autoconv_grad(x):
        h = x[:N]
        # Gradient of (M - sum(h_i h_{k-i})) w.r.t h_i
        # d/dh_i (h*h)_k = sum_j delta_{k, i+j} h_j + sum_j delta_{k, j+i} h_j
        # Since h*h is symmetric, (h*h)_k = sum_i h_i h_{k-i}.
        # d/dh_i = h_{k-i} + h_{k-i} = 2 h_{k-i} if i != k-i?
        # Actually, (h*h)_k = sum_{i=0}^{N-1} h_i h_{k-i} (with h_j=0 if j out of bounds).
        # d/dh_i (h*h)_k = h_{k-i} + h_{k-i} = 2 h_{k-i} if we treat the sum carefully?
        # Let's just compute the gradient numerically or analytically.
        # Analytical: 
        # c_k = sum_{i} h_i h_{k-i}
        # dc_k/dh_i = h_{k-i} + (if i == k-i then h_i else 0? No)
        # The term h_i h_{k-i} appears. If we vary h_i, it affects the term where the first index is i (h_i h_{k-i}) and where the second index is i (h_{k-i} h_i).
        # So derivative is 2 * h_{k-i} if k-i is in bounds and i != k-i?
        # If i == k-i, the term is h_i^2, derivative is 2 h_i.
        # So in all cases, derivative is 2 * h_{k-i} (with h_{k-i}=0 if out of bounds).
        
        grad = np.zeros((2 * N - 1, N + 1))
        for k in range(2 * N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs  # We want M - c_k >= 0, so grad of (M - c_k) is -grad(c_k)
            grad[k, N] = 1.0       # grad of M w.r.t M is 1
        return grad

    h_init = np.asarray(h_init, dtype=np.float64)
    s = np.sum(h_init)
    if s <= 0:
        h_init = np.ones(N) / N
        s = 1.0
    h_init = h_init / s
    
    c_init = np.convolve(h_init, h_init, mode="full")
    M_init = np.max(c_init)
    x0 = np.concatenate([h_init, [M_init]])
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]
    
    try:
        res = minimize(
            obj, 
            x0, 
            method='SLSQP', 
            jac=obj_grad,
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': ftol}
        )
        
        if res.success or res.fun < M_init:
            h_res = res.x[:N]
            h_res = np.clip(h_res, 0, None)
            s = np.sum(h_res)
            if s > 0:
                h_res = h_res / s
            else:
                return None, 0.0
                
            ratio_res = _ratio(h_res)
            return h_res, ratio_res
    except Exception:
        pass
    return None, 0.0

def _find_sidon_support(target_size, N):
    """
    Greedy construction of a Sidon set.
    """
    if target_size <= 0:
        return []
    
    support = [0]
    used_sums = {0}
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_score = -1
        
        for candidate in range(N):
            if candidate in support:
                continue
            
            # Count collisions
            collisions = 0
            for s in support:
                if candidate + s in used_sums:
                    collisions += 1
            
            # We want to minimize collisions
            score = -collisions
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is not None:
            support.append(best_candidate)
            for s in support:
                used_sums.add(best_candidate + s)
        
        if best_candidate is None:
            break
            
    return support

def construct():
    np.random.seed(12345)
    
    best_h = None
    best_ratio = -np.inf

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = _optimize_epigraph(h_uniform, max_iter=300)
    if h_opt is not None and s_opt > best_ratio:
        best_ratio = s_opt
        best_h = h_opt.copy()

    # 2. Try Sidon-like structured supports
    # Smaller Sidon sets might allow for "purer" structures
    for target_k in [5, 6, 7, 8, 10, 12]:
        if _time_left() < 15.0:
            break
        
        support = _find_sidon_support(target_k, N)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            
            # Optimize from this start
            h_opt, s_opt = _optimize_epigraph(h_sidon, max_iter=300)
            if h_opt is not None and s_opt > best_ratio:
                best_ratio = s_opt
                best_h = h_opt.copy()

    # 3. Try random sparse starts with varying support sizes
    # This helps escape local optima of the structured starts
    num_restarts = 20
    for i in range(num_restarts):
        if _time_left() < 10.0:
            break
        
        K = np.random.randint(3, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = _optimize_epigraph(h_rand, max_iter=200)
        if h_opt is not None and s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt.copy()

    # 4. Polish the best solution found
    # Use tighter tolerances and more iterations on the best candidate
    if best_h is not None and _time_left() > 20.0:
        # Zero out very small values to potentially simplify the problem
        h_polish_init = best_h.copy()
        h_polish_init[h_polish_init < 1e-6] = 0.0
        s = np.sum(h_polish_init)
        if s > 0:
            h_polish_init /= s
        
        h_polished, s_polished = _optimize_epigraph(h_polish_init, max_iter=1000, ftol=1e-10)
        if h_polished is not None and s_polished > best_ratio:
            best_ratio = s_polished
            best_h = h_polished.copy()

    # 5. If we have time, try a few more random restarts with higher K
    while _time_left() > 10.0:
        K = np.random.randint(5, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = _optimize_epigraph(h_rand, max_iter=100)
        if h_opt is not None and s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)

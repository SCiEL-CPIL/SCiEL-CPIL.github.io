"""
Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm times sup norm of autoconvolution.

Strategy:
1. Normalize L1=1 (ratio is scale-invariant).
2. The problem is equivalent to minimizing M = max(h*h) subject to sum(h)=1, h>=0.
3. Use a hybrid approach:
   a. Generate sparse supports (Sidon-like sets) that minimize pairwise sum collisions.
   b. Optimize weights on these fixed supports using SLSQP (low dimensional, fast).
   c. Use the best sparse solution as a seed for full-space SLSQP to allow support growth/refinement.
   d. Final polish with tight tolerances.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconv(h):
    """Linear autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _score_raw(h):
    """Raw score: L2^2 / (L1 * max(autoconv)). Note: delta cancels out in the ratio if consistent, 
    but the task asks for L2^2 / (L1 * sup(autoconv)). 
    Wait, the task says: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution".
    Usually L2 norm is sqrt(sum h^2). So L2^2 = sum h^2.
    Score = sum(h^2) / (sum(h) * max(conv(h,h))).
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    c = _autoconv(h)
    max_c = np.max(c)
    if max_c <= 0:
        return 0.0
    return l2_sq / (l1 * max_c)

def _solve_fixed_support(h_init, support, maxiter=500, ftol=1e-10):
    """
    Optimize weights on a fixed support.
    Variables: w (length K)
    h_i = w_j if i in support[j], else 0.
    Minimize M subject to sum(w)=1, M >= (h*h)_k, w>=0.
    """
    K = len(support)
    if K == 0:
        return None
    
    # Initial weights: uniform
    w0 = np.ones(K) / K
    
    # Map from index in support to index in full vector
    # We need to compute autoconv of h given w.
    # h = zeros(N); h[support] = w
    
    def obj(x):
        return x[K] # Minimize M
    
    def obj_grad(x):
        g = np.zeros(K + 1)
        g[K] = 1.0
        return g
    
    def cons_eq(x):
        return np.sum(x[:K]) - 1.0
    
    def cons_eq_grad(x):
        g = np.zeros(K + 1)
        g[:K] = 1.0
        return g
    
    def cons_ineq(x):
        w = x[:K]
        M = x[K]
        # Construct h
        h = np.zeros(N)
        h[support] = w
        c = _autoconv(h)
        return M - c
    
    def cons_ineq_grad(x):
        w = x[:K]
        M = x[K]
        h = np.zeros(N)
        h[support] = w
        
        grad = np.zeros((2*N - 1, K + 1))
        for k in range(2*N - 1):
            for j_idx in range(K):
                i = support[j_idx]
                # d(c_k)/dw_j = d(c_k)/dh_i = 2 * h_{k-i}
                j = k - i
                if 0 <= j < N:
                    # Check if j is in support? No, h_{k-i} is just the value at k-i.
                    # If k-i is in support, it corresponds to some w_m.
                    # But we are differentiating with respect to w_j.
                    # The term in c_k involving h_i is h_i * h_{k-i}.
                    # Derivative wrt w_j (which is h_i) is h_{k-i}.
                    # Also the term h_{k-i} * h_i. Derivative is h_i.
                    # So total derivative is 2 * h_{k-i}.
                    # We need h_{k-i}.
                    # If k-i is in support, h_{k-i} is some w_m.
                    # If not, it's 0.
                    # So we just use the value h[k-i].
                    grad[k, j_idx] = -2.0 * h[j]
            grad[k, K] = 1.0
        return grad

    bounds = [(0.0, None)] * K + [(0.0, None)]
    
    constraints = [
        {'type': 'eq', 'fun': cons_eq, 'jac': cons_eq_grad},
        {'type': 'ineq', 'fun': cons_ineq, 'jac': cons_ineq_grad}
    ]
    
    # Initial M
    h0 = np.zeros(N)
    h0[support] = w0
    c0 = _autoconv(h0)
    M0 = np.max(c0) + 1e-6
    
    x0 = np.concatenate([w0, [M0]])
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', jac=obj_grad,
            bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        
        w_res = np.maximum(0, res.x[:K])
        s_res = np.sum(w_res)
        if s_res > 1e-12:
            w_res = w_res / s_res
            h_res = np.zeros(N)
            h_res[support] = w_res
            return h_res
        return None
    except Exception:
        return None

def _solve_full_slsqp(h_init, maxiter=500, ftol=1e-10):
    """
    Solve full problem using SLSQP.
    """
    h_init = np.asarray(h_init, dtype=np.float64)
    s = np.sum(h_init)
    if s <= 0:
        return None
    h_init = h_init / s
    
    c_init = _autoconv(h_init)
    M_init = np.max(c_init) + 1e-6
    
    x0 = np.concatenate([h_init, [M_init]])
    
    def obj(x):
        return x[N]
    
    def obj_grad(x):
        g = np.zeros(N + 1)
        g[N] = 1.0
        return g
    
    def cons_eq(x):
        return np.sum(x[:N]) - 1.0
    
    def cons_eq_grad(x):
        g = np.zeros(N + 1)
        g[:N] = 1.0
        return g
    
    def cons_ineq(x):
        h = x[:N]
        M = x[N]
        c = _autoconv(h)
        return M - c
    
    def cons_ineq_grad(x):
        h = x[:N]
        grad = np.zeros((2*N - 1, N + 1))
        for k in range(2*N - 1):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    grad[k, i] = -2.0 * h[j]
            grad[k, N] = 1.0
        return grad
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    constraints = [
        {'type': 'eq', 'fun': cons_eq, 'jac': cons_eq_grad},
        {'type': 'ineq', 'fun': cons_ineq, 'jac': cons_ineq_grad}
    ]
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', jac=obj_grad,
            bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        h_res = np.maximum(0, res.x[:N])
        s_res = np.sum(h_res)
        if s_res > 1e-12:
            h_res = h_res / s_res
            return h_res
        return None
    except Exception:
        return None

def _find_sidon_support(target_size):
    """
    Greedy construction of a set with minimal pairwise sum collisions.
    """
    if target_size <= 0:
        return []
    
    support = [0]
    used_sums = {0: 1}
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_score = -1
        
        for candidate in range(N):
            if candidate in support:
                continue
            
            # Count new collisions
            new_collisions = 0
            for s in support:
                pair_sum = candidate + s
                if pair_sum in used_sums:
                    new_collisions += 1
            
            # We want to minimize collisions
            score = -new_collisions
            if score > best_score:
                best_score = score
                best_candidate = candidate
            
        if best_candidate is not None:
            support.append(best_candidate)
            for s in support:
                pair_sum = best_candidate + s
                used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
        else:
            break
            
    return support

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    def update_best(h):
        nonlocal best_h, best_score
        if h is None:
            return
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(0, h)
        s = np.sum(h)
        if s <= 0:
            return
        h = h / s
        sc = _score_raw(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
            
    # Phase 1: Sparse Support Search
    # Try various support sizes
    # Sidon sets are good starts
    for K in range(2, 16):
        if time.time() > deadline - 40:
            break
            
        # Generate a few Sidon-like supports
        # Use different random seeds or starting points to get variety
        for trial in range(5):
            if time.time() > deadline - 40:
                break
                
            # Perturb the greedy choice slightly to find different local minima
            # Simple approach: just use the standard greedy, but maybe start with different first elements
            support = _find_sidon_support(K)
            if len(support) < K:
                continue
                
            h_opt = _solve_fixed_support(best_h, support, maxiter=300, ftol=1e-8)
            if h_opt is not None:
                update_best(h_opt)
                
    # Phase 2: Random Sparse Supports
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        K = np.random.randint(3, 15)
        support = np.random.choice(N, K, replace=False)
        h_opt = _solve_fixed_support(best_h, support, maxiter=200, ftol=1e-8)
        if h_opt is not None:
            update_best(h_opt)
            
    # Phase 3: Full Space Refinement from Best Sparse Solution
    # The best solution found so far might have a support that is optimal, 
    # but allowing new elements might improve it further.
    if time.time() < deadline - 10:
        h_full = _solve_full_slsqp(best_h, maxiter=500, ftol=1e-10)
        if h_full is not None:
            update_best(h_full)
            
    # Phase 4: Local Search / Perturbation
    # Try perturbing the best solution and re-optimizing on its support
    support_best = np.where(best_h > 1e-6)[0]
    if len(support_best) > 0 and time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            # Randomly add a new element to support
            if len(support_best) < N:
                new_idx = np.random.choice([i for i in range(N) if i not in support_best])
                new_support = list(support_best) + [new_idx]
            else:
                new_support = list(support_best)
                
            h_opt = _solve_fixed_support(best_h, new_support, maxiter=200, ftol=1e-8)
            if h_opt is not None:
                update_best(h_opt)
                # Update support if improved
                new_support_best = np.where(best_h > 1e-6)[0]
                if len(new_support_best) != len(support_best):
                    support_best = new_support_best
                    
    # Final Polish
    if best_h is not None and time.time() < deadline - 1:
        h_polished = _solve_full_slsqp(best_h, maxiter=1000, ftol=1e-12)
        if h_polished is not None:
            update_best(h_polished)
            
    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return list(best_h)

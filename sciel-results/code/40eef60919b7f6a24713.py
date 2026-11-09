import numpy as np
import time
import scipy.optimize as opt

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    timeout = 600  # seconds
    safety_margin = 30  # seconds before timeout

    def _time_ok():
        return (time.time() - start_time) < (timeout - safety_margin)

    def compute_P(support, weights):
        """Compute P(X1+X2+X3 < 2*X4) efficiently."""
        s = np.asarray(support, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        n = len(s)
        
        # P = sum_{i,j,k,l} w_i w_j w_k w_l * I(s_i + s_j + s_k < 2*s_l)
        # = sum_l w_l * [sum_{i,j,k} w_i w_j w_k * I(s_i + s_j + s_k < 2*s_l)]
        
        # Precompute all pairwise sums s_i + s_j
        s_ij = s[:, None] + s[None, :]  # (n, n)
        w_ij = w[:, None] * w[None, :]  # (n, n)
        
        # For each l, we need sum over i,j,k of w_i w_j w_k * I(s_ij + s_k < 2*s_l)
        # = sum_{ij} w_ij * [sum_k w_k * I(s_k < 2*s_l - s_ij)]
        
        P_val = 0.0
        for l in range(n):
            threshold = 2.0 * s[l]
            # For each (i,j), compute sum_k w_k * I(s_k < threshold - s_ij)
            # This is a CDF lookup for each (i,j)
            # s_ij shape: (n, n), threshold - s_ij shape: (n, n)
            # For each element, we need the CDF of the discrete distribution at that value
            
            # Vectorized: for each (i,j), find how much weight is below threshold - s_ij
            # Use searchsorted on sorted support
            targets = threshold - s_ij  # (n, n)
            
            # Sort support and compute cumulative weights
            sort_idx = np.argsort(s)
            s_sorted = s[sort_idx]
            w_sorted = w[sort_idx]
            cumsum_w = np.cumsum(w_sorted)
            
            # searchsorted gives the index where each target would be inserted
            # We want sum of w_k for s_k < target (strict inequality)
            # searchsorted with side='left' gives first index where s_sorted[idx] >= target
            # So we want cumsum_w at index-1, or 0 if index==0
            idx = np.searchsorted(s_sorted, targets, side='left')
            # idx has shape (n, n)
            # CDF value: if idx > 0, cumsum_w[idx-1], else 0
            cdf_vals = np.where(idx > 0, cumsum_w[np.maximum(idx - 1, 0)], 0.0)
            
            # Now sum over (i,j): w_ij * cdf_vals
            inner_sum = np.sum(w_ij * cdf_vals)
            P_val += w[l] * inner_sum
        
        return P_val

    def objective(params):
        """Objective function for optimization. params = [weights..., support...]"""
        n_w = len(params) // 2
        n_s = len(params) - n_w
        w = params[:n_w]
        s = params[n_w:]
        
        # Ensure weights are positive
        w = np.maximum(w, 1e-10)
        w = w / np.sum(w)
        
        # Ensure support is non-negative
        s = np.maximum(s, 0.0)
        
        return -compute_P(s, w)

    def optimize_weights_only(support, w0, max_iter=500):
        """Optimize weights with fixed support using SLSQP."""
        n = len(support)
        
        def obj_w(w):
            w = np.maximum(w, 1e-10)
            w = w / np.sum(w)
            return -compute_P(support, w)
        
        def grad_w(w):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(n)
            w_base = np.maximum(w, 1e-10)
            w_base = w_base / np.sum(w_base)
            f0 = obj_w(w_base)
            for i in range(n):
                w_pert = w_base.copy()
                w_pert[i] += eps
                w_pert = np.maximum(w_pert, 1e-10)
                w_pert = w_pert / np.sum(w_pert)
                grad[i] = (obj_w(w_pert) - f0) / eps
            return grad
        
        res = opt.minimize(obj_w, w0, method='SLSQP', 
                          bounds=[(1e-8, None)] * n,
                          constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                          options={'maxiter': max_iter, 'ftol': 1e-10})
        
        w_opt = np.maximum(res.x, 0)
        w_opt = w_opt / np.sum(w_opt)
        return w_opt

    def optimize_joint(s0, w0, max_iter=300):
        """Jointly optimize support and weights."""
        n = len(s0)
        
        # Parameterize: weights (n) and support (n)
        # Total params: 2n
        params0 = np.concatenate([w0, s0])
        
        res = opt.minimize(objective, params0, method='SLSQP',
                          bounds=[(1e-8, None)] * (2 * n),
                          constraints=[{'type': 'eq', 'fun': lambda p: np.sum(p[:n]) - 1}],
                          options={'maxiter': max_iter, 'ftol': 1e-10})
        
        if res.success:
            w_opt = np.maximum(res.x[:n], 0)
            w_opt = w_opt / np.sum(w_opt)
            s_opt = np.maximum(res.x[n:], 0)
            return s_opt, w_opt
        else:
            return s0, w0

    def nelder_mead_search(s0, w0, max_iter=200):
        """Use Nelder-Mead for local search."""
        n = len(s0)
        params0 = np.concatenate([w0, s0])
        
        res = opt.minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        
        w_opt = np.maximum(res.x[:n], 0)
        w_opt = w_opt / np.sum(w_opt)
        s_opt = np.maximum(res.x[n:], 0)
        return s_opt, w_opt

    # Generate initial configurations
    configs = []
    
    # Config 1: Uniform grid [0, 10]
    N1 = 15
    s1 = np.linspace(0, 10, N1)
    w1 = np.ones(N1) / N1
    configs.append((s1, w1))
    
    # Config 2: Geometric spacing
    N2 = 15
    s2 = np.exp(np.linspace(0, 2, N2)) - 1
    w2 = np.ones(N2) / N2
    configs.append((s2, w2))
    
    # Config 3: Concentrated near 0 with spread
    N3 = 15
    s3 = np.linspace(0, 5, N3)
    w3 = np.exp(-s3)  # Exponential decay
    w3 = w3 / np.sum(w3)
    configs.append((s3, w3))
    
    # Config 4: Two-point heavy at 0
    N4 = 12
    s4 = np.array([0.0] * 6 + [5.0] * 6)
    w4 = np.array([0.4] * 6 + [0.1] * 6)
    w4 = w4 / np.sum(w4)
    configs.append((s4, w4))
    
    # Config 5: Log-spaced
    N5 = 15
    s5 = np.logspace(0, 1, N5) - 1  # [0, 9]
    w5 = np.ones(N5) / N5
    configs.append((s5, w5))
    
    # Config 6: Dense near 0
    N6 = 20
    s6 = np.linspace(0, 1, 10)
    s6 = np.concatenate([s6, np.linspace(2, 10, 10)])
    w6 = np.concatenate([np.ones(10) * 0.5, np.ones(10) * 0.05])
    w6 = w6 / np.sum(w6)
    configs.append((s6, w6))
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    for s0, w0 in configs:
        if not _time_ok():
            break
        
        # Step 1: Optimize weights only
        w_opt = optimize_weights_only(s0, w0, max_iter=300)
        P = compute_P(s0, w_opt)
        if P > best_P:
            best_P = P
            best_support = s0.copy()
            best_weights = w_opt.copy()
        
        if not _time_ok():
            break
        
        # Step 2: Joint optimization with SLSQP
        s_new, w_new = optimize_joint(s0, w_opt, max_iter=200)
        P_new = compute_P(s_new, w_new)
        if P_new > best_P:
            best_P = P_new
            best_support = s_new.copy()
            best_weights = w_new.copy()
        
        if not _time_ok():
            break
        
        # Step 3: Nelder-Mead refinement
        s_nm, w_nm = nelder_mead_search(s_new, w_new, max_iter=150)
        P_nm = compute_P(s_nm, w_nm)
        if P_nm > best_P:
            best_P = P_nm
            best_support = s_nm.copy()
            best_weights = w_nm.copy()
    
    # Final cleanup
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    else:
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        if np.sum(best_weights) <= 1e-9:
            best_weights = np.ones_like(best_weights)
        best_weights = best_weights / np.sum(best_weights)
    
    return best_support, best_weights

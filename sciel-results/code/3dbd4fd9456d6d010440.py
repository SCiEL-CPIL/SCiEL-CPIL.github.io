import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin before 120s limit
    
    def ratio(h):
        h = np.array(h)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        # Autoconvolution
        conv = np.convolve(h, h, mode='full')
        sup_conv = np.max(conv)
        if sup_conv == 0:
            return 0.0
        return l2_sq / (l1 * sup_conv)
    
    best_h = [2.0/3.0, 1.0/3.0] + [0.0] * (N - 2)
    best_val = ratio(best_h)
    
    # Try various support sizes and optimize
    for support_size in range(2, min(20, N + 1)):
        if time.time() > deadline:
            break
        
        # Try different initializations
        n_trials = 5
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Random initial support pattern
            if trial == 0:
                # Uniform on first support_size elements
                h0 = np.zeros(N)
                h0[:support_size] = 1.0 / support_size
            elif trial == 1:
                # Geometric-like decay
                h0 = np.zeros(N)
                weights = np.array([0.5**i for i in range(support_size)])
                h0[:support_size] = weights / weights.sum()
            elif trial == 2:
                # Two peaks
                h0 = np.zeros(N)
                mid = support_size // 2
                h0[0] = 0.4
                h0[mid] = 0.6
                h0 = h0 / h0.sum()
            else:
                # Random
                h0 = np.random.dirichlet(np.ones(support_size))
                h0 = np.concatenate([h0, np.zeros(N - support_size)])
            
            # Optimize using L-BFGS-B
            def neg_ratio(x):
                return -ratio(x)
            
            def jac_ratio(x):
                # Numerical gradient
                eps = 1e-6
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    grad[i] = (neg_ratio(x_plus) - neg_ratio(x_minus)) / (2 * eps)
                return grad
            
            result = minimize(neg_ratio, h0, method='L-BFGS-B', jac=jac_ratio,
                           bounds=[(0, 1)] * N,
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < -best_val:
                h_opt = result.x
                val = -result.fun
                if val > best_val:
                    best_val = val
                    best_h = h_opt.tolist()
    
    # Final cleanup: ensure nonnegative and valid
    best_h = [max(0.0, h) for h in best_h]
    total = sum(best_h)
    if total > 0:
        best_h = [h / total for h in best_h]
    
    return best_h
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize

def compute_prob(support, weights):
    w_sum = np.sum(weights)
    if w_sum < 1e-9:
        return 0.0
    w = weights / w_sum
    n = len(support)
    S = support[:, None] + support[None, :]
    T = 2.0 * support[None, :] - support[:, None]
    
    Q = np.zeros((n, n))
    for k in range(n):
        for l in range(n):
            mask = S < T[k, l]
            Q[k, l] = np.sum(w[:, None] * w[None, :] * mask)
            
    P = np.sum(w[:, None] * w[None, :] * Q)
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_support = None
    best_weights = None
    best_prob = -1.0
    
    for n in [3, 4]:
        # Initial guesses
        # Try a few random initializations
        for trial in range(5):
            if trial == 0:
                if n == 3:
                    s0 = np.array([0.0, 1.0, 3.0])
                    w0 = np.array([0.347, 0.253, 0.4])
                else:
                    s0 = np.array([0.0, 0.5, 1.0, 2.0])
                    w0 = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                s0 = np.sort(np.random.uniform(0, 5, n))
                w0 = np.random.dirichlet(np.ones(n))
            
            x0 = np.concatenate([s0, w0])
            
            def objective(x):
                s = x[:n]
                w = x[n:]
                if np.any(s < -1e-9) or np.any(w < -1e-9):
                    return 1.0 # Penalty
                if np.sum(w) < 1e-9:
                    return 1.0
                # Ensure distinct support? 
                # If points are too close, it's effectively a lower n distribution.
                # We don't strictly need to enforce distinctness for correctness,
                # but it helps optimization.
                p = compute_prob(s, w)
                return -p
            
            bounds = [(0, 10)] * n + [(0, 1)] * n
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
            
            if res.success or res.fun < -best_prob:
                s_opt = res.x[:n]
                w_opt = res.x[n:]
                p_opt = -res.fun
                if p_opt > best_prob:
                    best_prob = p_opt
                    best_support = s_opt
                    best_weights = w_opt
    
    # Clean up: merge very close support points if necessary?
    # The problem allows any support. If two are identical, it's fine.
    # But to be safe and clean, we can sort and check.
    if best_support is None:
         # Fallback to seed
         return np.array([0.0, 1.0, 3.0]), np.array([0.347, 0.253, 0.4])
         
    return best_support, best_weights

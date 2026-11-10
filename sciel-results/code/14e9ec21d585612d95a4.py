# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization using rank-1 (product of marginals)
    parameterization P[y,x] = R[y] * C[x].
    
    This reduces the 100-dimensional search to 20 dimensions (10 row weights + 10 col weights,
    minus normalization constraints = 18 free parameters).
    """
    n = 10
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Precompute grid indices
    x_idx = np.arange(n)
    y_idx = np.arange(n)
    
    # Precompute projection values for each slope
    # P[y, x] -> value = x + r*y
    proj_minus1 = x_idx[None, :] + (-1) * y_idx[:, None]  # x - y, shape (10,10)
    proj_0 = x_idx[None, :] + 0 * y_idx[:, None]          # x, shape (10,10)
    proj_1 = x_idx[None, :] + 1 * y_idx[:, None]          # x + y, shape (10,10)
    proj_2 = x_idx[None, :] + 2 * y_idx[:, None]          # x + 2y, shape (10,10)
    
    def compute_entropies_from_marginals(log_r, log_c):
        """Compute D and M from log-marginals."""
        r = np.exp(log_r)
        c = np.exp(log_c)
        P = np.outer(r, c)  # P[y, x] = r[y] * c[x]
        
        def entropy_of_projection(proj_vals):
            # Pool cells with equal projection value
            from collections import defaultdict
            sums = defaultdict(float)
            for y in range(n):
                for x in range(n):
                    v = proj_vals[y, x]
                    sums[v] += P[y, x]
            probs = np.array(list(sums.values()))
            probs = probs[probs > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        D = entropy_of_projection(proj_minus1)
        H0 = entropy_of_projection(proj_0)
        H1 = entropy_of_projection(proj_1)
        H2 = entropy_of_projection(proj_2)
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return -1e10
        return D / M
    
    def objective(params):
        """Objective function to maximize (we minimize negative)."""
        log_r = params[:n]
        log_c = params[n:]
        
        # Soft normalization: use softmax-like to ensure positivity
        r = np.exp(log_r - np.max(log_r))
        c = np.exp(log_c - np.max(log_c))
        
        # Normalize
        r = r / np.sum(r)
        c = c / np.sum(c)
        
        P = np.outer(r, c)
        
        def entropy_of_projection(proj_vals, P):
            sums = {}
            for y in range(n):
                for x in range(n):
                    v = int(proj_vals[y, x])
                    if v not in sums:
                        sums[v] = 0.0
                    sums[v] += P[y, x]
            probs = np.array(list(sums.values()))
            probs = probs[probs > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        D = entropy_of_projection(proj_minus1, P)
        H0 = entropy_of_projection(proj_0, P)
        H1 = entropy_of_projection(proj_1, P)
        H2 = entropy_of_projection(proj_2, P)
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return 1e10  # Large penalty (we're minimizing)
        return -(D / M)
    
    # Initial guess: uniform
    log_r0 = np.zeros(n)
    log_c0 = np.zeros(n)
    params0 = np.concatenate([log_r0, log_c0])
    
    # Try multiple random restarts
    best_score = -np.inf
    best_params = params0.copy()
    
    # First try from uniform
    result = minimize(objective, params0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    if result.fun < best_score:
        best_score = result.fun
        best_params = result.x
    
    # Random restarts
    rng = np.random.default_rng(42)
    n_restarts = 30
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        log_r_rand = rng.normal(0, 1, n)
        log_c_rand = rng.normal(0, 1, n)
        params_rand = np.concatenate([log_r_rand, log_c_rand])
        
        result = minimize(objective, params_rand, method='L-BFGS-B',
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    
    # Extract best solution
    log_r_best = best_params[:n]
    log_c_best = best_params[n:]
    
    r_best = np.exp(log_r_best - np.max(log_r_best))
    c_best = np.exp(log_c_best - np.max(log_c_best))
    r_best = r_best / np.sum(r_best)
    c_best = c_best / np.sum(c_best)
    
    P = np.outer(r_best, c_best)
    
    # Ensure validity
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        # Fallback to uniform
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total
    
    return P


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for discrete distribution.
    support: np.array of shape (n,)
    weights: np.array of shape (n,)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    n = len(s)
    # Precompute pairwise sums s_i + s_j and their probabilities w_i * w_j
    # We need sum of w_i w_j for s_i + s_j < T
    # It is faster to iterate k, l and check i, j
    # n is small (<= 10), so O(n^4) is fine.
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum w_i * w_j for i, j such that s_i + s_j < threshold
            inner_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if s[i] + s[j] < threshold:
                        inner_sum += w[i] * w[j]
            P += w[k] * w[l] * inner_sum
    return P

def optimize_measure(n, time_budget=60):
    """
    Optimize support and weights for a fixed n points using L-BFGS-B.
    Supports are in [0, 1], weights are non-negative and sum to 1.
    """
    if n < 2:
        return None, 0.0, 0.0
    
    # Parameters: log-weights (unconstrained) and support positions
    # Use softmax for weights to ensure positivity and sum to 1
    # Support: use cumulative sum of positive variables to ensure ordering
    
    def _compute_P_from_params(params):
        """Compute P from unconstrained parameters."""
        # Split params
        log_w = params[:n]  # n log-weight variables
        s_pos = params[n:]  # n-1 position variables (for n-2 internal points + 1 for scaling)
        
        # Convert log_w to weights via softmax
        log_w_shifted = log_w - np.max(log_w)  # for numerical stability
        exp_w = np.exp(log_w_shifted)
        w = exp_w / np.sum(exp_w)
        
        # Convert s_pos to sorted support in [0, 1]
        # Use cumulative sum of positive values
        if n == 2:
            s = np.array([0.0, 1.0])
        else:
            # s_pos has n-1 elements: first n-2 are gaps, last is scale
            gaps = np.abs(s_pos[:n-2]) + 1e-8  # positive gaps
            total_gap = np.sum(gaps)
            if total_gap < 1e-12:
                s = np.linspace(0, 1, n)
            else:
                s = np.zeros(n)
                s[0] = 0.0
                s[1:n] = np.cumsum(gaps) / total_gap
                # Ensure last point is exactly 1
                s[n-1] = 1.0
        
        return calculate_P(s, w), s, w
    
    def objective(params):
        p_val, s, w = _compute_P_from_params(params)
        return -p_val
    
    def objective_grad(params):
        """Compute gradient via finite differences for L-BFGS-B."""
        grad = np.zeros_like(params)
        eps = 1e-6
        f0 = objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (objective(params_plus) - f0) / eps
        return grad
    
    # Initial guess
    rng = np.random.RandomState(42)
    log_w_init = np.zeros(n)
    if n > 2:
        s_pos_init = rng.uniform(0.5, 2.0, size=n-1)
    else:
        s_pos_init = np.array([])
    
    params_init = np.concatenate([log_w_init, s_pos_init])
    
    try:
        from scipy.optimize import minimize
        
        # Use L-BFGS-B
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        if result.success or result.fun < 0:
            best_params = result.x
            p_val, s, w = _compute_P_from_params(best_params)
            if p_val > 0:
                return s, w, p_val
        
        # Try a few random restarts if time allows
        best_p = 0.0
        best_s = None
        best_w = None
        
        start_t = time.time()
        max_restarts = min(10, int((time_budget - (time.time() - start_t)) / 5))
        
        for restart in range(max_restarts):
            if time.time() - start_t > time_budget * 0.8:
                break
            
            log_w_r = rng.normal(0, 1, size=n)
            if n > 2:
                s_pos_r = rng.uniform(0.5, 2.0, size=n-1)
            else:
                s_pos_r = np.array([])
            
            params_r = np.concatenate([log_w_r, s_pos_r])
            
            result_r = minimize(
                objective,
                params_r,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result_r.success or result_r.fun < 0:
                p_r, s_r, w_r = _compute_P_from_params(result_r.x)
                if p_r > best_p:
                    best_p = p_r
                    best_s = s_r
                    best_w = w_r
        
        if best_s is not None and best_p > 0:
            return best_s, best_w, best_p
        elif p_val > 0:
            return s, w, p_val
        else:
            return None, 0.0, 0.0
            
    except Exception as e:
        return None, 0.0, 0.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 550  # 50s margin
    
    best_support = None
    best_weights = None
    best_p = 0.0
    
    # 1. Analytic n=2 solution
    p_at_1 = (1.0 + np.sqrt(33.0)) / 16.0
    p_at_0 = 1.0 - p_at_1
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_at_0, p_at_1])
    p2 = calculate_P(s2, w2)
    
    if p2 > best_p:
        best_p = p2
        best_support = s2.copy()
        best_weights = w2.copy()
        
    if time.time() > deadline:
        return best_support, best_weights

    # 2. Try various n values with the optimized calculator
    # With O(n^2 log n) P calculation, we can try larger n
    n_values = [3, 5, 8, 12, 20, 32, 50, 80, 128, 200, 300, 400, 512]
    
    for n in n_values:
        if time.time() > deadline - 30:
            break
        
        # Estimate time budget based on remaining time
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate more time to larger n values that might be better
        time_budget = min(remaining * 0.4, 60)
        
        res = optimize_measure(n, time_budget=time_budget)
        if res is not None and res[2] > best_p:
            best_p = res[2]
            best_support = res[0].copy()
            best_weights = res[1].copy()
    
    # 3. Also try some structured distributions
    # Beta-like distributions
    if time.time() < deadline - 10:
        for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
            for beta_val in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
                if time.time() > deadline - 5:
                    break
                # Sample from Beta distribution
                n_pts = 100
                x = np.linspace(0.01, 0.99, n_pts)
                # Beta PDF
                from scipy.special import beta as beta_func
                pdf = x**(alpha-1) * (1-x)**(beta_val-1) / beta_func(alpha, beta_val)
                w = pdf / np.sum(pdf)
                s = x
                p_val = calculate_P(s, w)
                if p_val > best_p:
                    best_p = p_val
                    best_support = s.copy()
                    best_weights = w.copy()
    
    # Final cleanup
    if best_support is None:
        # Fallback to uniform
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)/n

    # Ensure types and validity
    best_support = best_support.astype(np.float64)
    best_weights = best_weights.astype(np.float64)
    
    # Validate
    if len(best_support) != len(best_weights):
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)/n
    
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)/n
    
    if np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)/n
    
    if np.any(best_support < -1e-9):
        best_support = np.clip(best_support, 0, None)
    
    if np.any(best_weights < -1e-9):
        best_weights = np.clip(best_weights, 0, None)
    
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)/n
    
    best_weights = best_weights / w_sum
    
    return best_support, best_weights


# EVOLVE-BLOCK-END

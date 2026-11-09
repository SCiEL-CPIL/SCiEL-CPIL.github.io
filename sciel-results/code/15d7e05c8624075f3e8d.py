"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds a root for a 2D system F(x) = 0 using CMA-ES.
    F: callable taking a numpy array of length 2, returns a numpy array of length 2.
    x0: initial guess, numpy array of length 2.
    Returns: numpy array of length 2 such that ||F(x)|| <= 1e-8.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin for the 120s limit (evaluator overhead etc.)
    
    x0 = np.array(x0, dtype=float)
    
    # Initial state
    x = x0.copy()
    fx = F(x)
    best_x = x.copy()
    best_norm = np.linalg.norm(fx)
    
    if best_norm <= 1e-8:
        return best_x

    # CMA-ES Parameters
    # Population size lambda
    lambda_pop = 8
    # Initial step size sigma
    # Heuristic: if x0 is large, start wide; if small, start moderate.
    # We want to cover a reasonable range. 
    # Let's estimate scale from x0. If x0 is 0, use 1.0.
    scale = max(1.0, np.linalg.norm(x0))
    sigma = scale * 0.5
    
    # Covariance matrix C (initially identity)
    C = np.eye(2)
    
    # Evolution path for covariance adaptation
    p_sigma = np.zeros(2)
    p_c = np.zeros(2)
    
    # Adaptation parameters (standard CMA-ES for n=2)
    # mu_eff = lambda / 4 (for n=2, usually lambda=4+floor(3log(n))=8)
    # Weights: log weights
    w = np.zeros(lambda_pop)
    for i in range(lambda_pop):
        if i < lambda_pop // 2:
            w[i] = np.log(lambda_pop / 2) - np.log(i + 1)
    w /= w.sum()
    mu_eff = 1.0 / np.sum(w**2)
    
    c_sigma = (mu_eff + 2) / (n + mu_eff / 2)  # n=2
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_cov = (2 + 2 * mu_eff / n) / (n**1.8 + mu_eff)
    
    n = 2
    
    # Precompute square roots of weights for sampling
    sqrt_w = np.sqrt(w)
    
    # Helper to generate random normal matrix
    def randn_matrix(rows, cols):
        return np.random.randn(rows, cols)
    
    # Main loop
    while time.time() - start_time < time_limit:
        # Check time before generating population
        if time.time() - start_time > time_limit - 1.0:
            break
            
        # Sample population
        Y = randn_matrix(lambda_pop, n)
        # Apply Cholesky decomposition of C
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # If C is not positive definite, reset
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            sigma = scale * 0.5
            
        X = np.empty((lambda_pop, n))
        F_vals = np.empty(lambda_pop)
        
        for i in range(lambda_pop):
            # x_i = x + sigma * Y_i^T C_sqrt^T (or similar, C is symmetric)
            # Standard: x_i = m + sigma * C_sqrt @ (sqrt(w_i) * y_i)
            # Actually standard CMA-ES: x_i = m + sigma * (C_sqrt @ (sqrt(w_i) * y_i)) is not quite right.
            # It is: x_i = m + sigma * (C_sqrt @ y_i) where y_i is standard normal, 
            # but the distribution is defined by C.
            # Correct: x_i = m + sigma * (C_sqrt @ y_i)
            # Wait, the variance is sigma^2 * C.
            # So x_i = m + sigma * (C_sqrt @ y_i) where y_i ~ N(0, I)
            # Let's stick to the standard formulation:
            # z_i ~ N(0, C)
            # x_i = m + sigma * z_i
            
            # Efficient: z_i = C_sqrt @ y_i
            z = C_sqrt @ Y[i]
            X[i] = x + sigma * z
            
            # Evaluate
            try:
                f_val = F(X[i])
                norm_val = np.linalg.norm(f_val)
                F_vals[i] = norm_val
                if norm_val < best_norm:
                    best_norm = norm_val
                    best_x = X[i].copy()
                    if best_norm <= 1e-8:
                        return best_x
            except:
                F_vals[i] = np.inf

        # Sort by fitness
        order = np.argsort(F_vals)
        X_sorted = X[order]
        F_sorted = F_vals[order]
        
        # Update mean
        x_new = np.zeros(n)
        for i in range(lambda_pop):
            x_new += w[i] * X_sorted[i]
        
        # Update evolution paths
        # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_new - x) / sigma
        # Need C_sqrt_inv. 
        # C = C_sqrt @ C_sqrt.T
        # C_sqrt_inv = inv(C_sqrt)
        try:
            C_sqrt_inv = np.linalg.inv(C_sqrt)
        except:
            C_sqrt_inv = np.eye(n)
            
        # (x_new - x) / sigma
        diff = (x_new - x) / sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ diff)
        
        # h_sigma = 1 if ||p_sigma|| < (1.4 + 1/n) * E[||N(0,I)||] * sqrt(1 - (1-c_sigma)^{2*gen})
        # Approximation for E[||N(0,I)||] in 2D: sqrt(pi/2) ~ 1.25
        # We can just use a threshold
        h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 1.0/n) * 1.25 else 0.0
        
        # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv @ (x_new - x) / sigma
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ diff)
        
        # Update C
        # C = (1 - c_cov) * C + c_cov * (p_c @ p_c.T + (1 - h_sigma^2) * c_c * (2 - c_c) * C)
        # Wait, the standard update is:
        # C = (1 - c_cov) * C + c_cov * (sum w_i (y_i - y_bar)(y_i - y_bar)^T + h_sigma^2 * (1 - c_c) * (2 - c_c) * C)
        # Where y_i are the standardized steps.
        
        # Let's use the standard rank-mu update.
        # y_i = (X_sorted[i] - x) / sigma
        # y_bar = sum w_i y_i = (x_new - x)/sigma = diff
        
        # We need the weighted outer product sum.
        # H = sum w_i (y_i - y_bar) (y_i - y_bar)^T
        
        # To save time, we can approximate or just do it. n=2 is small.
        H = np.zeros((n, n))
        for i in range(lambda_pop):
            y_i = (X_sorted[i] - x) / sigma
            diff_y = y_i - diff
            H += w[i] * np.outer(diff_y, diff_y)
            
        # Add the h_sigma term
        # The term is h_sigma^2 * (1 - c_c) * (2 - c_c) * C
        # Wait, the formula is:
        # C = (1 - c_cov) * C + c_cov * (H + h_sigma^2 * (1 - c_c) * (2 - c_c) * C)
        # Actually, the h_sigma term corrects for the bias when h_sigma=0.
        # Let's use:
        # C_new = (1 - c_cov) * C + c_cov * (H + h_sigma^2 * (1 - c_c) * (2 - c_c) * C)
        
        C = (1 - c_cov) * C + c_cov * (H + h_sigma^2 * (1 - c_c) * (2 - c_c) * C)
        
        # Ensure C is symmetric
        C = (C + C.T) / 2
        
        # Update sigma
        # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E[||N(0,I)||] - 1))
        # E[||N(0,I)||] for n=2 is sqrt(pi/2)
        e_norm = np.sqrt(np.pi / 2)
        sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
        
        # Clamp sigma
        if sigma < 1e-10:
            sigma = 1e-10
        if sigma > 100 * scale:
            sigma = 100 * scale
            
        x = x_new

    return best_x
# EVOLVE-BLOCK-END

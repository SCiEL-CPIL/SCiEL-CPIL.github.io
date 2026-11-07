import numpy as np
import time

class Optimizer:
    """
    Robust CMA-ES implementation for BBOB f14 (Different powers).
    Uses standard CMA-ES with proper covariance adaptation, restarts,
    and a final local polish phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit
        
        n = self.dim
        total_budget = self.budget
        
        def time_left():
            return deadline - time.time()
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Initial evaluation at origin
        x0 = np.zeros(n)
        f0 = func(x0)
        update_best(x0, f0)
        calls_made = 1
        
        # CMA-ES parameters
        lambda_ = 4 + 3 * int(np.floor(np.log2(n)))
        mu = lambda_ // 2
        weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ + 1))
        weights = weights / weights.sum()
        mueff = 1 / np.sum(weights[:mu] ** 2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        def run_cma_es(m0, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f
            
            m = m0.copy()
            sigma = sigma0
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            gen = 0
            stagnation_counter = 0
            best_gen_f = np.inf
            best_gen_x = m.copy()
            
            while calls_made < max_calls and time_left() > 5:
                # Sample population
                Y = np.zeros((lambda_, n))
                F = np.zeros(lambda_)
                
                for i in range(lambda_):
                    z = np.random.standard_normal(n)
                    # Transform z using current C
                    # Use Cholesky or eigendecomp. For n=5, Cholesky is fast if C is SPD.
                    # We'll maintain C and use np.linalg.cholesky, falling back to eig if needed.
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback to eigendecomposition
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-12, None)
                        L = eigvecs @ np.diag(np.sqrt(eigvals))
                    
                    Y[i] = m + sigma * (L @ z)
                    F[i] = func(Y[i])
                    calls_made += 1
                    
                    if time_left() < 5:
                        break
                
                if calls_made >= max_calls or time_left() < 5:
                    break
                
                # Sort
                idx = np.argsort(F)
                Y_sorted = Y[idx]
                F_sorted = F[idx]
                
                # Update best
                update_best(Y_sorted[0], F_sorted[0])
                
                # Check for stagnation
                if F_sorted[0] < best_gen_f - 1e-10:
                    best_gen_f = F_sorted[0]
                    best_gen_x = Y_sorted[0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # Update mean
                m_new = np.dot(weights[:mu], Y_sorted[:mu])
                
                # Update sigma path
                # C^{-1/2} (m_new - m) / sigma
                # We need C^{-1/2}. Let's compute it via eigendecomposition for stability in update.
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                
                delta_m = m_new - m
                z_m = C_inv_sqrt @ delta_m / sigma
                
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * z_m
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)
                
                # Update covariance path
                h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mueff) * z_m
                
                # Update covariance
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma) c_c (2-c_c) C) + c_mu sum w_i y_i y_i^T
                C_new = (1 - c_1 - c_mu) * C
                C_new += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
                
                for i in range(mu):
                    delta_i = Y_sorted[i] - m
                    z_i = C_inv_sqrt @ delta_i / sigma
                    C_new += c_mu * weights[i] * np.outer(z_i, z_i)
                
                C = C_new
                m = m_new
                gen += 1
                
                # Restart condition: if stagnation is high and sigma is small, break
                if stagnation_counter > 30 and sigma < 1e-4:
                    break
                if sigma < 1e-10:
                    break
            
            return m, sigma, best_gen_x, best_gen_f
        
        # Multi-start strategy
        # Start 1: At origin
        m, sigma, _, _ = run_cma_es(np.zeros(n), 2.0, total_budget)
        
        # If we have budget left, try a few other starting points if the first one didn't find a very good solution
        # Heuristic: if best_f is still large, try other points
        if calls_made < total_budget and time_left() > 10:
            # Try a few random starts if we haven't converged
            if best_f > -50:  # Arbitrary threshold, f14 can go quite negative
                for _ in range(3):
                    if calls_made >= total_budget or time_left() < 10:
                        break
                    m0 = np.random.uniform(-4, 4, n)
                    f0 = func(m0)
                    calls_made += 1
                    update_best(m0, f0)
                    
                    if f0 < best_f - 10:  # Significant improvement
                        m, sigma, _, _ = run_cma_es(m0, 1.0, total_budget)
        
        # Final polish phase
        if calls_made < total_budget and time_left() > 5:
            polish_sigma = max(sigma * 0.1, 1e-6)
            polish_lambda = max(2, lambda_ // 2)
            
            while calls_made < total_budget and time_left() > 2:
                Y_polish = np.zeros((polish_lambda, n))
                F_polish = np.zeros(polish_lambda)
                
                for i in range(polish_lambda):
                    z = np.random.standard_normal(n)
                    Y_polish[i] = best_x + polish_sigma * z
                    F_polish[i] = func(Y_polish[i])
                    calls_made += 1
                    
                    if time_left() < 2:
                        break
                
                if calls_made >= total_budget or time_left() < 2:
                    break
                
                idx_best = np.argmin(F_polish)
                if F_polish[idx_best] < best_f:
                    best_f = F_polish[idx_best]
                    best_x = Y_polish[idx_best].copy()
                    polish_sigma *= 0.8
                else:
                    polish_sigma *= 0.5
                
                if polish_sigma < 1e-10:
                    break
        
        return best_x, best_f

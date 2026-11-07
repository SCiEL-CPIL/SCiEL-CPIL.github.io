"""Combined optimizer for BBOB f14 (Different powers) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s kill
        rng = np.random.default_rng(42)
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to safely evaluate function and update best
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        # For dim=5, lambda is typically 4 + floor(3 * log(dim))
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2  # Number of parents
        
        # Weights
        # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu
        weights = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial step size
        sigma = 1.0
        
        # Initial mean
        # Start from origin or a random point? Origin is often safe, but for shifted functions, random might be better.
        # Let's do a very short coarse exploration first to find a better start.
        
        # Phase 1: Coarse Exploration (10% of budget)
        n_coarse = max(5, int(budget * 0.1))
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        for _ in range(n_coarse - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
            
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = f0
            
        mean = best_x.copy()
        
        # Initialize covariance matrix C = I
        C = np.eye(dim)
        # Initialize evolution path p_sigma = 0, p_c = 0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute sqrtm_inv for efficient sampling
        # We need to update C, and thus its inverse square root, every generation.
        # For dim=5, this is cheap.
        
        def update_covariance(C, weights, z, sigma):
            # z is (mu, dim) matrix of selected normalized steps
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T) + c_mu * sigma^2 * sum(w_i * z_i @ z_i.T)
            # Wait, the standard update is:
            # C = (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu sigma^2 sum w_i z_i z_i^T
            # But z_i are the normalized steps (y_i / sigma).
            # Actually, the formula uses the raw steps y_i? No, it uses the normalized steps z_i = y_i / sigma.
            # Let's stick to the standard formulation.
            pass # We'll implement it directly in the loop for clarity and control.

        # Main CMA-ES Loop
        while calls_made < budget and time.time() < deadline - 5:
            # 1. Sample lambda points
            # Need sqrtm_inv of C.
            # Use eigendecomposition for stability and to get the inverse square root.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid negative due to numerical errors
            eigvals = np.maximum(eigvals, 1e-12)
            sqrtm_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Generate lambda candidates
            candidates = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                x_new = mean + sigma * (sqrtm_inv @ z)
                candidates.append(x_new)
            
            # 2. Evaluate candidates
            # We must check budget before each evaluation
            f_values = []
            valid_indices = []
            for i, x_new in enumerate(candidates):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_clipped = np.clip(x_new, -5, 5)
                f_val = func(x_clipped)
                calls_made += 1
                f_values.append(f_val)
                valid_indices.append(i)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
            
            if len(f_values) < mu:
                break
            
            # Sort by function value
            sorted_indices = np.argsort(f_values)
            selected_indices = sorted_indices[:mu]
            selected_x = [candidates[i] for i in selected_indices]
            selected_f = [f_values[i] for i in selected_indices]
            
            # 3. Update mean
            new_mean = np.sum([weights[i] * selected_x[i] for i in range(mu)], axis=0)
            
            # 4. Update evolution paths
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * (1/sigma) * sum w_i z_i
            # We need the z vectors for the selected individuals.
            # z_i = (x_i - mean) / sigma
            z_selected = [(x - mean) / sigma for x in selected_x]
            weighted_z = np.sum([weights[i] * z_selected[i] for i in range(mu)], axis=0)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * weighted_z
            
            # Check h_sigma (hysteresis operator)
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2 gen}) < (1.4 + 2/(dim+1)) * sqrt((mu_eff - 1) + 1) * chi_N
            # This is complex to track 'gen' and chi_N exactly without a counter.
            # Simplification: If ||p_sigma|| is small, h_sigma = 1, else 0.
            # Or just set h_sigma = 1 if ||p_sigma|| < some threshold.
            # Let's use a simpler heuristic: h_sigma = 1 if ||p_sigma|| < 1.0 else 0.0
            # This is not standard CMA-ES but often works well enough.
            p_sigma_norm = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if p_sigma_norm < 1.0 else 0.0
            
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * (1/sigma) * (new_mean - mean)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            
            # 5. Update covariance matrix
            # C = (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu sigma^2 sum w_i z_i z_i^T
            # Note: The term is sum w_i z_i z_i^T, where z_i are the normalized steps.
            # Let's compute the rank-mu update term.
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                z_i = z_selected[i]
                rank_mu_term += weights[i] * np.outer(z_i, z_i)
            
            # Rank-1 term
            rank_1_term = np.outer(p_c, p_c)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_1_term + c_mu * rank_mu_term
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # 6. Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_N - 1))
            # chi_N = sqrt(dim) * (1 - 1/(4dim) + 1/(21dim^2))
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            d_sigma = (mu_eff + 2) / 2 + 2 / (mu_eff + 1) # Approximate
            sigma = sigma * np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_N - 1))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # 7. Update mean
            mean = new_mean
            
            # Check for stagnation or convergence
            if np.linalg.norm(new_mean - mean) < 1e-8: # This check is redundant as we just updated mean
                pass
            
            # If sigma is very small, we might be stuck.
            # We could implement a restart, but let's see if CMA-ES handles it.
            # If sigma < 1e-4, break and try a local polish or restart?
            # For now, let it run.

        # Phase 3: Local Polish (Nelder-Mead) if budget remains
        if calls_made < budget and time.time() < deadline - 3:
            # Use Nelder-Mead on the best point found by CMA-ES
            x = best_x.copy()
            fx = best_f
            initial_step = 0.1
            simplex = [x.copy()]
            simplex_f = [fx]
            
            for i in range(dim):
                v = x.copy()
                v[i] += initial_step
                v = np.clip(v, -5, 5)
                fv = evaluate(v)
                simplex.append(v)
                simplex_f.append(fv)
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            while calls_made < budget and time.time() < deadline - 1:
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                f_spread = simplex_f[-1] - simplex_f[0]
                if f_spread < 1e-8:
                    break
                
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                x_second_worst = simplex[-2]
                f_second_worst = simplex_f[-2]
                x_best = simplex[0]
                f_best = simplex_f[0]
                
                x_c = np.sum(simplex[:-1], axis=0) / dim
                
                x_r = x_c + alpha * (x_c - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                
                if f_r < f_best:
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    if f_r < f_worst:
                        x_co = x_c + 0.5 * (x_r - x_c)
                    else:
                        x_co = x_c + 0.5 * (x_worst - x_c)
                    x_co = np.clip(x_co, -5, 5)
                    f_co = evaluate(x_co)
                    
                    if f_co < f_worst:
                        simplex[-1] = x_co
                        simplex_f[-1] = f_co
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = evaluate(simplex[i])
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END

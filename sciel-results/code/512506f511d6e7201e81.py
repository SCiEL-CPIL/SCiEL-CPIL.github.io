import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 325.0
        rng = np.random.default_rng(0)
        bounds = np.array([-5.0, 5.0])

        # Phase 1: LHS
        n_samples = 50
        samples = np.zeros((n_samples, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_samples + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            samples[:, d] = points

        f_samples = np.array([func(x) for x in samples])
        calls = n_samples

        sorted_idx = np.argsort(f_samples)
        best_x = samples[sorted_idx[0]].copy()
        best_f = f_samples[sorted_idx[0]]

        # Phase 2: CMA-ES with restarts
        # We will run CMA-ES starting from the best LHS point.
        # If it stagnates, we restart from the next best LHS point.
        
        lambda_pop = 10
        max_generations_per_run = 20 # ~200 calls per run
        
        for restart_idx in range(len(sorted_idx)):
            if calls >= budget or time.time() - start_time > max_time:
                break
            
            x0 = samples[sorted_idx[restart_idx]].copy()
            f0 = f_samples[sorted_idx[restart_idx]]
            
            if f0 > best_f:
                break # No point restarting from a worse point than current best

            # Initialize CMA-ES
            m = x0.copy()
            sigma = 0.5
            C = np.eye(dim)
            # Precompute Cholesky
            D_inv = np.sqrt(np.diag(C)) # For identity, this is 1
            B = np.eye(dim)
            
            # Strategy params
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_f = f0
            local_best_x = x0.copy()
            last_improve_gen = 0
            gen = 0
            
            while calls < budget and time.time() - start_time < max_time:
                # Check stagnation
                if gen - last_improve_gen > 10: # 10 generations without improvement
                    break
                
                # Sample
                if calls + lambda_pop > budget:
                    lambda_pop = budget - calls
                    if lambda_pop <= 0: break
                
                z = rng.standard_normal((lambda_pop, dim))
                y = z @ B.T # B is identity initially, but C changes
                # Actually, we need to update B and D from C.
                # For simplicity and speed, I will use a simplified update or just (1+lambda) ES with adaptive sigma?
                # No, CMA-ES is the goal.
                
                # Update B and D from C every few steps? Or just use Cholesky of C.
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
                
                y = z @ L.T
                x_pop = m[None, :] + sigma * y
                for i in range(lambda_pop):
                    x_pop[i] = np.clip(x_pop[i], bounds[0], bounds[1])
                
                f_pop = np.array([func(x) for x in x_pop])
                calls += lambda_pop
                
                # Sort
                order = np.argsort(f_pop)
                y_sorted = y[order]
                f_sorted = f_pop[order]
                x_sorted = x_pop[order]
                
                # Update best
                if f_sorted[0] < local_best_f:
                    local_best_f = f_sorted[0]
                    local_best_x = x_sorted[0].copy()
                    last_improve_gen = gen
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                
                # Update mean
                m_old = m.copy()
                m = m + sigma * np.dot(weights[:mu], y_sorted[:mu].T)
                
                # Update path
                y_mean = (m - m_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (np.linalg.inv(L) @ y_mean)
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(np.trace(C))) else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                # Update Covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_c * (2 - c_c) - c_1 * delta_h) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                C += c_mu * np.sum(weights[:mu, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(np.trace(C)) - 1))
                
                # Ensure C is positive definite
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, 1e12)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                
                gen += 1

        return best_x, best_f

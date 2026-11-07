"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer with large population for robust global search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng, state):
        """Run CMA-ES from initial point x0 with step size sigma0."""
        n = self.dim
        calls_made = state['calls_made']
        
        # CMA-ES parameters
        # Larger population size (30) as per directive to improve exploration and covariance estimation
        pop_size = 30
        mu = pop_size // 2
        
        # Standard CMA-ES hyperparameters
        cc = (4 + n / (4 + n)) / ((n + 4) + (4 + n / (4 + n)))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 1))
        cmu = min(0.3, (2 - cc + 2 / n) / ((n + 2) + cc + 2 / n))
        damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # State variables
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute inverse square root of C (initially identity)
        inv_sqrt_C = np.eye(n)
        
        gen = 0
        while state['calls_made'] < self.budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(pop_size):
                if state['calls_made'] >= self.budget:
                    break
                if time.time() > deadline - 1: # Safety margin
                    break
                
                z = rng.standard_normal(n)
                y = mean + sigma * (inv_sqrt_C @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                state['calls_made'] += 1
                if fy < state['best_f']:
                    state['best_f'] = fy
                    state['best_x'] = y.copy()
                
                offspring.append((fy, y))
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            lam = len(offspring)
            mu_eff = min(mu, lam)
            
            # Update mean
            new_mean = np.zeros(n)
            for i in range(mu_eff):
                new_mean += weights[i] * offspring[i][1]
            new_mean /= np.sum(weights[:mu_eff]) # Normalize if lam < pop_size
            
            # Update covariance
            y_inv_sqrt = np.zeros((mu_eff, n))
            for i in range(mu_eff):
                y_inv_sqrt[i] = (offspring[i][1] - mean) / sigma
            
            # Update path covariance
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (inv_sqrt_C @ (new_mean - mean))
            
            # Update path evolution strategy
            norm_pc = np.linalg.norm(pc)
            # Standard hsig condition
            hsig = 1 if norm_pc / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (n + 1)) * np.sqrt(np.trace(C)) / n else 0
            
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ (new_mean - mean))
            
            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + cmu * (delta_hsig - 1) + 2 * cmu) * C
            for i in range(mu_eff):
                yv = y_inv_sqrt[i]
                C += cmu * weights[i] * (np.outer(yv, yv) - C)
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update inverse square root
            try:
                L = np.linalg.cholesky(C)
                inv_sqrt_C = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
            
            # Update step size (CSA)
            norm_ps = np.linalg.norm(ps)
            c_sigma_norm = np.sqrt(-2 * np.log(np.exp(-1))) # This is just a constant approx 1.597...
            # Standard CSA formula: sigma *= exp(cs/damps * (norm_ps / c_sigma_norm - 1))
            sigma *= np.exp(cs / damps * (norm_ps / 1.597 - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            
            mean = new_mean
            gen += 1
            
            # Convergence check
            if sigma < 1e-8:
                break

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285 # 15s safety margin for 300s limit
        rng = np.random.default_rng(0)

        state = {
            'calls_made': 0,
            'best_x': None,
            'best_f': np.inf
        }

        def tracked_func(x):
            state['calls_made'] += 1
            fx = func(x)
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x.copy()
            return fx

        # Initial point: center of the box
        x0 = np.zeros(self.dim)
        # Initial sigma: large enough to explore the box [-5, 5]
        # Box width is 10, so sigma=2.0 covers ~80% of the range in 1 step
        sigma0 = 2.0
        
        try:
            self._cma_es(func, x0, sigma0, self.budget, deadline, rng, state)
        except Exception:
            pass

        # Final check
        if state['best_x'] is None:
            state['best_x'] = np.zeros(self.dim)
            if state['calls_made'] < self.budget:
                state['best_f'] = func(state['best_x'])
                state['calls_made'] += 1
            else:
                state['best_f'] = 0.0 # Fallback

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END

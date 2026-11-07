import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f_val = func(x)
                if np.isfinite(f_val):
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global random search to find a good starting point
        n_random = 30
        random_points = []
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break
            random_points.append((f_val, x.copy()))
        
        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0
            random_points.append((f0, x0.copy()))

        # Sort random points by fitness
        random_points.sort(key=lambda t: t[0])
        m = random_points[0][1].copy()
        
        # Phase 2: CMA-ES
        lambda_pop = 12
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        sigma = 3.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        try:
            C_cholesky = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_cholesky = np.eye(dim)
            
        gen = 0
        max_gen = 500 // lambda_pop + 10
        no_improvement = 0
        best_f_cma = best_f
        
        while calls < budget and time.time() < deadline and gen < max_gen:
            # Sample population
            z = rng.standard_normal((lambda_pop, dim))
            pop = m + sigma * (z @ C_cholesky.T)
            
            # Clip to box
            pop = np.clip(pop, -5.0, 5.0)
            
            # Evaluate population
            f_values = []
            for x in pop:
                f_val = safe_call(x)
                if f_val is None:
                    break
                f_values.append(f_val)
            
            if len(f_values) < lambda_pop:
                break
                
            f_values = np.array(f_values)
            
            # Sort by fitness
            sort_idx = np.argsort(f_values)
            pop_sorted = pop[sort_idx]
            f_sorted = f_values[sort_idx]
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)
            
            # Check improvement
            if f_sorted[0] < best_f_cma:
                best_f_cma = f_sorted[0]
                no_improvement = 0
            else:
                no_improvement += 1
                
            # Update step size and covariance
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_cholesky.T @ (m_new - m))
            
            # h_sigma check
            h_sigma = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            
            # pc update
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            
            # C update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            
            diffs = (pop_sorted[:mu] - m) / sigma
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('id,jd->ijd', diffs, diffs), axis=0)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is positive semi-definite
            try:
                C_cholesky = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity and restart
                C = np.eye(dim)
                C_cholesky = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                sigma = 3.0
                m = m_new
                continue
                
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            gen += 1
            
            # Convergence check
            if sigma < 1e-6:
                break
            if no_improvement > 20:
                # Try a random restart
                if calls < budget - 10:
                    m = rng.uniform(-5.0, 5.0, size=dim)
                    sigma = 3.0
                    C = np.eye(dim)
                    C_cholesky = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    no_improvement = 0
                else:
                    break

        return best_x, best_f

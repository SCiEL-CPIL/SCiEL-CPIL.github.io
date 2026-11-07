import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n = self.dim
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(0)
        calls = 0
        
        def try_eval(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: CMA-ES with adaptive population
        lambda_ = 20
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state - smaller sigma for better initial convergence
        mean = rng.uniform(-3, 3, n)
        C = np.eye(n)
        sigma = 1.5
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff/n - 1)) + c_sigma
        c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff/n - 1) + 1) / ((n + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        no_improve_count = 0
        gen = 0
        
        while calls < self.budget and time.time() < deadline:
            # Adaptive population: larger early, smaller late
            if gen < 10:
                lam = 25
                mu = lam // 2
            elif gen < 25:
                lam = 20
                mu = lam // 2
            else:
                lam = 15
                mu = lam // 2
            
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff_local = 1.0 / np.sum(w**2)
            
            # Check for stagnation/restart
            if no_improve_count > 15 and gen > 5:
                # Restart with new center near current best
                if best_x is not None:
                    mean = best_x + rng.standard_normal(n) * 0.5
                else:
                    mean = rng.uniform(-3, 3, n)
                C = np.eye(n)
                sigma = 1.0
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                no_improve_count = 0
                continue
            
            # Generate population
            try:
                L = np.linalg.cholesky(C + np.eye(n) * 1e-10)
            except np.linalg.LinAlgError:
                C += np.eye(n) * 1e-5
                L = np.linalg.cholesky(C + np.eye(n) * 1e-10)
            
            pop = []
            f_vals = []
            for i in range(lam):
                if calls >= self.budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x_new = mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                f = func(x_new)
                calls += 1
                pop.append(x_new)
                f_vals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x_new.copy()
            
            if len(f_vals) < lam:
                while len(f_vals) < lam:
                    f_vals.append(np.inf)
                    pop.append(pop[-1] if pop else np.zeros(n))
            
            # Sort
            order = np.argsort(f_vals)
            sorted_pop = [pop[i] for i in order]
            sorted_f = [f_vals[i] for i in order]
            
            # Update no_improve
            if sorted_f[0] < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean
            new_mean = np.sum([w[i] * sorted_pop[i] for i in range(mu)], axis=0)
            
            # Update paths
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff_local) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            chi_n = np.sqrt(n) * (1 + 1/(2*n) + 1/(8*n*n))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls / lam))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff_local) * (new_mean - mean) / sigma
            
            # Update covariance
            v = (new_mean - mean) / sigma
            C = (1 + c_1 * (h_sigma * (1 + (1 - c_sigma) * c_sigma) - c_1) - c_mu) * C \
                + c_1 * (np.outer(v, v) + (1 - h_sigma) * c_sigma * (1 - c_sigma) * C)
            
            y = np.array([(sorted_pop[i] - mean) / sigma for i in range(mu)])
            C += c_mu * np.sum([w[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)
            
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            gen += 1
        
        # Phase 2: Coordinate descent polish with remaining budget
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            f_curr = best_f
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            for step in step_sizes:
                if calls >= self.budget or time.time() > deadline:
                    break
                improved = True
                while improved and calls < self.budget and time.time() < deadline:
                    improved = False
                    for i in range(n):
                        if calls >= self.budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            calls += 1
                            if f_trial < f_curr:
                                f_curr = f_trial
                                x = x_trial
                                improved = True
                if not improved:
                    break
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1
        
        return best_x, best_f

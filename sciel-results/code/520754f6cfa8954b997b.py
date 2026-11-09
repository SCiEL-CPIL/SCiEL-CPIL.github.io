import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Latin Hypercube Sampling for rapid exploration
        # Use ~100 LHS points in [-4, 4]^dim to quickly locate the basin
        n_lhs = min(100, self.budget // 5)
        lhs_points = []
        for i in range(n_lhs):
            # Generate LHS: for each dimension, divide [0,1] into n_lhs strata,
            # pick one uniform point per stratum, then shuffle the permutation
            x = np.zeros(self.dim)
            for d in range(self.dim):
                stratum = (i + rng.uniform()) / n_lhs
                x[d] = -4.0 + 8.0 * stratum
            lhs_points.append(x)
        
        for x in lhs_points:
            if time.time() > deadline or call_count >= self.budget:
                break
            try:
                wrapped_func(x)
            except (TimeoutError, StopIteration):
                break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)
        
        # Phase 2: CMA-ES optimization from the best point found
        remaining_budget = self.budget - call_count
        if remaining_budget > 20:
            new_best_x, new_best_f = self._run_cmaes(wrapped_func, best_x, best_f, remaining_budget, deadline, rng)
            if new_best_f < best_f:
                best_f = new_best_f
                best_x = new_best_x
        
        return best_x, best_f

    def _run_cmaes(self, wrapped_func, x0, f0, budget, deadline, rng):
        """Run CMA-ES with restarts and stagnation detection."""
        dim = self.dim
        best_x = x0.copy()
        best_f = f0
        call_count = 0
        
        def cmaes_step(mean, sigma, C, pc, ps, lambda_, mu, weights, cs, c1, cmu):
            """One step of CMA-ES. Returns updated parameters."""
            nonlocal call_count
            
            # Generate lambda_ samples
            samples = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Whitening: x = mean + sigma * C^{1/2} * z
                # Use eigendecomposition for C^{1/2}
                y = mean + sigma * (C @ z)
                y = np.clip(y, -5.0, 5.0)
                samples.append(y)
            
            # Evaluate
            fvals = []
            for y in samples:
                if call_count >= budget or time.time() > deadline:
                    break
                try:
                    fy = wrapped_func(y)
                    call_count += 1
                    fvals.append((fy, y))
                except (TimeoutError, StopIteration):
                    break
            
            if len(fvals) < 2:
                return mean, sigma, C, pc, ps, None
            
            # Sort by fitness
            fvals.sort(key=lambda t: t[0])
            
            # Update global best
            if fvals[0][0] < best_f:
                best_f = fvals[0][0]
                best_x = fvals[0][1].copy()
            
            # Compute weighted mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * fvals[i][1]
            
            # Update covariance matrix
            C_old = C.copy()
            diff = new_mean - mean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(diff, diff) + (1 - mu_eff) * C_old) + cmu * sum(
                weights[i] * np.outer(fvals[i][1] - mean, fvals[i][1] - mean) for i in range(mu)
            )
            
            # Update evolution paths
            ps = (1 - cs) * ps + cs / np.sqrt(mu_eff) * (C_old @ (diff / sigma))
            pc = (1 - c1) * pc + c1 * (np.sign(ps @ C_old @ np.ones(dim)) - np.ones(dim))
            
            # Adapt sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(2 * dim) * (1 + 0.5 * min(1, (np.linalg.norm(ps)**2 / dim) - 1)) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            return new_mean, sigma, C, pc, ps, call_count
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = (2 - mu_eff / (dim**2 + 1)) / ((dim + 2)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        c_sigma = cs * (2 + 2 / (mu_eff + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Initial state
        mean = best_x.copy()
        sigma = 1.0  # initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Run CMA-ES until budget exhausted or stagnation
        max_iterations = budget
        for iteration in range(max_iterations):
            if call_count >= budget or time.time() > deadline:
                break
            
            mean, sigma, C, pc, ps, _ = cmaes_step(mean, sigma, C, pc, ps, lambda_, mu, weights, cs, c1, cmu)
            
            # Stagnation check: if sigma is very small, restart with perturbation
            if sigma < 1e-6:
                # Restart from best with new sigma
                mean = best_x.copy() + 0.5 * rng.standard_normal(dim)
                mean = np.clip(mean, -5.0, 5.0)
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
            
            # Also restart if we've made little progress
            if iteration > 50 and np.linalg.norm(mean - best_x) < 1e-4:
                mean = best_x.copy() + 0.5 * rng.standard_normal(dim)
                mean = np.clip(mean, -5.0, 5.0)
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
        
        return best_x, best_f

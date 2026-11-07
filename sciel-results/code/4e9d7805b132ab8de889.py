"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Multi-start CMA-ES with diagonal covariance adaptation for ill-conditioned problems.
    Uses 7 diverse starting points to cover the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_diagonal(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        CMA-ES with diagonal covariance matrix adaptation.
        This directly models the ill-conditioning of Schaffers F7 by adapting
        step sizes per dimension.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1
        
        # Diagonal covariance: per-dimension step sizes
        sigma_vec = np.ones(dim) * sigma
        
        # Population size
        lambda_ = max(2 * dim, 4)
        mu = lambda_ // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Variance-effective population size
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4.3 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Evolution path for sigma adaptation
        ps = np.zeros(dim)
        
        # Evolution path for covariance adaptation
        pc = np.zeros(dim)
        
        # Diagonal covariance (per-dimension variances)
        diag_cov = np.ones(dim)
        
        best_x = x.copy()
        best_f = fx
        
        # Initial Hessian estimation for better diagonal covariance initialization
        # Use finite differences to estimate local curvature
        h = 1e-3
        grad = np.zeros(dim)
        hess_diag = np.zeros(dim)
        
        for i in range(dim):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            fp = func(x_plus)
            fm = func(x_minus)
            calls += 2
            grad[i] = (fp - fm) / (2 * h)
            hess_diag[i] = (fp - 2 * fx + fm) / (h ** 2)
        
        # Initialize diagonal covariance based on Hessian estimates
        # If Hessian is large, use smaller step; if small, use larger step
        for i in range(dim):
            if hess_diag[i] > 1e-6:
                diag_cov[i] = 1.0 / max(hess_diag[i], 1e-6)
                diag_cov[i] = np.clip(diag_cov[i], 0.01, 100.0)
            else:
                diag_cov[i] = 1.0
        
        # Scale initial sigma to match the scale of the problem
        # Estimate typical function value scale
        scale = np.sqrt(np.mean(diag_cov))
        sigma = sigma0 * scale
        
        # Generate initial population
        def generate_population(x, sigma, sigma_vec, rng):
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                pop[i] = x + sigma * sigma_vec * z
                pop[i] = np.clip(pop[i], -5, 5)
            return pop
        
        pop = generate_population(x, sigma, sigma_vec, rng)
        fvals = np.array([func(x_i) for x_i in pop])
        calls += lambda_
        
        # Sort by fitness
        order = np.argsort(fvals)
        pop_sorted = pop[order]
        fvals_sorted = fvals[order]
        
        # Update best
        if fvals_sorted[0] < best_f:
            best_f = fvals_sorted[0]
            best_x = pop_sorted[0].copy()
        
        while calls + lambda_ <= max_calls:
            if time.time() > deadline - 2.0:
                break
            
            # Compute mean of top mu
            w = weights[:mu]
            mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # Update evolution paths
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                (mean - x) / (sigma * sigma_vec)
            )
            
            # hsig check
            hsig = np.sqrt((1 + 2 / (cs + 1) + 1e-30))  # Simplified
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - 1))) < (1.4 + 2 / (dim + 1)) * hsig:
                pc = (1 - cc) * pc + cc * (mean - x) / (sigma * sigma_vec)
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / hsig - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update diagonal covariance
            # For diagonal CMA-ES, we adapt each dimension independently
            for i in range(dim):
                # Compute rank-1 and rank-mu updates for dimension i
                diff = (mean - x)[i] / (sigma * sigma_vec[i])
                
                # Rank-1 update
                rank1 = (1 + 2 * (hsig > np.sqrt((1 + 2 / (cs + 1) + 1e-30))) - 1) * cc * (2 - cc) * pc[i] ** 2
            
            # Simplified diagonal update: adapt sigma_vec based on progress
            # If the mean moved in a direction, increase step size in that direction
            for i in range(dim):
                progress = abs((mean - x)[i]) / (sigma * sigma_vec[i] + 1e-30)
                if progress > 1.0:
                    sigma_vec[i] *= 1.05
                else:
                    sigma_vec[i] *= 0.98
                sigma_vec[i] = np.clip(sigma_vec[i], 0.01, 10.0)
            
            # Update mean
            x = mean
            
            # Evaluate new population
            pop = generate_population(x, sigma, sigma_vec, rng)
            fvals = np.array([func(x_i) for x_i in pop])
            calls += lambda_
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            
            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # Check for convergence
            if np.all(np.abs(pop_sorted[:mu] - x) < 1e-8):
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(12345)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Multi-start strategy with 7 diverse starting points
        # Center, corners of the box, and random points
        starts = [np.zeros(dim)]
        
        # Add points near the boundaries (minimizer is in [-4, 4]^5)
        starts.append(np.array([4.0] * dim))
        starts.append(np.array([-4.0] * dim))
        
        # Random points
        for _ in range(4):
            x0 = rng.uniform(-4, 4, size=dim)
            starts.append(x0)
        
        # Add a few more diverse points
        for _ in range(2):
            x0 = rng.uniform(-3, 3, size=dim)
            starts.append(x0)
        
        num_restarts = len(starts)
        
        # Allocate budget: give more to the first few restarts
        # First restart gets 30%, second gets 25%, remaining split
        budget_alloc = [
            int(budget * 0.30),
            int(budget * 0.25),
        ]
        remaining_budget = budget - sum(budget_alloc)
        per_restart = remaining_budget // (num_restarts - 2)
        budget_alloc.extend([per_restart] * (num_restarts - 2))
        
        # Adjust last to use all budget
        budget_alloc[-1] += budget - sum(budget_alloc)
        
        for i, x0 in enumerate(starts):
            if total_calls >= budget:
                break
            
            remaining = budget - total_calls
            max_calls = min(budget_alloc[i], remaining)
            
            if max_calls < 10:
                break
            
            # Initial sigma
            sigma0 = 2.0
            
            bx, bf, used = self._cma_es_diagonal(func, rng, x0, sigma0, max_calls, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            
            # If we are close to the deadline, stop
            if time.time() > deadline - 5.0:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END

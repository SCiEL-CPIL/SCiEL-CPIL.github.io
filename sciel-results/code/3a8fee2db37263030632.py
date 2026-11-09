"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: LHS spread + CMA-ES + L-BFGS-B polish.
    
    Designed for BBOB-style Sphere function with hidden shift/rotation/offset.
    AOCC rewards early progress, so we start with a cheap global spread.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples * 10.0 - 5.0  # Map to [-5, 5]

    def _cmaes(self, func, x0, sigma, budget, deadline, dim, call_count):
        """Simple CMA-ES implementation."""
        rng = np.random.default_rng(42)
        x = x0.copy()
        fx = func(x)
        n = dim
        popsize = max(4 + int(3 * np.log(n)), 6)
        
        # CMA-ES parameters
        cmu = 1. / (4 * n + 4)
        ccs = (4 + n / 4) / (n + 4 + 3)
        c1 = 2 / ((n + 1.3) ** 2 + 4)
        cmu = min(cmu, 2 * (1 - c1) / (n ** 2 + 1))
        csigma = (1 + cmu / 2) / (n + 3)
        
        # Initialize state
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        step_size = sigma
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))
        
        for iteration in range(budget // popsize):
            if time.time() > deadline or call_count[0] >= budget:
                break
                
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(popsize):
                if call_count[0] >= budget:
                    break
                z = rng.standard_normal(n)
                y = x + step_size * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)
            
            if len(offspring) == 0:
                break
            
            offspring_f = np.array(offspring_f)
            offspring = np.array(offspring)
            
            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]
            
            # Update best
            if offspring_f[0] < fx:
                x = offspring[0].copy()
                fx = offspring_f[0]
            
            # Update covariance matrix and step size
            w = np.log(popsize / 2) - np.arange(popsize / 2)
            w = np.concatenate([w, w[::-1]])[:popsize]
            w = w / w.sum()
            mu_eff = 1 / np.sum(w[:int(popsize / 2)] ** 2)
            
            y_mean = np.zeros(n)
            for i in range(int(popsize / 2)):
                y_mean += w[i] * (offspring[i] - x) / step_size
            
            p_sigma = (1 - cmu) * p_sigma + cmu * y_mean
            h = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cmu) ** (2 * (iteration + 1))) < (1.4 + 2 / (n + 1)) * chi_n)
            p_c = (1 - ccs) * p_c + h * ccs * y_mean
            
            # Compute rank-mu update using precomputed C_sqrt inverse
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            rank_mu = np.zeros((n, n))
            for i in range(int(popsize / 2)):
                y_i = C_sqrt_inv @ ((offspring[i] - x) / step_size)
                rank_mu += w[i] * np.outer(y_i, y_i)
            
            C = (1 + c1 * (2 - c1) + cmu) * C + \
                c1 * (np.outer(p_c, p_c) - C) + \
                cmu / mu_eff * rank_mu
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Adapt step size
            step_size *= np.exp(csigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            step_size = np.clip(step_size, 1e-10, 10.0)
            
            # Update mean
            x = x + step_size * C_sqrt @ y_mean
        
        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = [0]
        
        def wrapped(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))
        
        # Phase 1: Stratified sampling with center bias for fast AOCC progress
        # Start with center, then expand outward in a structured way
        rng = np.random.default_rng(42)
        
        # First: evaluate center (likely near optimum for many BBOB instances)
        center = np.zeros(dim)
        best_f = wrapped(center)
        best_x = center.copy()
        
        # Then: structured radial sampling at multiple scales
        # This gives rapid improvement in the first few calls
        for scale in [1.0, 2.0, 3.0, 4.0, 5.0]:
            if call_count[0] >= budget or time.time() > deadline:
                break
            # Sample points on a sphere of given radius
            n_at_scale = 5
            for i in range(n_at_scale):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                # Random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                x = center + scale * direction
                x = np.clip(x, -5, 5)
                fx = wrapped(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        # Then: LHS for broader coverage
        n_lhs = 20
        lhs_samples = self._lhs_sample(n_lhs, dim, rng)
        for x in lhs_samples:
            if time.time() > deadline or call_count[0] >= budget:
                break
            fx = wrapped(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES from best sample (rotation-adaptive)
        if time.time() < deadline - 30 and call_count[0] < budget - 50:
            try:
                x_cma, f_cma = self._cmaes(wrapped, best_x, 2.0, budget - call_count[0], deadline - 30, dim, call_count)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma
            except Exception:
                pass
            
            # Phase 2b: Second CMA-ES restart with smaller sigma if still time
            if time.time() < deadline - 15 and call_count[0] < budget - 20:
                try:
                    x_cma2, f_cma2 = self._cmaes(wrapped, best_x, 0.5, budget - call_count[0], deadline - 15, dim, call_count)
                    if f_cma2 < best_f:
                        best_f = f_cma2
                        best_x = x_cma2
                except Exception:
                    pass
        
        # Phase 3: L-BFGS-B final polish (convex Sphere function)
        if time.time() < deadline - 5 and call_count[0] < budget - 5:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped, best_x, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        # Phase 4: Centered quadratic model fit for super-linear convergence on Sphere
        if time.time() < deadline - 2 and call_count[0] < budget - 10:
            try:
                # Fit a quadratic model around best_x using 2*dim+1 points
                # f(x) ≈ f0 + g^T (x-x0) + 0.5 * (x-x0)^T H (x-x0)
                # For Sphere: H = 2*I, so we can estimate g and confirm H structure
                # Use central differences for gradient, then do exact Newton step
                for _ in range(5):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    grad = np.zeros(dim)
                    for d in range(dim):
                        h = 1e-5
                        x_plus = best_x.copy()
                        x_minus = best_x.copy()
                        x_plus[d] += h
                        x_minus[d] -= h
                        fp = wrapped(np.clip(x_plus, -5, 5))
                        fm = wrapped(np.clip(x_minus, -5, 5))
                        grad[d] = (fp - fm) / (2 * h)
                    
                    # For Sphere, Hessian is ~2*I, so Newton step: x_new = x - grad/2
                    # But to be safe, use a damped step
                    step = grad / 2.0
                    new_x = best_x - step
                    new_x = np.clip(new_x, -5, 5)
                    new_f = wrapped(new_x)
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_x.copy()
                    else:
                        # Try half step
                        new_x = best_x - step * 0.5
                        new_x = np.clip(new_x, -5, 5)
                        new_f = wrapped(new_x)
                        if new_f < best_f:
                            best_f = new_f
                            best_x = new_x.copy()
                        else:
                            break
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END

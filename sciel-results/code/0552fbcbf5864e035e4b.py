"""CMA-ES with Sobol seeding for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n_samples, dim):
        """Generate Sobol sequence in [0,1]^dim using scipy."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        m = int(np.ceil(np.log2(max(n_samples, 2))))
        n_pow2 = 2 ** m
        points = sampler.random(n_pow2)
        return points[:n_samples]

    def _cma_es(self, func, x0, sigma, budget, calls_ref, deadline):
        """
        Lightweight CMA-ES implementation.
        Returns best_x, best_f, calls_used.
        """
        n = self.dim
        lambda_pop = 4 + int(3 * np.log(n))
        if lambda_pop < 4: lambda_pop = 4
        
        # Parameters
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (n + 2) - 1))
        
        # State
        mean = x0.copy()
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        sigma = sigma
        best_x = mean.copy()
        best_f = float('inf')
        
        calls_used = 0
        max_iter = budget // lambda_pop
        
        for gen in range(max_iter):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
                
            # Sample points
            B = np.linalg.cholesky(C)
            D = 1.0 / np.sqrt(np.diag(B)) # Actually C = B B^T, so step is B * z
            # Correction: If C = B @ B.T, then sqrt(C) is B.
            # Standard CMA-ES uses C^1/2.
            # Let's use eigendecomposition for stability or just B from Cholesky.
            # B from Cholesky is lower triangular such that C = B @ B.T.
            # So x = mean + sigma * B @ z.
            
            ys = []
            xs = []
            for i in range(lambda_pop):
                z = np.random.normal(size=n)
                x = mean + sigma * (B @ z)
                # Clip to bounds? The problem says func clips, but CMA-ES works better if we stay in bounds.
                # However, the contract says func clips. We can rely on that, or clip ourselves to avoid wasted calls?
                # Clipping ourselves is better for CMA-ES convergence near boundaries.
                x = np.clip(x, -5.0, 5.0)
                
                if calls_used >= budget:
                    break
                f = func(x)
                calls_used += 1
                ys.append(f)
                xs.append(x)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(ys) < lambda_pop:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            
            # Sort by fitness
            order = np.argsort(ys)
            xs_sorted = xs[order]
            ys_sorted = ys[order]
            
            # Update mean
            mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Update step size
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} (mean_new - mean_old) / sigma
            # We need C^{-1/2}.
            # Using B from previous iteration is an approximation.
            # Better: use eigendecomposition of C.
            # For simplicity and speed in this small dim, let's use the B approximation or recompute.
            # Recomputing eig is O(n^3) = 125 ops, negligible.
            w, V = np.linalg.eigh(C)
            C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
            
            y_mean = mean - (mean - (mean - sigma * (B @ np.random.normal(size=n)))) # This is wrong, mean_old is needed
            # We didn't store mean_old. Let's store it.
            # Actually, let's simplify.
            # p_sigma = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean_new - mean_old) / (sigma * ||C^{-1/2}||) ?
            # Standard formula:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} (x_mean_new - x_mean_old) / sigma
            
            # We need mean_old.
            # Let's restructure the loop to store mean_old.
            
            # Update covariance
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - mean_old) / sigma
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc pc^T + (1 - delta_sigma) * c_sigma * (2 - c_sigma) * ps ps^T) + c_mu * sum(wi (xi - mean_old)(xi - mean_old)^T / sigma^2)
            
            # This manual implementation is error-prone.
            # Let's use a simpler robust derivative-free method if CMA-ES is too complex to get right in one shot.
            # `scipy.optimize.dual_annealing` is a strong candidate.
            # Or `scipy.optimize.differential_evolution`.
            
            # Let's abort CMA-ES and use `differential_evolution` which is built-in and robust.
            # But DE is population based and might waste budget on random walks.
            # `dual_annealing` is often better for smooth functions.
            
            # Let's try `dual_annealing` in the main call instead.
            break
            
        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n = dim

        # Phase 1: Sobol-based global sampling for early coverage
        n_sobol = min(50, max(10, budget // 10))
        sobol_pts = self._sobol_sequence(n_sobol, n)
        sobol_pts = sobol_pts * 10.0 - 5.0

        best_x = sobol_pts[0].copy()
        best_f = float('inf')
        calls = 0

        for i in range(n_sobol):
            if time.time() - start_time > deadline:
                break
            x = sobol_pts[i]
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if calls >= budget or time.time() - start_time > deadline:
            return best_x, best_f

        # Phase 2: Use Dual Annealing for global optimization
        from scipy.optimize import dual_annealing

        remaining = budget - calls

        if remaining < 10:
            return best_x, best_f

        def _wrap(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        try:
            res = dual_annealing(
                _wrap,
                x0=best_x,
                bounds=[(-5.0, 5.0)] * n,
                maxiter=remaining // 5, # Limit iterations
                restarts=5,
                seed=42
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        if calls >= budget or time.time() - start_time > deadline:
            return best_x, best_f

        # Phase 3: Polish with L-BFGS-B from the best point found by Dual Annealing
        remaining = budget - calls
        if remaining >= 5:
            from scipy.optimize import minimize
            try:
                res2 = minimize(
                    _wrap,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={'maxiter': remaining, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

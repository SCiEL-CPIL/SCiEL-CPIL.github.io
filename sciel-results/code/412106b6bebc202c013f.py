"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False


class Optimizer:
    """Optimizer combining CMA-ES, Nelder-Mead, and (1+1)-ES for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _custom_cmaes(self, func, x0, sigma0, budget, rng, evals_ref, deadline):
        """Fallback CMA-ES implementation if cma library is not available."""
        dim = self.dim
        N = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = N // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize
        x_mean = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = 1.0 / np.sqrt(dim)
        norm_s = np.sqrt(dim)

        best_x = x_mean.copy()
        best_f = func(np.clip(best_x, -5, 5))
        evals = 1

        while evals < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []
            
            # Compute Cholesky factorization once per generation
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                except:
                    break

            for i in range(N):
                if evals >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) < mu:
                break

            # Sort by fitness
            sorted_idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in sorted_idx[:mu]]
            sorted_offspring_f = [offspring_f[i] for i in sorted_idx[:mu]]

            # Update best
            if sorted_offspring_f[0] < best_f:
                best_f = sorted_offspring_f[0]
                best_x = sorted_offspring[0].copy()

            # Weighted mean of selected
            x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)

            # Update evolution paths
            diff = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals + 1) // N))) / norm_s
            hsig = min(hsig, 1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hsig * c_c * diff
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * sum(
                weights[i] * np.outer(sorted_offspring[i] - x_mean, sorted_offspring[i] - x_mean) / sigma ** 2
                for i in range(mu)
            )

            # Symmetrize C
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(c_sigma / norm_s * (np.linalg.norm(ps) / norm_s - 1))

            x_mean = x_new

            # Check for stagnation
            if sigma < 1e-10:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration to find a good starting point
        # Use a small number of random points to estimate the scale and find a basin
        n_init = min(20, budget // 5)
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        init_vals = [func(np.clip(p, -5, 5)) for p in init_points]
        evals = n_init

        best_idx = int(np.argmin(init_vals))
        best_x = init_points[best_idx].copy()
        best_f = init_vals[best_idx]

        # Estimate initial sigma from spread of good points
        sorted_idx = np.argsort(init_vals)
        top_k = min(5, len(init_vals))
        top_points = np.array([init_points[i] for i in sorted_idx[:top_k]])
        if len(top_points) > 1:
            distances = np.linalg.norm(top_points[1:] - top_points[0], axis=1)
            sigma0 = max(0.1, np.mean(distances) * 0.5)
        else:
            sigma0 = 1.0

        # Phase 2: Global Optimization (CMA-ES)
        remaining_budget = budget - evals
        if remaining_budget > 0:
            if HAS_CMA:
                try:
                    x0 = best_x.tolist()
                    # Use 70% of remaining budget for CMA-ES
                    cma_budget = int(remaining_budget * 0.7)
                    maxiter = max(1, cma_budget // (4 + int(3 * math.log(dim))))
                    
                    options = {
                        'verbose': -9,
                        'seed': 1,
                        'bounds': [-5.0] * dim + [5.0] * dim,
                        'CMA_boundaryhandler': 'reflect',
                        'popsize': 4 + int(3 * math.log(dim)),
                        'maxiter': maxiter,
                        'stopitersize': 1e-10,
                        'stopconditionnumber': 1e12,
                        'stopfgap': 1e-12,
                    }
                    
                    es = cma.CMA(x0, sigma0, options)
                    
                    while not es.stop() and evals < self.budget:
                        if time.time() > deadline - 10:
                            break
                        
                        xs = es.ask()
                        fs = []
                        for x in xs:
                            if evals >= self.budget:
                                break
                            if time.time() > deadline - 10:
                                break
                            fx = func(x)
                            evals += 1
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            fs.append(fx)
                        
                        if len(fs) == len(xs):
                            es.tell(xs, fs)
                        else:
                            break
                            
                except Exception:
                    # Fallback to custom CMA-ES if cma fails
                    new_best_x, new_best_f, evals_used = self._custom_cmaes(
                        func, best_x, sigma0, remaining_budget, rng, evals, deadline
                    )
                    evals += evals_used
                    if new_best_f < best_f:
                        best_f = new_best_f
                        best_x = new_best_x.copy()
            else:
                # Fallback to custom CMA-ES
                new_best_x, new_best_f, evals_used = self._custom_cmaes(
                    func, best_x, sigma0, remaining_budget, rng, evals, deadline
                )
                evals += evals_used
                if new_best_f < best_f:
                    best_f = new_best_f
                    best_x = new_best_x.copy()

        # Phase 3: Local Refinement (Nelder-Mead)
        if HAS_SCIPY_OPT and evals < self.budget and time.time() < deadline - 10:
            try:
                def wrapper(x):
                    nonlocal evals, best_f, best_x
                    if evals >= self.budget or time.time() > deadline - 5:
                        return best_f # Return best known to stop early? No, better to return large value or just let it run out.
                        # Actually, if we exceed budget, we should not call func.
                        # But minimize doesn't check budget. We must limit maxiter.
                    
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    evals += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                # Estimate max iterations for Nelder-Mead
                nm_budget = self.budget - evals
                # Nelder-Mead uses dim+1 function evaluations per iteration roughly
                maxiter_nm = max(10, nm_budget // (dim + 1))
                
                res = minimize(wrapper, best_x, method='Nelder-Mead', 
                               options={'maxiter': maxiter_nm, 'xatol': 1e-6, 'fatol': 1e-6})
                
                # minimize returns the best point found
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Phase 4: Local Polish: (1+1)-ES with adaptive step size
        remaining = self.budget - evals
        if remaining > 0 and time.time() < deadline - 5:
            rng_local = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma
            sigma = 0.05
            no_improve = 0
            limit = 15
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = 0.1
                    no_improve = 0
                    
                y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                evals += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if sigma < 1e-5:
                        break
                        
        return best_x, best_f
# EVOLVE-BLOCK-END

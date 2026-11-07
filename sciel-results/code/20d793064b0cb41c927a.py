"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start optimizer with CMA-ES-like refinement and L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        call_count = [0]

        def counted_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return val

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.clip(x, -5, 5).copy()

        # Phase 1: Diverse initial sampling using Sobol + random points
        n_sobol = 32
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_sobol)
        sobol_pts = sobol_pts * 10.0 - 5.0

        # Add center and some random points
        extra_pts = [np.zeros(dim)]
        for _ in range(8):
            extra_pts.append(rng.uniform(-4, 4, dim))

        all_starts = np.vstack([sobol_pts, np.array(extra_pts)])

        start_values = []
        for pt in all_starts:
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            try:
                fv = counted_func(pt)
            except Exception:
                fv = 1e10
            start_values.append((fv, pt.copy()))
            update_best(pt, fv)

        if not start_values:
            pt = np.zeros(dim)
            try:
                fv = counted_func(pt)
            except Exception:
                fv = 1e10
            start_values.append((fv, pt.copy()))
            update_best(pt, fv)

        # Sort by function value
        start_values.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES-like local search from top starts
        def cmaes_local_search(x0, max_evals=30):
            """Simple CMA-ES-like local search."""
            if time.time() > deadline:
                return x0, counted_func(x0) if call_count[0] < budget else 1e10
            
            x = x0.copy()
            sigma = 0.5
            mean = x.copy()
            best_local_x = x.copy()
            best_local_f = counted_func(x) if call_count[0] < budget else 1e10
            update_best(x, best_local_f)
            
            n = dim
            lambda_pop = 10
            mu = lambda_pop // 2
            c_alpha = (mu_eff + 2) / (n + mu_eff + 5)
            c_sigma = (mu_eff + 1) / (n + 2)
            mu_eff = 1.0 / sum(1.0 / i for i in range(1, mu + 1))
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            for iteration in range(max_evals // lambda_pop + 1):
                if call_count[0] >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for _ in range(lambda_pop):
                    if call_count[0] >= budget - 5:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Sample from multivariate normal
                    z = rng.standard_normal(n)
                    # Simple diagonal approximation for C
                    B = np.eye(n)
                    D = np.sqrt(np.diag(C))
                    x_new = mean + sigma * (B @ (D * z))
                    x_new = np.clip(x_new, -5, 5)
                    
                    try:
                        f_new = counted_func(x_new)
                    except Exception:
                        f_new = 1e10
                    
                    offspring.append((f_new, x_new))
                    update_best(x_new, f_new)
                
                if len(offspring) < mu:
                    break
                
                # Sort offspring
                offspring.sort(key=lambda t: t[0])
                
                # Update mean
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                
                new_mean = sum(weights[i] * offspring[i][1] for i in range(mu))
                
                # Update covariance matrix (simplified)
                diff = new_mean - mean
                pc = (1 - c_alpha) * pc + c_alpha * diff / (sigma * chi_n)
                
                # Update sigma (simplified)
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(c_alpha * (2 - c_alpha) / (n + 1)) * diff / sigma
                
                sigma *= np.exp((1/chi_n - 1) * (np.linalg.norm(ps) - chi_n) / (5 * (n + 3)))
                sigma = max(sigma, 0.01)
                sigma = min(sigma, 5.0)
                
                # Update C (simplified rank-1 update)
                C = (1 + c_alpha * (2 - c_alpha)) * C + c_alpha * np.outer(pc, pc)
                # Normalize to maintain determinant
                det_C = np.linalg.det(C)
                if det_C > 0:
                    C /= det_C ** (1.0 / n)
                
                mean = new_mean
                mean = np.clip(mean, -5, 5)
                
                # Track best local
                if offspring[0][0] < best_local_f:
                    best_local_f = offspring[0][0]
                    best_local_x = offspring[0][1].copy()
            
            return best_local_x, best_local_f

        # Apply CMA-ES to top starts
        k = min(8, len(start_values))
        for i in range(k):
            if call_count[0] >= budget - 10:
                break
            if time.time() > deadline:
                break
            
            f0, x0 = start_values[i]
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            remaining = budget - call_count[0]
            if remaining < 15:
                break
            
            try:
                x_best, f_best = cmaes_local_search(x0, max_evals=min(30, remaining - 5))
                update_best(x_best, f_best)
            except Exception:
                pass

        # Phase 3: L-BFGS-B refinement from best point
        if best_x is not None and call_count[0] < budget - 10:
            remaining = budget - call_count[0]
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, remaining // 3), 'maxls': 20}
                )
                if res.success and np.isfinite(res.fun):
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 4: If budget remains, do more random perturbations around best
        remaining = budget - call_count[0]
        if remaining > 10 and best_x is not None:
            for _ in range(min(remaining // 2, 20)):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Adaptive perturbation size
                scale = max(0.1, 0.5 * (1 - call_count[0] / budget))
                perturbation = rng.standard_normal(dim) * scale
                x_try = np.clip(best_x + perturbation, -5, 5)
                
                try:
                    fv = counted_func(x_try)
                except Exception:
                    fv = 1e10
                
                update_best(x_try, fv)

        # Final: if we haven't evaluated anything, do a basic search
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                try:
                    best_f = counted_func(best_x)
                except Exception:
                    best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END

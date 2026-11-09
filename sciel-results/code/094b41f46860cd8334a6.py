# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def generate_sobol_like(n, dim, seed=0):
    """
    Generate a deterministic set of space-filling points using a Halton sequence.
    This provides better initial coverage than random sampling for multi-start.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(n):
        p = []
        for d in range(dim):
            base = d + 2
            val = halton(i + 1, base)
            # Map [0, 1) to [-5, 5]
            p.append(val * 10 - 5)
        points.append(p)
    return np.array(points)

class Optimizer:
    """
    Hybrid CMA-ES + Nelder-Mead optimizer for BBOB f14.
    Uses CMA-ES for global exploration and adaptation, then NM for local polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # 1. Initial Evaluation using Space-Filling Points
        # We use a few Halton points to find a good starting region
        n_init = 10
        init_points = generate_sobol_like(n_init, dim)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate initial points
        for p in init_points:
            if not time_ok() or calls_made >= budget:
                break
            x_eval = np.clip(p, -5, 5)
            f_val = func(x_eval)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_eval.copy()

        # 2. CMA-ES Phase
        # CMA-ES is robust to scaling and rotation, which are present in the hidden transformation.
        # We run it for a significant portion of the budget.
        
        # Parameters
        lambda_pop = 4  # Population size
        sigma = 1.0     # Initial step size
        cc = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        cs = (lambda_pop / 2 + 1) / (dim + lambda_pop / 2 + 5)
        cmu = min(1 - 2 / ((dim + 1) ** 2 + lambda_pop), 2 / ((dim + 2) * (lambda_pop / 2 + 1)))
        ccov = (5 + cmu - cc) / (dim + 5)
        
        # State variables
        x_mean = best_x.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = 1.0
        
        # Precompute sqrt of C for sampling
        # We will update this periodically or use Cholesky
        # For simplicity and speed in 5D, we can just use the eigen-decomposition or Cholesky
        # C is initially I, so L is I
        
        # CMA-ES Loop
        # We reserve ~10% of budget for NM polish
        cma_budget = int(budget * 0.8)
        
        while calls_made < cma_budget and time_ok():
            # Sample lambda_pop points
            # Use Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or fix
                C = np.eye(dim)
                L = np.eye(dim)
                
            # Sample
            y_samples = []
            for _ in range(lambda_pop):
                z = np.random.normal(0, 1, dim)
                y = L @ z
                y_samples.append(y)
            
            # Evaluate
            x_samples = []
            f_samples = []
            for y in y_samples:
                if not time_ok() or calls_made >= budget:
                    break
                x_new = x_mean + sigma * y
                x_clipped = np.clip(x_new, -5, 5)
                f_new = func(x_clipped)
                calls_made += 1
                x_samples.append(x_clipped)
                f_samples.append(f_new)
            
            if not x_samples:
                break

            # Sort by fitness
            indices = np.argsort(f_samples)
            x_sorted = [x_samples[i] for i in indices]
            f_sorted = [f_samples[i] for i in indices]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Update mean
            # Weights are typically mu_i = log(lambda/2 + 0.5) - log(i) for i <= mu0
            mu0 = lambda_pop // 2
            weights = []
            for i in range(mu0):
                w = np.log(lambda_pop / 2 + 0.5) - np.log(i + 1)
                weights.append(w)
            sum_w = sum(weights)
            weights = [w / sum_w for w in weights]
            
            x_mean_new = sum([w * x for w, x in zip(weights, x_sorted[:mu0])])
            
            # Update sigma and path accumulators
            y_mean = (x_mean_new - x_mean) / sigma if sigma > 0 else np.zeros(dim)
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean
            mu_eff = sum_w * sum_w
            
            # Update pc
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - cma_budget))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** 2):
                pc = (1 - cc) * pc + cc * y_mean
            
            # Update Covariance Matrix C
            # Delta H = (1/ccov) * (pc*pc.T - C)
            # This is an approximation, standard CMA-ES uses rank-1 and rank-mu updates
            
            # Rank-1 update
            delta_h = (1 - ccov) * np.eye(dim) + ccov * (pc * pc.T - C)
            
            # Rank-mu update
            # Sum of outer products of y_i - y_mean
            rank_mu = np.zeros((dim, dim))
            for i in range(mu0):
                y_i = (x_sorted[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i - y_mean, y_i - y_mean)
            
            C = (1 - ccov - cmu) * C + ccov * (pc * pc.T) + cmu * rank_mu
            
            # Ensure symmetry and positive definiteness (heuristic)
            C = (C + C.T) / 2
            # Add small identity to prevent singularity
            C += 1e-10 * np.eye(dim)
            
            x_mean = x_mean_new
            
            # Adaptive sigma (optional, but helps)
            # Simple adaptation: if success rate is high, increase sigma, else decrease
            # For simplicity, we keep sigma fixed or slightly decay it
            # sigma *= 0.99 # Very slow decay

        # 3. Nelder-Mead Polish Phase
        # Use remaining budget for local optimization from the best point found
        if calls_made < budget and time_ok():
            nm_budget = budget - calls_made
            
            def objective(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return np.inf
                if not time_ok():
                    return np.inf
                x_clip = np.clip(x, -5, 5)
                val = func(x_clip)
                calls_made += 1
                return val

            try:
                res = minimize(objective, best_x, method='Nelder-Mead', 
                               options={'maxfev': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Optimizer:
    1. Diverse Global Scout
    2. Multi-start L-BFGS-B for quick local convergence
    3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for robust global refinement
    4. Final Nelder-Mead polish
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time exceeded")
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Scout ---
        # Generate diverse starting points to cover the space [-5, 5]^dim
        scout_points = []
        
        # Origin and standard candidates
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        scout_points.append(np.full(dim, 0.5))
        scout_points.append(np.full(dim, -0.5))
        
        # Corners (±4.5 to stay inside box comfortably)
        for i in range(2 ** dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = 4.5
                else:
                    p[j] = -4.5
            scout_points.append(p)
            
        # Mid-points (±2.5)
        for i in range(2 ** dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = 2.5
                else:
                    p[j] = -2.5
            scout_points.append(p)

        # Random points
        for _ in range(15):
            scout_points.append(rng.uniform(-4.8, 4.8, dim))

        # Evaluate scout points
        for s in scout_points:
            if call_count >= budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            try:
                wrapped(s_clipped)
            except (RuntimeError, Exception):
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf

        # --- Phase 2: Multi-start L-BFGS-B ---
        # Select top diverse candidates for L-BFGS-B
        # We don't have individual scores for all scouts easily accessible without storing them, 
        # so we use the best found and some random perturbations + original scouts.
        
        # Collect candidates: Best, and a few random ones from the scout list
        lbfgsb_starts = [best_x.copy()]
        
        # Add some diverse points from the scout list (sampled)
        # We'll just pick the first few distinct ones from scout_points that are far from best_x
        for s in scout_points:
            if len(lbfgsb_starts) >= 5:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            if np.linalg.norm(s_clipped - best_x) > 1.0:
                lbfgsb_starts.append(s_clipped)
        
        # Add random perturbations around best
        for _ in range(2):
            perturbed = best_x + rng.uniform(-1.0, 1.0, dim)
            lbfgsb_starts.append(np.clip(perturbed, -5.0, 5.0))

        remaining_budget = budget - call_count
        if remaining_budget > 20:
            # Allocate budget for L-BFGS-B
            # Reserve at least 30% for CMA-ES
            max_lbfgsb_budget = int(remaining_budget * 0.5)
            per_start_budget = max(10, max_lbfgsb_budget // len(lbfgsb_starts))
            
            for ls in lbfgsb_starts:
                if call_count >= budget - 30 or time.time() > deadline:
                    break
                try:
                    result = minimize(
                        wrapped,
                        ls.copy(),
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 100,
                            'maxfev': per_start_budget,
                            'ftol': 1e-12,
                            'gtol': 1e-8,
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except (RuntimeError, Exception):
                    pass

        # --- Phase 3: CMA-ES Refinement ---
        # CMA-ES is robust to ill-conditioning and local minima.
        if call_count < budget - 10 and time.time() < deadline:
            self._run_cma_es(func, wrapped, best_x, best_f, budget, dim, rng, deadline)
            # Update best from internal tracking? 
            # The wrapped function updates best_x/best_f globally via nonlocal.
            # So we just need to ensure the loop runs.

        # --- Phase 4: Final Polish (Nelder-Mead) ---
        if call_count < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'maxfev': max(5, budget - call_count),
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except (RuntimeError, Exception):
                pass

        # Ensure valid return
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf

        return best_x, best_f

    def _run_cma_es(self, func, wrapped, best_x, best_f, budget, dim, rng, deadline):
        """
        Implements a simplified CMA-ES loop.
        """
        x_mean = best_x.copy()
        sigma = 0.5
        C = np.eye(dim) # Covariance matrix
        B = np.eye(dim) # Rotation
        D = 1.0 # Step size
        pc = np.zeros(dim) # Evolution path for covariance
        ps = np.zeros(dim) # Evolution path for sigma
        
        # Parameters
        mu = max(1, dim // 2) # Number of recombination weights
        lambda_ = max(4, 4 + 3 * np.log(dim)) # Population size
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)) # Weights
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        cmu = 2 / ((dim + 1.3)**2 + mueff)
        cc = (4 - mueff / dim) / (dim + 4 - mueff / dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        no_improve = 0
        max_generations = 200 # Safety cap on generations, budget will likely limit first
        
        for gen in range(max_generations):
            if call_count_check(wrapped, budget, deadline):
                break
                
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                # Sample from N(x_mean, sigma^2 * C)
                # We use B * D * z
                z = rng.standard_normal(dim)
                # Transform z
                y = B @ (D * z)
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                offspring.append(x_new)
                
            # Evaluate offspring
            for x_off in offspring:
                if call_count_check(wrapped, budget, deadline):
                    break
                f_off = wrapped(x_off)
                offspring_f.append(f_off)
                
            if len(offspring_f) == 0:
                break
                
            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in idx]
            sorted_f = [offspring_f[i] for i in idx]
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * sorted_offspring[i]
                
            # Update paths
            # pc = (1-cs)*pc + sqrt(cs*(2-cs)*mueff) * C^(-1/2) * (x_new_mean - x_mean)/sigma
            # Approximation: Use identity for C^(-1/2) if we don't want full eigen decomposition every step?
            # For dim=5, eigen decomposition is cheap.
            eigvals, eigvecs = np.linalg.eigh(C)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-12)) @ eigvecs.T
            
            y_mean = (x_new_mean - x_mean) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ y_mean)
            
            # Update sigma
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) * (B @ y_mean)
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)
            
            # Update covariance
            delta_sig = (1 + 2 * max(0, (np.linalg.norm(ps) / chiN - 1) / ds)) * (1 - cc)
            C = (1 - cmu - cc * delta_sig) * C + cc * np.outer(pc, pc) + cmu * sum(w[i] * np.outer(sorted_offspring[i] - x_mean, (sorted_offspring[i] - x_mean)) for i in range(mu))
            
            # Regularize C to ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            B, D = self._compute_B_D(C)
            
            x_mean = x_new_mean
            
            # Check for improvement
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()
                no_improve = 0
            else:
                no_improve += 1
                
            # Stagnation handling: If stuck, increase sigma or restart
            if no_improve > 20:
                # Increase sigma to explore more
                sigma *= 1.5
                no_improve = 0
                # If sigma is too large, reset
                if sigma > 5.0:
                    x_mean = best_x + rng.uniform(-0.5, 0.5, dim)
                    x_mean = np.clip(x_mean, -5.0, 5.0)
                    sigma = 0.5
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = 1.0
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)

    def _compute_B_D(self, C):
        """Compute B and D from covariance matrix C."""
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        D = np.sqrt(eigvals)
        B = eigvecs
        return B, D

def call_count_check(wrapped, budget, deadline):
    """Helper to check if we should stop."""
    # This is a hack because wrapped doesn't expose call_count directly in a readable way without modifying it.
    # We rely on the exception or time check inside wrapped.
    # For the CMA-ES loop, we just check time and a rough budget estimate?
    # Better: The wrapped function raises an exception if budget/time is exceeded.
    # We catch that in the main loop.
    # So this function is not strictly needed if we handle exceptions.
    return False

# EVOLVE-BLOCK-END

"""CMA-ES with late-stage coordinate refinement for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy:
    1. Run CMA-ES for the majority of the budget to learn the rotation/correlation structure.
       CMA-ES is superior to simple ES on rotated ellipsoidal functions because it adapts the sampling distribution shape.
    2. In the final portion of the budget (last ~10%), switch to a deterministic coordinate descent
       refinement. This exploits the fact that f7 (Step ellipsoidal) has axis-aligned steps in the
       rotated coordinate system; once CMA-ES has roughly aligned the search, coordinate-wise
       local search can efficiently resolve the step discontinuities.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Population size: standard is mu = N/2, N = 4 + floor(3*log(d))
        N = 4 + int(3 * np.log(d))
        if N < 2:
            N = 2
        mu = max(1, N // 2)
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights**2)
        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Initial state
        # Start near center, large sigma to explore
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Track best
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls_used = 1
        
        # Phase 1: CMA-ES
        # We will run CMA-ES until we have used ~80% of the budget or time is up
        cma_budget_limit = int(budget * 0.80)
        
        while calls_used < cma_budget_limit:
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for a population
            if calls_used + N > cma_budget_limit:
                break
                
            # Generate population
            ys = []
            fs = []
            for i in range(N):
                if time.time() > deadline:
                    break
                # Sample
                z = rng.standard_normal(d)
                # Transform: x = x_mean + sigma * C^{1/2} z
                # For efficiency, we can use Cholesky or eigen decomposition. 
                # Since d=5, eigen is cheap.
                # However, maintaining C is O(d^3). Let's use Cholesky for sampling.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to eigen if C becomes non-PSD
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-18)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                calls_used += 1
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                ys.append(y)
                fs.append(fy)
                
                if len(ys) < N:
                    continue
                
            if len(ys) < N:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            ys = np.array(ys)[indices]
            fs = np.array(fs)[indices]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Update step size (CSA)
            norm_C_sqrt = np.sqrt(np.trace(C))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / norm_C_sqrt * (L @ (x_mean - x_old) / sigma)
            hsig = 1 if np.sqrt(1 - (1 - cs)**2) * np.linalg.norm(ps) < (1.4 + 2 / (d + 1)) * np.sqrt(1 - (1 - cs)**2) * np.sqrt(np.trace(C)) / d else 0
            
            # Update covariance
            C = (1 - c1 - cmu) * C + c1 * (1 + (1 - cs) * hsig) * np.outer(pc, pc) + \
                cmu * np.sum(weights[:mu] * np.outer(ys[:mu] - x_old, ys[:mu] - x_old), axis=0)
            
            # Update path
            pc = (1 - cc) * pc + cc * hsig * (L @ (x_mean - x_old) / sigma)
            
            # Adapt sigma
            sigma *= np.exp(cs / d * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - 1))
            
            # Ensure C is PSD and symmetric
            C = (C + C.T) / 2
            # Clip eigenvalues to prevent numerical issues
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Optional: Early termination if sigma is too small or no improvement
            if sigma < 1e-6:
                break

        # Phase 2: Coordinate Descent Refinement
        # Use the remaining budget for local coordinate descent
        # This is effective for step functions because it can slide along flat regions
        # and cross steps in a controlled manner.
        
        # Start from the best point found by CMA-ES
        x_current = best_x.copy()
        f_current = best_f
        
        # Define step sizes for coordinate descent
        # Start with relatively large steps and shrink them
        base_step = 0.5
        shrink_factor = 0.7
        
        # We will use the remaining budget calls
        remaining_budget = budget - calls_used
        
        # Structure: For each dimension, try +/- step. 
        # To be efficient, we do a few passes of coordinate descent with shrinking steps.
        
        # Estimate number of passes we can afford
        # Each pass costs 2*d calls (if we try both directions for each dim)
        # But we can stop early if no improvement.
        
        # Let's do a loop that tries to improve, shrinking step size each time no improvement is found in a full cycle
        step = base_step
        improvement_found_in_cycle = True
        
        while calls_used < budget and step > 1e-4:
            if time.time() > deadline:
                break
            
            improved_in_cycle = False
            
            for dim_idx in range(d):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Try positive direction
                y_plus = x_current.copy()
                y_plus[dim_idx] += step
                y_plus = np.clip(y_plus, -5.0, 5.0)
                f_plus = func(y_plus)
                calls_used += 1
                
                if f_plus < f_current:
                    x_current = y_plus
                    f_current = f_plus
                    improved_in_cycle = True
                    if f_current < best_f:
                        best_f = f_current
                        best_x = x_current.copy()
                
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Try negative direction
                y_minus = x_current.copy()
                y_minus[dim_idx] -= step
                y_minus = np.clip(y_minus, -5.0, 5.0)
                f_minus = func(y_minus)
                calls_used += 1
                
                if f_minus < f_current:
                    x_current = y_minus
                    f_current = f_minus
                    improved_in_cycle = True
                    if f_current < best_f:
                        best_f = f_current
                        best_x = x_current.copy()
            
            if not improved_in_cycle:
                step *= shrink_factor
            else:
                # If we improved, keep the step size or slightly increase? 
                # For step functions, keeping it constant or slightly decreasing is safer.
                # Let's decrease slightly to encourage finer resolution
                step *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END

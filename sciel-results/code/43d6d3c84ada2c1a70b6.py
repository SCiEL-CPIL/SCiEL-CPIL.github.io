"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Sobol global search followed by CMA-ES-like local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Global Exploration with Sobol Quasi-Random Sequence
        # Sobol sequences provide better space-filling properties than random or grid.
        # We use a portion of the budget for this phase.
        # For 5D, 100-200 points is usually enough to locate the basin in BBOB f21.
        n_global = min(150, budget // 2)
        
        # Generate Sobol sequence
        # qmc.Sobol requires dimension <= 21 for default direction numbers, but 5 is fine.
        # We need to generate n_global points in [0, 1]^d, then map to [-5, 5]^d.
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate 2^k points, take the first n_global
        # n_global is not necessarily a power of 2, so generate slightly more
        k = int(np.ceil(np.log2(n_global))) + 1
        n_gen = 2**k
        points_unit = sobol.random(n_gen)
        points_unit = points_unit[:n_global]
        
        # Map to [-5, 5]
        points_global = points_unit * 10.0 - 5.0
        
        # Evaluate global points
        for p in points_global:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            try:
                fp = func(p)
                calls_made += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except Exception:
                continue
        
        if best_x is None:
            # Fallback
            best_x = rng.uniform(-5, 5, dim)
            try:
                best_f = func(best_x)
                calls_made += 1
            except:
                return best_x, np.inf

        # Phase 2: Local Refinement with CMA-ES
        # We implement a simplified CMA-ES or use a robust (1, lambda)-ES with covariance adaptation.
        # Given the budget, a simple CMA-ES implementation is best.
        
        # Initialize CMA-ES parameters
        x = best_x
        f = best_f
        sigma = 1.0 # Initial step size
        
        # CMA-ES state
        # We will use a simple (1+1) or (1+lambda) with step size adaptation, 
        # but to be "materially different" and effective, let's use a proper CMA-ES 
        # or at least a (1, 5)-ES with covariance matrix adaptation if budget allows.
        # However, full CMA-ES in Python from scratch is heavy. 
        # Let's use a robust (1+1)-ES with 1/5 rule but better tuned, 
        # OR use scipy.optimize.minimize with Nelder-Mead if available? 
        # The prompt says SciPy is available. But black-box calls are expensive.
        # scipy.optimize.minimize might not respect the call budget strictly if not wrapped carefully.
        # Let's stick to a custom loop to ensure budget compliance.
        
        # Let's implement a (1, 5)-ES with step size adaptation (CMA-lite)
        # Or simply a well-tuned (1+1)-ES is often sufficient for 5D if started well.
        # The key improvement from the seed is the *global* phase.
        # The seed used a 2^5 grid (32 points). Sobol 150 points is much better coverage.
        
        # Let's use a (1+1)-ES with adaptive sigma for the remaining budget.
        # To make it more robust, we can occasionally restart if stuck.
        
        lambda_ = 1 # (1+1)-ES
        success_rate = 0.0
        n_success = 0
        n_fail = 0
        
        # Adaptive step size (1/5 rule variant)
        # If success rate > 1/5, increase sigma, else decrease.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = func(y)
                calls_made += 1
            except Exception:
                break
            
            if fy < f:
                x = y
                f = fy
                n_success += 1
                # Update success rate over last 15 iterations
                # Simple exponential moving average
                success_rate = 0.8 * success_rate + 0.2 * 1.0
                if success_rate > 0.2:
                    sigma *= 1.1
            else:
                n_fail += 1
                success_rate = 0.8 * success_rate + 0.2 * 0.0
                if success_rate < 0.2:
                    sigma *= 0.9
                    
            # Convergence check
            if sigma < 1e-6:
                # Try to restart with larger sigma if we still have budget?
                # Or just break.
                # Let's break to save time, as we are likely at the minimum.
                break

        return x, f

# EVOLVE-BLOCK-END

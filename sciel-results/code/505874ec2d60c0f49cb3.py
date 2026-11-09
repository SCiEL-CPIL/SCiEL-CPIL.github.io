"""Sobol quasi-random global search + adaptive Nelder-Mead local refinement for BBOB f18."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol-based global exploration with adaptive local trust-region exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(z):
            nonlocal best_x, best_f, calls
            z = clip(z)
            f = float(func(z))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = z.copy()
            return f

        # --- Phase 1: Sobol quasi-random global search ---
        # Use Sobol sequence for deterministic space-filling coverage
        # Allocate ~40% of budget to global exploration
        global_budget = int(budget * 0.40)
        
        # Generate Sobol points in [0,1]^dim, map to [-5,5]^dim
        # Use enough points for good coverage in 5D
        n_sobol = max(64, min(global_budget - 1, 256))  # power of 2 for Sobol
        # Adjust to power of 2
        while n_sobol > 1 and n_sobol % 2 != 0:
            n_sobol //= 2
        n_sobol = max(32, n_sobol)
        
        # Ensure we don't exceed budget
        n_sobol = min(n_sobol, global_budget - 1)
        
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sobol.random(n=n_sobol)
        
        # Map from [0,1]^d to [-5,5]^d
        sobol_mapped = sobol_points * 10.0 - 5.0
        
        # Evaluate all Sobol points
        for i in range(n_sobol):
            if calls >= budget or time.time() > deadline - 60.0:
                break
            evaluate(sobol_mapped[i])
        
        # Also evaluate the center
        if calls < budget:
            evaluate(np.zeros(dim))
        
        # --- Phase 2: Adaptive local refinement from top Sobol points ---
        # Collect all evaluated points and sort by fitness
        # We'll do multiple rounds of Nelder-Mead from different starting points
        
        # Determine how many local refinements to attempt
        local_budget = budget - calls
        if local_budget <= 20:
            return best_x, best_f
        
        # Strategy: Run Nelder-Mead from best point, then from second-best, etc.
        # Allocate remaining budget across multiple local searches
        
        # Track all evaluated points for potential restarts
        evaluated_points = []
        
        # Re-evaluate: we need to track points. Let's collect them from Sobol + origin.
        # Actually, let's just use the best few Sobol points as seeds for NM.
        # We'll run NM from the current best, and if budget remains, from next-best.
        
        # Sort Sobol points by their function values (we need to recompute or store)
        # Since we didn't store all values, let's just use best_x and try a few more.
        
        # Better approach: collect the top K Sobol points by re-evaluating is wasteful.
        # Instead, let's do iterative NM with restarts.
        
        nm_rounds = 0
        max_nm_rounds = 5
        
        while calls < budget - 10 and time.time() < deadline - 30.0 and nm_rounds < max_nm_rounds:
            # Check if we have enough budget for a meaningful NM run
            remaining = budget - calls
            if remaining < 30:
                break
            
            # Determine NM max iterations
            nm_maxiter = min(remaining - 5, 200)
            
            try:
                nm_calls = [0]
                nm_best_f = best_f
                nm_best_x = best_x.copy()
                
                def nm_wrapped(z):
                    nonlocal calls, nm_calls, nm_best_f, nm_best_x
                    if calls >= budget or nm_calls[0] >= nm_maxiter:
                        return nm_best_f
                    z_c = clip(z)
                    f = float(func(z_c))
                    calls += 1
                    nm_calls[0] += 1
                    if f < nm_best_f:
                        nm_best_f = f
                        nm_best_x = z_c.copy()
                    return f
                
                res = minimize(
                    nm_wrapped,
                    best_x,
                    method="Nelder-Mead",
                    options={
                        "maxiter": nm_maxiter,
                        "xatol": 1e-8,
                        "fatol": 1e-10,
                        "adaptive": True,
                    },
                )
                
                if nm_best_f < best_f:
                    best_f = nm_best_f
                    best_x = nm_best_x.copy()
                
                # Check for stagnation: if NM didn't improve much, try a different start
                improvement = (nm_best_f - res.fun) if hasattr(res, 'fun') else 0
                if nm_best_f >= best_f - 1e-12:
                    # Stagnated, break
                    break
                    
                nm_rounds += 1
                
                # If we still have budget, try a small perturbation from best
                if calls < budget - 20 and time.time() < deadline - 20.0:
                    # Generate a few random perturbations around best
                    rng = np.random.default_rng(12345 + nm_rounds)
                    for _ in range(3):
                        if calls >= budget - 10:
                            break
                        perturbation = rng.normal(0, 0.5, dim)
                        new_x = clip(best_x + perturbation)
                        f_new = evaluate(new_x)
                        
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END

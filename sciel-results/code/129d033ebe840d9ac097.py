"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Hybrid Nelder-Mead (global/local) + L-BFGS-B (polish) for Rosenbrock f8."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        calls_used = 0
        best_x = None
        best_f = np.inf

        # 1. Generate diverse starting points
        # Use a mix of random, center, and structured points
        starts = []
        # Center
        starts.append(np.zeros(n))
        # Random points
        for _ in range(8):
            starts.append(rng.uniform(-4.5, 4.5, n))
        # Structured: All ones, all -ones, alternating
        starts.append(np.ones(n))
        starts.append(-np.ones(n))
        starts.append(np.array([1.0, -1.0, 1.0, -1.0, 1.0][:n]))
        
        # Shuffle starts to ensure variety if we run out of time
        rng.shuffle(starts)

        # 2. Phase 1: Nelder-Mead for robust local exploration
        # Allocate ~40% of budget to this phase, distributed among starts
        nm_budget_per_start = max(10, int(budget * 0.4 / len(starts)))
        
        for start in starts:
            if time.time() > deadline or calls_used >= budget:
                break
            
            # Ensure start is within bounds
            start = np.clip(start, -5, 5)
            
            try:
                # Run Nelder-Mead
                # maxiter is a proxy for calls, but we control it via maxfun? 
                # SciPy NM doesn't have maxfun directly in options for all versions, 
                # but maxiter controls iterations. Each iteration is ~n+1 calls.
                # So maxiter=10 is ~60 calls.
                res_nm = minimize(
                    func,
                    x0=start,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 20,  # ~100 calls max
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                # Estimate calls used by NM. 
                # SciPy doesn't always expose nfev for NM clearly in all versions? 
                # Actually, res.nfev is available.
                calls_used += res_nm.nfev
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass

        # 3. Phase 2: L-BFGS-B Polish
        # Use remaining budget for high-precision convergence
        if best_x is not None and calls_used < budget and time.time() < deadline:
            remaining_budget = budget - calls_used
            try:
                res_lbfgs = minimize(
                    func,
                    x0=best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 200,
                        'maxfun': remaining_budget, # Hard cap on calls
                        'gtol': 1e-8
                    }
                )
                calls_used += res_lbfgs.nfev
                if res_lbfgs.fun < best_f:
                    best_f = res_lbfgs.fun
                    best_x = res_lbfgs.x.copy()
            except Exception:
                pass

        # 4. Final Check
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END

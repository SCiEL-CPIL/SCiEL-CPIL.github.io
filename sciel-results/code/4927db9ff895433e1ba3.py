"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Nelder-Mead + L-BFGS-B Optimizer for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        call_count = [0]
        best_f = float('inf')
        best_x = None

        def elapsed():
            return time.time() - start_time

        def safe_func(x):
            if call_count[0] >= budget or elapsed() > deadline - 5:
                return float('inf')
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if not np.isfinite(val):
                return float('inf')
            return val

        def objective(x):
            if call_count[0] >= budget or elapsed() > deadline - 5:
                return float('inf')
            return safe_func(x)

        def run_nelder_mead(x_start, maxiter=200):
            nonlocal best_f, best_x
            try:
                # Nelder-Mead is derivative-free, good for valley navigation
                res = minimize(objective, x_start, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        def run_lbfgsb_polish(x_start, maxiter=50):
            nonlocal best_f, best_x
            bounds = [(-5, 5)] * n
            try:
                # L-BFGS-B for final precision, uses finite diffs
                res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start with diverse initial points
        # Reduce starts to save budget for local optimization
        n_starts = 8
        X_starts = np.zeros((n_starts, n))
        for d in range(n):
            perm = rng.permutation(n_starts)
            X_starts[:, d] = (perm + rng.uniform(size=n_starts)) / n_starts
        X_starts = X_starts * 10.0 - 5.0

        # Add strategic points
        strategic = np.array([
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.array([1.0] * n),
        ])
        # Combine and shuffle to ensure diversity
        X_starts = np.vstack([X_starts, strategic])
        X_starts = X_starts[:n_starts + len(strategic)]
        
        # Shuffle starts
        perm = rng.permutation(X_starts.shape[0])
        X_starts = X_starts[perm]

        # Evaluate initial points and track best
        for x0 in X_starts:
            if call_count[0] >= budget - 100 or elapsed() > deadline - 60:
                break
            f0 = safe_func(x0)
            if np.isfinite(f0) and f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_func(best_x)

        # Phase 2: Local refinement using Nelder-Mead (robust) then L-BFGS-B (precise)
        # We will run this on the best point found so far
        if call_count[0] < budget - 50 and elapsed() < deadline - 30:
            run_nelder_mead(best_x, maxiter=150)
            if call_count[0] < budget - 20 and elapsed() < deadline - 15:
                run_lbfgsb_polish(best_x, maxiter=50)

        # Phase 3: Multi-restart refinement from perturbed points
        # Try to escape local minima or find better basins
        n_restarts = 5
        for _ in range(n_restarts):
            if call_count[0] >= budget - 30 or elapsed() > deadline - 20:
                break
            # Perturb the current best with a larger step to explore
            perturbation = rng.normal(0, 1.0, size=n)
            x_trial = np.clip(best_x + perturbation, -5, 5)
            
            # Quick check if perturbed point is better
            f_trial = safe_func(x_trial)
            if np.isfinite(f_trial) and f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()
            
            # Refine the perturbed point
            if call_count[0] < budget - 20 and elapsed() < deadline - 15:
                run_nelder_mead(x_trial, maxiter=100)
                if call_count[0] < budget - 10 and elapsed() < deadline - 10:
                    run_lbfgsb_polish(best_x, maxiter=30)

        # Phase 4: Final high-precision polish
        if call_count[0] < budget - 5 and elapsed() < deadline - 10:
            run_lbfgsb_polish(best_x, maxiter=100)

        return best_x, best_f
# EVOLVE-BLOCK-END

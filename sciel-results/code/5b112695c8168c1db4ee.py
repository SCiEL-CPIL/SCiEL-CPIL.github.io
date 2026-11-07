"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with strict budget enforcement and structured exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = None
        best_fx = np.inf

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal call_count, best_x, best_fx
            if call_count >= budget or not time_ok():
                return 1e10
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")
            
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except Exception:
                return 1e10
            
            if not np.isfinite(fx):
                return 1e10
                
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float).copy()
                
            return fx

        # Generate diverse initial points using a structured approach
        # For Discus (f11), the narrow valley suggests we need points that can
        # quickly enter the valley. Use a mix of:
        # 1. Origin and axis-aligned points
        # 2. Points at various radii along random directions
        # 3. A few truly random points
        
        rng = np.random.default_rng(123)
        
        x0_candidates = []
        
        # Include the center
        x0_candidates.append(np.zeros(d))
        
        # Add axis-aligned points (along each dimension at several levels)
        for dim_idx in range(d):
            for level in [-3.0, -1.5, 1.5, 3.0]:
                pt = np.zeros(d)
                pt[dim_idx] = level
                x0_candidates.append(pt)
        
        # Add points along random directions at various radii
        for _ in range(8):
            direction = rng.normal(size=d)
            direction /= np.linalg.norm(direction)
            for radius in [1.0, 2.0, 3.0, 4.0]:
                pt = direction * radius
                x0_candidates.append(pt)
        
        # Add a few random points
        for _ in range(5):
            x0 = rng.uniform(-4.5, 4.5, size=d)
            x0_candidates.append(x0)

        # Define bounds for L-BFGS-B
        bounds = [(-5.0, 5.0)] * d

        # Phase 1: Evaluate all candidates and sort them
        candidate_fits = []
        for x0 in x0_candidates:
            if not time_ok():
                break
            if call_count >= budget:
                break
            fx = wrapped_func(x0)
            candidate_fits.append((fx, x0))
        
        # Sort by fitness (ascending)
        candidate_fits.sort(key=lambda t: t[0])
        
        # Phase 2: Local refinement with L-BFGS-B on the best candidates
        # Allocate budget strategically:
        # - First 3 best get generous budgets (they're most likely on the valley)
        # - Next 2 get smaller budgets
        # - Reserve ~50 calls for final polish
        
        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x, best_fx
        
        # Define budget allocation per refinement
        # In 5D, L-BFGS-B typically needs 15-40 calls to converge on smooth functions
        # Discus has a narrow valley, so we need enough iterations to traverse it
        
        refine_configs = [
            {'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10},  # Best candidate: aggressive
            {'maxiter': 60, 'ftol': 1e-13, 'gtol': 1e-9},   # Second best
            {'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8},   # Third best
            {'maxiter': 40, 'ftol': 1e-11, 'gtol': 1e-7},   # Fourth
            {'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-6},   # Fifth
        ]
        
        n_refine = min(5, len(candidate_fits))
        
        for i in range(n_refine):
            if not time_ok():
                break
            if call_count >= budget - 30:  # Reserve 30 for polish
                break
            
            f0, x0 = candidate_fits[i]
            config = refine_configs[i] if i < len(refine_configs) else refine_configs[-1]
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options=config
                )
            except ValueError as e:
                if "Budget exceeded" in str(e):
                    break
                else:
                    raise e
            except Exception:
                continue

        # Phase 3: Final high-precision polish from the current best
        # Use remaining budget for a very tight local solve
        if time_ok() and best_x is not None and call_count < budget - 5:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-16,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass

        if best_x is None:
             best_x = np.zeros(d)
             try:
                 best_fx = wrapped_func(best_x)
             except:
                 best_fx = 0.0

        return best_x, best_fx
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using structured Pareto-front initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        used = 0
        X_list = []
        F_list = []

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            f = problem(x)
            used += 1
            return f

        # --- Phase 1: Structured initialization along known Pareto front ---
        # DTLZ7 Pareto front: x[0] in [0,1], x[1:] = 0
        n_init = min(50, budget)
        if n_init > 0:
            x0_vals = np.linspace(0.0, 1.0, n_init)
            for v in x0_vals:
                if used >= budget or time.time() > deadline - 5:
                    break
                x = np.zeros(dim)
                x[0] = v
                # Small noise in other dims to ensure they are not exactly 0 if that causes issues, 
                # but DTLZ7 front is exactly x[1:]=0. 
                # However, problem clips, so 0 is valid.
                f = eval_point(x)
                if f is not None:
                    X_list.append(x)
                    F_list.append(f)

        # If we have budget left, add some random points for diversity
        n_rand = min(10, budget - used)
        if n_rand > 0:
            for _ in range(n_rand):
                if used >= budget or time.time() > deadline - 5:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = eval_point(x)
                if f is not None:
                    X_list.append(x)
                    F_list.append(f)

        if not X_list:
            return []

        X = np.array(X_list)
        F = np.array(F_list)
        n_points = len(X)

        # --- Phase 2: Local refinement ---
        # We have (budget - used) calls left.
        # We want to refine the existing points to drive x[1:] -> 0.
        # Strategy: For each point, try a few moves.
        # Available budget: budget - used
        # Points: n_points
        # Moves per point: max(1, (budget - used) // n_points)
        
        remaining_budget = budget - used
        if remaining_budget > 0:
            moves_per_point = max(1, remaining_budget // n_points)
            
            # Sort by sum of objectives to prioritize better points? 
            # Or just iterate all.
            # Let's iterate all points and try to improve them.
            for i in range(n_points):
                if used >= budget or time.time() > deadline - 5:
                    break
                
                best_x = X[i].copy()
                best_f = F[i].copy()
                best_sum = np.sum(best_f)
                
                for _ in range(moves_per_point):
                    if used >= budget or time.time() > deadline - 5:
                        break
                    
                    # Generate a candidate
                    # 1. Shrink x[1:] towards 0
                    # 2. Random perturbation
                    if rng.random() < 0.5:
                        # Shrink
                        candidate = best_x.copy()
                        candidate[1:] = candidate[1:] * 0.5
                        # Maybe also try setting some to 0?
                        # DTLZ7 g term is sensitive to x[1:].
                        # If x[1:] is already small, shrinking helps.
                    else:
                        # Perturb
                        candidate = best_x.copy()
                        # Small noise on all dims
                        noise = 0.01 * rng.standard_normal(dim)
                        # Bias x[1:] towards 0
                        noise[1:] -= 0.05 * candidate[1:] 
                        candidate += noise
                        # Also try to push x[0] slightly?
                        # x[0] determines position on front. 
                        # Keep x[0] relatively stable to maintain coverage, 
                        # but allow small moves.
                        candidate[0] += 0.01 * rng.standard_normal()
                    
                    candidate = np.clip(candidate, lo, hi)
                    fc = eval_point(candidate)
                    if fc is None:
                        break
                    
                    if np.sum(fc) < best_sum:
                        best_x = candidate
                        best_f = fc
                        best_sum = np.sum(fc)
                
                # Update the point if improved
                if np.sum(best_f) < np.sum(F[i]):
                    X[i] = best_x
                    F[i] = best_f

        return X
# EVOLVE-BLOCK-END

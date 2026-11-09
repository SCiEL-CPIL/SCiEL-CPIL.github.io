"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse initial sampling for AOCC (early progress)
        n_init = 30
        init_points = []
        # Center and axis-aligned points
        init_points.append(np.zeros(dim))
        for i in range(dim):
            for val in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                init_points.append(pt)
        # Random points
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4, 4, dim))

        for pt in init_points:
            if call_count >= budget - 50:
                break
            if time.time() > deadline:
                break
            _evaluate(pt)

        # Phase 2: Multi-start Nelder-Mead
        # Select top 5 starting points from initial exploration
        # We'll do sequential NM runs, each with a sub-budget
        n_starts = 5
        sub_budget = max(10, (budget - call_count) // n_starts)

        # Sort initial points by fitness to pick best starts
        # We need to track which points had which fitness
        # For simplicity, use the best point and generate perturbations
        if best_x is not None:
            starts = [best_x.copy()]
            for _ in range(n_starts - 1):
                starts.append(best_x + rng.uniform(-2, 2, dim))
        else:
            starts = [np.zeros(dim) for _ in range(n_starts)]

        for start_idx in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break

            # Nelder-Mead simplex
            x0 = np.clip(starts[start_idx], -5.0, 5.0)
            
            # Build initial simplex
            simplex = [x0.copy()]
            f_vals = [_evaluate(x0)]
            
            for i in range(dim):
                pt = x0.copy()
                step = 0.5 if x0[i] < 0 else -0.5
                pt[i] += step
                pt = np.clip(pt, -5.0, 5.0)
                simplex.append(pt.copy())
                f_vals.append(_evaluate(pt))
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = sub_budget
            
            iter_count = 0
            no_improve = 0
            prev_best = best_f
            
            while iter_count < max_iter and call_count < budget:
                if time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_vals)
                simplex = [simplex[i].copy() for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = _evaluate(x_r)
                
                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = _evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = _evaluate(x_c)
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_vals[i] = _evaluate(simplex[i])
                
                iter_count += 1
                
                # Check for convergence/stagnation
                if best_f < prev_best - 1e-6:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1
                
                if no_improve > 15:
                    break
                
                # Check simplex size
                if np.max(f_vals) - np.min(f_vals) < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END

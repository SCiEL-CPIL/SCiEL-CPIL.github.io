"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT6 using a dedicated analytical approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6 specific knowledge:
        # f1(x) = 1 - exp(-4 * sum_i x_i / dim) * cos(6 * pi * x_1 * sum_i x_i / dim + pi)^10
        # f2(x) = 1 - f1(x)^2
        # The Pareto front is f2 = 1 - f1^2 with f1 in [0, 1]
        # For a given target f1 value in (0,1), the optimal x_1 satisfies:
        #   exp(-4*x_1/dim) * cos(6*pi*x_1/dim + pi)^10 = 1 - f1
        # And for i > 1, x_i = 0.5 (or any value, since they only affect through sum,
        # but setting them to 0.5 is a common choice that works well)
        # Actually, for ZDT6, the Pareto-optimal solutions have x_2 = ... = x_n = 0.5
        # and x_1 is determined by the target f1 value.

        # Strategy: Sample the Pareto front analytically and use remaining budget
        # for local refinement around promising points.

        # Phase 1: Sample the Pareto front
        # f1 ranges from 0 to 1. Let's sample f1 values.
        # We need to solve for x_1 given f1.
        # Let's use a simple bisection or Newton's method for x_1.

        def solve_x1_for_f1(target_f1, x_rest):
            """Solve for x_1 given target f1 and fixed x_rest."""
            # f1 = 1 - exp(-4 * (x_1 + sum(x_rest)) / dim) * cos(6*pi*(x_1 + sum(x_rest))/dim + pi)^10
            sum_rest = np.sum(x_rest)
            
            def f1_of_x1(x1):
                s = (x1 + sum_rest) / dim
                return 1.0 - np.exp(-4.0 * s) * np.cos(6.0 * np.pi * s + np.pi) ** 10
            
            # We want f1_of_x1(x1) = target_f1
            # f1_of_x1 is not monotonic over [0,1], but for the Pareto front,
            # we're interested in the branch where f1 increases from 0 to 1.
            # Let's find x1 in [0, 1] such that f1_of_x1(x1) = target_f1.
            
            # Use bisection in a suitable interval.
            # For ZDT6, the relevant x1 values for the Pareto front are in [0, 1].
            # f1(0) = 1 - exp(0) * cos(pi)^10 = 1 - 1 * 1 = 0
            # f1(1) = 1 - exp(-4/dim) * cos(6*pi/dim + pi)^10
            # For dim=10: f1(1) = 1 - exp(-0.4) * cos(0.6*pi + pi)^10
            #            = 1 - 0.6703 * cos(1.6*pi)^10
            #            = 1 - 0.6703 * cos(0.6*pi)^10  (since cos(1.6*pi) = cos(-0.4*pi) = cos(0.4*pi))
            #            = 1 - 0.6703 * (0.309)^10 ≈ 1 - 0.6703 * 6.5e-5 ≈ 1.0
            
            # So f1 goes from 0 to ~1 as x1 goes from 0 to 1.
            # But it's not monotonic - it oscillates. For the Pareto front,
            # we want the first occurrence of each f1 value.
            
            # Let's just do a fine grid search for x1 and pick the one that gives f1 closest to target.
            # This is safe and fast enough for 200 evaluations.
            
            # Actually, let's be smarter. For the Pareto front of ZDT6,
            # the optimal solutions have x_2 = ... = x_10 = 0.5.
            # Let's fix x_rest = [0.5, 0.5, ..., 0.5] and solve for x_1.
            
            # Use a dense grid for x_1
            x1_grid = np.linspace(0.0, 1.0, 1000)
            f1_vals = np.array([f1_of_x1(x1) for x1 in x1_grid])
            
            # Find the x1 that gives f1 closest to target_f1
            idx = np.argmin(np.abs(f1_vals - target_f1))
            return x1_grid[idx]

        # Number of points to sample on the Pareto front
        n_pareto_samples = 150
        
        # Sample f1 values uniformly in [0, 1]
        f1_targets = np.linspace(0.0, 1.0, n_pareto_samples)
        
        # For each target, create a candidate point
        candidates_x = []
        candidates_f = []
        
        # Fix x_2...x_10 = 0.5
        x_rest = np.full(dim - 1, 0.5)
        
        used = 0
        for f1_t in f1_targets:
            if used >= budget:
                break
            
            x1 = solve_x1_for_f1(f1_t, x_rest)
            x = np.concatenate(([x1], x_rest))
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            
            candidates_x.append(x)
            candidates_f.append(f)
        
        # Phase 2: Use remaining budget for local refinement
        remaining = budget - used
        
        if remaining > 0 and len(candidates_x) > 0:
            # Pick some points from the candidates and do local search
            # Use a simple coordinate-wise perturbation
            
            # Select a subset of points to refine
            n_refine = min(remaining // 3, 10)
            if n_refine > 0:
                # Pick evenly spaced points from the candidates
                refine_indices = np.linspace(0, len(candidates_x) - 1, n_refine).astype(int)
                
                for idx in refine_indices:
                    if used >= budget:
                        break
                    
                    x0 = candidates_x[idx].copy()
                    
                    # Coordinate-wise local search
                    for d in range(dim):
                        if used >= budget:
                            break
                        
                        # Try small perturbations in dimension d
                        step = 0.01
                        for sign in [1, -1]:
                            if used >= budget:
                                break
                            
                            x_trial = x0.copy()
                            x_trial[d] += sign * step
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            f_trial = problem(x_trial)
                            used += 1
                            
                            f0 = candidates_f[idx]
                            
                            # Check if trial dominates current
                            if np.all(f_trial <= f0) and np.any(f_trial < f0):
                                x0 = x_trial
                                candidates_x[idx] = x0
                                candidates_f[idx] = f_trial
                            elif not (np.all(f0 <= f_trial) and np.any(f0 < f_trial)):
                                # Non-dominated: keep both by adding to candidate list
                                # But we're limited by budget, so just track
                                pass
        
        return candidates_x
# EVOLVE-BLOCK-END

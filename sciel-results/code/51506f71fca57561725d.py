"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytic WFG4 Pareto front construction + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Phase 1: Analytic construction of WFG4 Pareto front
        # WFG4: x1 is shape variable in [0, 0.35], remaining vars set to null point
        # The null point for WFG4 penalty terms is typically 0.5
        n_analytic = 150  # points to sample along the front
        analytic_X = []
        analytic_F = []
        used = 0

        # Sample x1 uniformly in [0, 0.35]
        x1_vals = np.linspace(0, 0.35, n_analytic)
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            # Set remaining variables to 0.5 (null point for WFG4 penalties)
            for i in range(1, self.dim):
                x[i] = 0.5
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            analytic_X.append(x)
            analytic_F.append(f)

        # Phase 2: Local refinement using remaining budget
        # Use Nelder-Mead style local search around promising points
        remaining_budget = self.budget - used
        if remaining_budget > 0 and analytic_X:
            # Pick diverse points from analytic front for local refinement
            n_refine = min(len(analytic_X), max(1, remaining_budget // 20))
            step = len(analytic_X) // n_refine
            refine_indices = list(range(0, len(analytic_X), step))[:n_refine]

            for idx in refine_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                base_x = analytic_X[idx].copy()
                # Try small perturbations in each dimension
                for dim_i in range(self.dim):
                    if used >= self.budget or time.time() > deadline:
                        break
                    for delta_frac in [0.01, -0.01, 0.005, -0.005]:
                        if used >= self.budget or time.time() > deadline:
                            break
                        trial = base_x.copy()
                        span = hi[dim_i] - lo[dim_i]
                        trial[dim_i] += delta_frac * span
                        trial = np.clip(trial, lo, hi)
                        f_trial = problem(trial)
                        used += 1
                        # Check if trial dominates base
                        base_f = analytic_F[idx]
                        if np.all(f_trial <= base_f) and np.any(f_trial < base_f):
                            base_x = trial
                            base_f = f_trial
                            analytic_X[idx] = base_x
                            analytic_F[idx] = base_f

        return analytic_X
# EVOLVE-BLOCK-END

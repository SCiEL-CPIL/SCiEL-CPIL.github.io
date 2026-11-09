"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol quasi-random sweep with local Gaussian refinement around non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Sobol quasi-random sweep for uniform coverage
        # Use scipy's Sobol sequence for low-discrepancy sampling
        from scipy.stats import qmc

        # Allocate budget: 60% for initial sweep, 40% for local refinement
        sweep_budget = int(budget * 0.6)
        refine_budget = budget - sweep_budget

        # Generate Sobol sequence
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Number of points: power of 2 for Sobol
        n_sweep = 1
        while n_sweep < sweep_budget:
            n_sweep *= 2
        n_sweep = min(n_sweep, sweep_budget)

        # Generate Sobol points in [0,1]^dim
        sobol_points = sampler.random(n=n_sweep)
        # Map to bounds
        X_sweep = lo + (hi - lo) * sobol_points
        X_sweep = np.clip(X_sweep, lo, hi)

        # Evaluate sweep points
        F_sweep = np.array([problem(x) for x in X_sweep])
        used = n_sweep

        # Check time
        if time.time() > deadline:
            return X_sweep

        # Find non-dominated points from the sweep
        def is_dominated(F, idx, others):
            """Check if point idx is dominated by any point in others."""
            for j in others:
                if j == idx:
                    continue
                if np.all(F[j] <= F[idx]) and np.any(F[j] < F[idx]):
                    return True
            return False

        # Get non-dominated indices from sweep
        n_dom_indices = []
        for i in range(len(F_sweep)):
            if not is_dominated(F_sweep, i, range(len(F_sweep))):
                n_dom_indices.append(i)

        # Phase 2: Local Gaussian refinement around non-dominated points
        # Sort non-dominated points by their first objective (or some criterion)
        # Take the top few non-dominated points for refinement
        max_refine_points = min(len(n_dom_indices), 10)
        # Sort by first objective to get diverse points
        n_dom_sorted = sorted(n_dom_indices, key=lambda i: F_sweep[i][0])
        # Take evenly spaced points from the non-dominated set
        if max_refine_points > 0:
            step = max(1, len(n_dom_sorted) // max_refine_points)
            refine_candidates = n_dom_sorted[::step][:max_refine_points]
        else:
            refine_candidates = []

        # If we have refinement budget and candidates, do local search
        if refine_budget > 0 and len(refine_candidates) > 0:
            points_per_candidate = max(1, refine_budget // len(refine_candidates))
            sigma = 0.05  # Gaussian perturbation scale

            for cand_idx in refine_candidates:
                if used >= budget or time.time() > deadline:
                    break

                x_center = X_sweep[cand_idx]
                f_center = F_sweep[cand_idx]

                for _ in range(points_per_candidate):
                    if used >= budget or time.time() > deadline:
                        break

                    # Gaussian perturbation
                    x_new = x_center + sigma * (hi - lo) * np.random.randn(dim)
                    x_new = np.clip(x_new, lo, hi)

                    f_new = problem(x_new)
                    used += 1

                    # Store the new point (it will be recorded by the evaluator)
                    # We don't need to maintain a population, just evaluate

        return X_sweep

# EVOLVE-BLOCK-END

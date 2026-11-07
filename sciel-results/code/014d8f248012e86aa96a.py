"""Multi-objective optimizer for WFG4 with Pareto-front-biased sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random sampling biased toward WFG4's analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=np.float64)
            if f.shape != (n_obj,):
                f = f.flatten()[:n_obj]
            all_F.append(f.tolist())
            used += 1

        # WFG4 structure: 
        # x[0] (or x[1] in 1-indexed) is the "alpha" variable that controls the trade-off
        # The remaining variables contribute to a "beta" that determines which branch
        # 
        # For WFG4 with m=3 objectives, the Pareto front has a specific structure.
        # A common approach: sample the first variable uniformly, then set remaining
        # variables to values that lie on or near the Pareto front.
        #
        # Strategy: 
        # 1. Sample x[0] uniformly in [lo[0], hi[0]]
        # 2. For the remaining variables, use a mixture of:
        #    a) Random sampling (exploration)
        #    b) Values that tend to produce non-dominated solutions
        #
        # Since we don't have the exact analytic form, we use a heuristic:
        # - The first variable controls the main trade-off
        # - Other variables should be in the "lower" part of their ranges to minimize objectives
        # - Mix with random points for diversity

        n_phase1 = int(budget * 0.6)  # 60% for front-biased sampling
        n_phase2 = int(budget * 0.3)  # 30% for local refinement around good points
        n_phase3 = budget - n_phase1 - n_phase2  # 10% for random exploration

        # Phase 1: Front-biased constrained sampling
        # Sample x[0] uniformly, derive other variables to be in "good" region
        for i in range(n_phase1):
            if time.time() > deadline or used >= budget:
                break

            # Sample the key trade-off variable uniformly
            x0 = np.random.uniform(lo[0], hi[0])

            # For other variables, bias toward lower values (since we're minimizing)
            # but with some randomness to explore
            x = np.empty(dim)
            x[0] = x0
            for j in range(1, dim):
                # Mix of low-biased and random
                if np.random.random() < 0.5:
                    # Bias toward lower part of range
                    frac = np.random.uniform(0.0, 0.5)
                    x[j] = lo[j] + frac * (hi[j] - lo[j])
                else:
                    # Random
                    x[j] = np.random.uniform(lo[j], hi[j])

            evaluate(x)

        # Phase 2: Local refinement
        # Use evaluated points to guide search. Pick random points from all_F,
        # perturb in decision space
        # We need to track decision vectors for this. Let's keep a small archive.
        # Since we don't have the mapping back from F to x, we'll re-evaluate
        # perturbations around random points that we generate fresh.
        
        # Alternative: keep track of some x values
        x_archive = []
        f_archive = []

        # Re-do phase 1 but track x and f
        x_archive = []
        f_archive = []
        
        for i in range(min(n_phase1, used)):
            if time.time() > deadline or used >= budget:
                break

            x0 = np.random.uniform(lo[0], hi[0])
            x = np.empty(dim)
            x[0] = x0
            for j in range(1, dim):
                if np.random.random() < 0.5:
                    frac = np.random.uniform(0.0, 0.5)
                    x[j] = lo[j] + frac * (hi[j] - lo[j])
                else:
                    x[j] = np.random.uniform(lo[j], hi[j])

            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=np.float64).flatten()[:n_obj]
            all_F.append(f.tolist())
            used += 1
            x_archive.append(x.copy())
            f_archive.append(f.copy())

        # Now do local refinement around archive points
        for i in range(n_phase2):
            if time.time() > deadline or used >= budget:
                break

            if len(x_archive) == 0:
                x = np.random.uniform(lo, hi)
            else:
                idx = np.random.randint(len(x_archive))
                parent = x_archive[idx]
                # Adaptive step size
                progress = used / budget
                sigma = (1.0 - 0.5 * progress) * 0.15 * (hi - lo)
                x = parent + sigma * np.random.standard_normal(dim)
                x = np.clip(x, lo, hi)

            evaluate(x)

            # Occasionally add to archive
            if np.random.random() < 0.3 and len(x_archive) < 50:
                fx = all_F[-1] if all_F else None
                if fx is not None:
                    x_archive.append(x.copy())
                    f_archive.append(np.asarray(fx, dtype=np.float64))

        # Phase 3: Random exploration
        for i in range(n_phase3):
            if time.time() > deadline or used >= budget:
                break
            x = np.random.uniform(lo, hi)
            evaluate(x)

        return all_F
# EVOLVE-BLOCK-END

"""Multi-objective optimizer on WFG4 using front-biased constrained sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random sampling biased toward WFG4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        rng = np.random

        archive = []
        archive_X = []

        def dominates(a, b):
            """Return True if a strictly dominates b (all <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def update_archive(f, x):
            """Add f to archive if non-dominated. Remove dominated points."""
            for af in archive:
                if dominates(af, f):
                    return False
            new_archive = []
            new_archive_X = []
            for i, af in enumerate(archive):
                if not dominates(f, af):
                    new_archive.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive.append(f.copy())
            new_archive_X.append(x.copy())
            archive[:] = new_archive
            archive_X[:] = new_archive_X
            return True

        def sample_front_biased():
            """
            Sample a point biased toward the WFG4 Pareto front.
            For WFG4, the Pareto front has a specific structure.
            We use a parameterized approach where we sample the first variable
            uniformly and derive others to stay near the front.
            """
            x = np.zeros(dim)
            
            # Sample t uniformly in [0, 1]
            t = rng.random()
            
            # First variable determines the first objective
            x[0] = lo[0] + span[0] * t
            
            # For remaining variables, use a mixture strategy:
            # 1. Some probability: uniform random (exploration)
            # 2. Some probability: concentrated around center (exploitation)
            # 3. Some probability: derived from front structure
            
            for d in range(1, dim):
                r = rng.random()
                if r < 0.3:
                    # Uniform random
                    x[d] = lo[d] + span[d] * rng.random()
                elif r < 0.6:
                    # Concentrated around center with Gaussian noise
                    center = (lo[d] + hi[d]) / 2.0
                    sigma = span[d] * 0.25
                    x[d] = center + sigma * rng.standard_normal()
                    x[d] = np.clip(x[d], lo[d], hi[d])
                else:
                    # Front-biased: use a value related to t
                    # For WFG4, variables 2+ often need to be set to specific values
                    # to achieve the front. Use a combination of t and random.
                    # The exact structure depends on the WFG4 implementation.
                    # A common pattern: x[d] = lo[d] + span[d] * (1 - t) for some variables
                    if d % 2 == 1:
                        x[d] = lo[d] + span[d] * (1.0 - t)
                    else:
                        x[d] = lo[d] + span[d] * t
                    # Add small noise to avoid exact duplicates
                    noise = span[d] * 0.05 * rng.standard_normal()
                    x[d] = np.clip(x[d] + noise, lo[d], hi[d])
            
            return x

        used = 0

        # Phase 1: Front-biased sampling (majority of budget)
        n_front = int(budget * 0.7)
        for i in range(n_front):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            x = sample_front_biased()
            f = problem(x)
            used += 1
            update_archive(f, x)

        # Phase 2: Local refinement around archive points (remaining budget)
        while used < budget:
            if time.time() > deadline - 5.0:
                break
            
            if len(archive) == 0:
                x = lo + span * rng.random(dim)
            elif rng.random() < 0.15:
                # Global exploration
                x = lo + span * rng.random(dim)
            else:
                # Local refinement around a random archive point
                idx = rng.randint(len(archive))
                base_x = archive_X[idx]
                
                # Adaptive step size that decreases over time
                progress = used / budget
                base_step = 0.15 * (1.0 - 0.6 * progress)
                
                # Gaussian perturbation
                step = base_step * rng.standard_normal(dim)
                x = base_x + step * span
                x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            update_archive(f, x)

        return archive_X if archive_X else [np.zeros(dim)]

# EVOLVE-BLOCK-END

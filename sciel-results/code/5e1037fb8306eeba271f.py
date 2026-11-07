"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured multi-objective optimizer for WFG9 using Pareto-guided sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        radii = (hi - lo) / 2.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # WFG9 specific: Pareto front in decision space has structure
        # For WFG9 with 12 vars, the Pareto front is characterized by:
        # - First few variables control objective trade-offs
        # - Remaining variables should be at specific values
        # We'll use a hybrid: structured sampling + local refinement

        pareto_archive = []  # List of (x, f) pairs that are non-dominated
        all_evaluated = []   # All evaluated objective vectors

        def is_dominated(f, archive):
            """Check if f is dominated by any point in archive."""
            for af in archive:
                if all(af[i] <= f[i] for i in range(self.n_obj)) and any(af[i] < f[i] for i in range(self.n_obj)):
                    return True
            return False

        def update_archive(x, f):
            """Add to Pareto archive if non-dominated, remove dominated points."""
            if is_dominated(f, [af for _, af in pareto_archive]):
                return False
            # Remove dominated points from archive
            new_archive = []
            for ax, af in pareto_archive:
                if not (all(f[i] <= af[i] for i in range(self.n_obj)) and any(f[i] < af[i] for i in range(self.n_obj))):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            pareto_archive[:] = new_archive
            return True

        # Phase 1: Structured global sampling (40% of budget)
        n_phase1 = int(budget * 0.4)
        # Sample in a grid-like pattern focused on the region where Pareto front likely is
        # For WFG9, the front is in the region where variables are near specific values
        for i in range(n_phase1):
            if time.time() > deadline:
                break

            # Create structured samples: mix of random and quasi-random
            if i % 3 == 0:
                # Pure random in full space
                x = rng.uniform(lo, hi)
            elif i % 3 == 1:
                # Sample in a smaller region around center (where front likely is)
                shrink = 0.3 + 0.5 * (1 - i / n_phase1)  # shrinks from 0.8 to 0.3
                x = center + radii * rng.uniform(-shrink, shrink)
                x = np.clip(x, lo, hi)
            else:
                # Quasi-random (Halton-like) for better coverage
                # Simple LCG-based quasi-random
                t = (i * 0.618033988749895) % 1.0  # golden ratio conjugate
                x = np.array([lo[j] + (hi[j] - lo[j]) * ((t * (j + 1) + 0.5) % 1.0) for j in range(dim)])

            f = np.array(problem(x))
            all_evaluated.append(f)
            update_archive(x, f)

        # Phase 2: Local refinement around Pareto archive (60% of budget)
        n_phase2 = budget - len(all_evaluated)
        for i in range(n_phase2):
            if time.time() > deadline:
                break

            if not pareto_archive:
                # Fallback to random
                x = rng.uniform(lo, hi)
            else:
                # Pick a random archive point and perturb it
                idx = rng.integers(0, len(pareto_archive))
                base_x = pareto_archive[idx][0]
                # Adaptive perturbation: larger early, smaller later
                frac = i / max(n_phase2 - 1, 1)
                sigma = 0.15 * (1.0 - 0.7 * frac)  # shrinks from 0.15 to 0.045
                x = base_x + rng.standard_normal(dim) * sigma * radii
                x = np.clip(x, lo, hi)

            f = np.array(problem(x))
            all_evaluated.append(f)
            update_archive(x, f)

        if not all_evaluated:
            return np.array([])

        return np.array(all_evaluated)
# EVOLVE-BLOCK-END

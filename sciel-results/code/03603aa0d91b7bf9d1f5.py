"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based decomposition with LHS for WFG4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        def lhs_sample(n_samples, dim, lo, hi, rng):
            """Generate Latin Hypercube Samples."""
            indices = np.array([rng.permutation(n_samples) for _ in range(dim)]).T
            u = (indices + rng.random((n_samples, dim))) / n_samples
            return lo + (hi - lo) * u

        # Phase 1: LHS for broad coverage
        n_lhs = 40
        X_lhs = lhs_sample(n_lhs, self.dim, lo, hi, rng)
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs

        # Phase 2: Structured refinement around promising regions
        # Keep track of non-dominated points
        def is_non_dominated(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        # Phase 3: Directed search toward the Pareto front
        # WFG4 has a known structure; explore systematically
        remaining = self.budget - used

        # Use a mix of:
        # 1. Local refinement around best points
        # 2. Systematic sweep along key dimensions
        # 3. Random exploration in uncovered regions

        best_archive = []
        for i in range(n_lhs):
            best_archive.append((X_lhs[i].copy(), F_lhs[i].copy()))

        # Identify non-dominated points for refinement
        nd_mask = is_non_dominated(F_lhs)
        nd_points = X_lhs[nd_mask]
        nd_f = F_lhs[nd_mask]

        # Phase 4: Iterative refinement with time check
        n_refine = min(remaining, 120)
        for k in range(n_refine):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break

            # Choose a parent: prefer non-dominated, but also explore
            if k % 3 == 0:
                # Explore: random point in a sub-region
                parent_idx = rng.integers(self.dim)
                parent = X_lhs[rng.integers(len(X_lhs))]
            else:
                # Exploit: refine around non-dominated points
                parent = nd_points[rng.integers(len(nd_points))] if len(nd_points) > 0 else X_lhs[rng.integers(len(X_lhs))]

            # Adaptive mutation: larger steps early, smaller later
            progress = k / max(n_refine, 1)
            step_size = 0.3 * (1.0 - 0.7 * progress)
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Add to archive if non-dominated
            if len(best_archive) < 50:
                best_archive.append((child.copy(), fc.copy()))
            else:
                # Check if child dominates any in archive
                dominated_any = False
                for idx, (_, f_old) in enumerate(best_archive):
                    if np.all(fc <= f_old) and np.any(fc < f_old):
                        best_archive[idx] = (child.copy(), fc.copy())
                        dominated_any = True
                        break
                if not dominated_any:
                    # Check if child is non-dominated by archive
                    is_dom = False
                    for _, f_old in best_archive:
                        if np.all(f_old <= fc) and np.any(f_old < fc):
                            is_dom = True
                            break
                    if not is_dom:
                        best_archive.append((child.copy(), fc.copy()))

        # Phase 5: Final systematic sweep if time allows
        if time.time() < deadline - 5.0 and used < self.budget:
            # Sweep along each dimension from a good starting point
            start_point = nd_points[0] if len(nd_points) > 0 else X_lhs[0]
            for d in range(self.dim):
                if used >= self.budget or time.time() > deadline:
                    break
                for t in [0.1, 0.5, 0.9]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    candidate = start_point.copy()
                    candidate[d] = lo[d] + t * (hi[d] - lo[d])
                    fc = problem(candidate)
                    used += 1

        return None
# EVOLVE-BLOCK-END

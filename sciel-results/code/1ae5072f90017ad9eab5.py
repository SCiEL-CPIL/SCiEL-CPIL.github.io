"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Simulated Annealing optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Initialize archive with random points
        archive_X = []
        archive_F = []
        evaluations = 0

        def dominates(f1, f2):
            """Check if f1 dominates f2 (both minimize)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive."""
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            non_local = True
            for i, af in enumerate(archive_F):
                if dominates(f, af):
                    non_local = False
                    archive_X.pop(i)
                    archive_F.pop(i)
                    break
                elif dominates(af, f):
                    return
            if non_local:
                archive_X.append(x)
                archive_F.append(f)

        def hypervolume_contribution(f, archive_F, ref):
            """Approximate hypervolume contribution of a point."""
            # Simple approximation: check if point is on the Pareto front
            # and estimate its contribution
            n = len(archive_F)
            if n == 0:
                return 0.0
            # For 3 objectives, contribution is roughly the volume of the box
            # that this point uniquely covers
            # This is a rough heuristic
            vol = 0.0
            for d in range(self.n_obj):
                if f[d] >= ref[d]:
                    return 0.0
            # Estimate based on position in sorted order
            sorted_idx = np.argsort([af[d] for af in archive_F])
            pos = list(sorted_idx).index(archive_F.index(f)) if f in archive_F else 0
            # Simplified: just use a small constant based on archive size
            return 1.0 / max(n, 1)

        def select_archive_point(archive_F, ref):
            """Select a point from archive to perturb, biased toward hypervolume contributors."""
            n = len(archive_F)
            if n == 0:
                return None
            # Prefer points that are not dominated and have good spread
            # Use weighted selection based on distance to reference
            weights = []
            for f in archive_F:
                if np.all(f < ref):
                    # Points closer to origin in objective space are more valuable
                    dist = np.sum(ref - f)
                    weights.append(dist)
                else:
                    weights.append(0.0)
            total_w = sum(weights)
            if total_w <= 0:
                return rng.integers(0, n)
            weights = np.array(weights) / total_w
            idx = rng.choice(n, p=weights)
            return idx

        # Phase 1: Initialize with random points
        init_size = 30
        for i in range(init_size):
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            add_to_archive(x, f)
            evaluations += 1
            if time.time() > deadline:
                break

        # Phase 2: MOSA main loop
        T0 = 1.0
        T_min = 0.001
        alpha = 0.95  # Cooling rate

        T = T0
        step_scale = 0.1  # Initial step size scale

        while evaluations < self.budget and T > T_min:
            if time.time() > deadline:
                break

            # Select a point from archive to perturb
            if not archive_X:
                # If archive is empty, generate random point
                x = lo + (hi - lo) * rng.random(self.dim)
            else:
                idx = select_archive_point(archive_F, np.array([3.0, 5.0, 7.0]))
                x = archive_X[idx].copy()

            # Generate perturbation
            # Use Gaussian-like perturbation scaled by temperature
            noise = rng.standard_normal(self.dim) * step_scale * np.sqrt(T)
            x_new = x + noise
            x_new = np.clip(x_new, lo, hi)

            # Evaluate
            f_new = problem(x_new)
            evaluations += 1

            # Check if new point is non-dominated
            if not is_dominated(f_new, archive_F):
                # Add to archive (removing dominated points)
                add_to_archive(x_new, f_new)
                # Accept and cool
                T *= alpha
            else:
                # Point is dominated, consider accepting based on temperature
                # Compute how much worse it is
                # Find the best dominator
                best_dom_diff = np.inf
                for af in archive_F:
                    if dominates(af, f_new):
                        diff = np.sum(f_new - af)
                        best_dom_diff = min(best_dom_diff, diff)
                
                # Accept with probability exp(-delta/T)
                if best_dom_diff < np.inf:
                    prob = np.exp(-best_dom_diff / max(T, 1e-10))
                    if rng.random() < prob:
                        # Accept inferior move to escape local optima
                        add_to_archive(x_new, f_new)
                        T *= alpha
                else:
                    T *= alpha

            # Occasionally add a random point to maintain diversity
            if evaluations % 20 == 0 and evaluations < self.budget:
                x_rand = lo + (hi - lo) * rng.random(self.dim)
                f_rand = problem(x_rand)
                if not is_dominated(f_rand, archive_F):
                    add_to_archive(x_rand, f_rand)
                evaluations += 1

            # Adapt step size
            step_scale *= 0.99  # Gradually reduce step size

        # Return the archive points
        return np.array(archive_X) if archive_X else np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume-guided local search for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])

        rng = np.random.default_rng(123)

        # Use Sobol quasi-random for initial spread
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42)

        # Initial population: Sobol points
        n_init = 40
        X_init = sobol.random(n_init)
        X_init = lo + (hi - lo) * X_init

        archive = []  # list of (x, f)
        used = 0

        def evaluate(x):
            nonlocal used
            fx = problem(x)
            used += 1
            return fx

        # Evaluate initial population
        for x in X_init:
            fx = evaluate(x)
            archive.append((x.copy(), fx))

        def is_dominated(f_new, archive_f_list):
            """Check if f_new is dominated by any point in archive."""
            for af in archive_f_list:
                if np.all(af <= f_new) and np.any(af < f_new):
                    return True
            return False

        def dominates(f_new, af):
            return np.all(f_new <= af) and np.any(f_new < af)

        def hv_contribution_2d(f_new, archive_f_list, ref_pt):
            """Compute hypervolume contribution of f_new given existing archive points (2D)."""
            # Only count if f_new is better than ref in both objectives
            if f_new[0] >= ref_pt[0] or f_new[1] >= ref_pt[1]:
                return 0.0

            # Collect all points including f_new that are better than ref
            pts = [(f_new[0], f_new[1])]
            for af in archive_f_list:
                if af[0] < ref_pt[0] and af[1] < ref_pt[1]:
                    pts.append((af[0], af[1]))

            if len(pts) <= 1:
                # Single point: area = (ref[0]-f[0]) * (ref[1]-f[1])
                return (ref_pt[0] - f_new[0]) * (ref_pt[1] - f_new[1])

            # Compute HV of pts, then HV of pts without f_new
            def hv_2d(points, ref_p):
                """Exact 2D hypervolume."""
                if not points:
                    return 0.0
                # Sort by first objective
                pts_sorted = sorted(points, key=lambda p: p[0])
                total = 0.0
                prev_y = ref_p[1]
                for px, py in pts_sorted:
                    if py >= ref_p[1]:
                        continue
                    if px >= ref_p[0]:
                        break
                    total += (ref_p[0] - px) * (prev_y - py)
                    if py < prev_y:
                        prev_y = py
                return total

            hv_with = hv_2d(pts, ref_pt)
            pts_without = [p for p in pts if p != (f_new[0], f_new[1])]
            hv_without = hv_2d(pts_without, ref_pt)
            return hv_with - hv_without

        def add_to_archive(x, fx):
            """Add point to archive if non-dominated, remove dominated points."""
            if fx[0] >= ref[0] or fx[1] >= ref[1]:
                return False
            # Check if dominated by existing
            for _, af in archive:
                if np.all(af <= fx) and np.any(af < fx):
                    return False
            # Remove points dominated by new point
            new_archive = []
            for px, pf in archive:
                if not (np.all(fx <= pf) and np.any(fx < pf)):
                    new_archive.append((px, pf))
            new_archive.append((x.copy(), fx))
            archive.clear()
            archive.extend(new_archive)
            return True

        # Main search loop
        # Strategy: alternate between:
        # 1. Sobol quasi-random exploration
        # 2. Local perturbation around high-HV-contribution archive points
        # 3. Directed search toward promising regions

        sobol_counter = n_init
        sobol_batch = 10

        # Track best HV contribution for guidance
        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute HV contributions for archive points to guide search
            if len(archive) > 0:
                archive_f = np.array([af for _, af in archive])
                contributions = []
                for i, (ax, af) in enumerate(archive):
                    # Approximate contribution: distance to ref weighted by density
                    if af[0] < ref[0] and af[1] < ref[1]:
                        # Simple proxy: area of rectangle from af to ref
                        contrib = (ref[0] - af[0]) * (ref[1] - af[1])
                    else:
                        contrib = 0.0
                    contributions.append(contrib)
                contributions = np.array(contributions)
            else:
                contributions = np.array([])

            # Choose strategy based on progress
            progress = used / self.budget

            if progress < 0.3:
                # Early: more exploration with Sobol
                n_explore = min(sobol_batch, self.budget - used)
                for _ in range(n_explore):
                    if used >= self.budget or time.time() > deadline:
                        break
                    if sobol_counter < 200:
                        x_new = sobol.random()
                        sobol_counter += 1
                    else:
                        x_new = rng.random(self.dim)
                    x_new = lo + (hi - lo) * x_new
                    fx = evaluate(x_new)
                    add_to_archive(x_new, fx)
            elif progress < 0.7:
                # Mid: mix of local search around good points and exploration
                n_local = min(15, self.budget - used)
                n_explore = min(5, self.budget - used - n_local)

                # Local search around top-contributing archive points
                if len(archive) > 0 and len(contributions) > 0:
                    top_k = min(5, len(archive))
                    top_idx = np.argsort(-contributions)[:top_k]
                    for _ in range(n_local):
                        if used >= self.budget or time.time() > deadline:
                            break
                        idx = top_idx[rng.integers(len(top_idx))]
                        base_x = archive[idx][0].copy()
                        # Perturb with decreasing step size
                        scale = 0.1 * (1.0 - progress * 0.5)
                        x_new = base_x + rng.normal(0, scale, self.dim)
                        x_new = np.clip(x_new, lo, hi)
                        fx = evaluate(x_new)
                        add_to_archive(x_new, fx)

                # Some exploration
                for _ in range(n_explore):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_new = lo + (hi - lo) * rng.random(self.dim)
                    fx = evaluate(x_new)
                    add_to_archive(x_new, fx)
            else:
                # Late: intensive local search around best points
                n_local = min(20, self.budget - used)
                if len(archive) > 0 and len(contributions) > 0:
                    top_k = min(10, len(archive))
                    top_idx = np.argsort(-contributions)[:top_k]
                    for _ in range(n_local):
                        if used >= self.budget or time.time() > deadline:
                            break
                        idx = top_idx[rng.integers(len(top_idx))]
                        base_x = archive[idx][0].copy()
                        # Very small perturbations
                        scale = 0.02 * (1.0 - progress * 0.5)
                        x_new = base_x + rng.normal(0, scale, self.dim)
                        x_new = np.clip(x_new, lo, hi)
                        fx = evaluate(x_new)
                        add_to_archive(x_new, fx)

                # Also try directed moves toward ideal point
                if used < self.budget:
                    ideal = np.array([min(af[0] for _, af in archive),
                                      min(af[1] for _, af in archive)])
                    for _ in range(5):
                        if used >= self.budget or time.time() > deadline:
                            break
                        idx = rng.integers(len(archive))
                        base_x = archive[idx][0].copy()
                        # Move toward ideal
                        direction = ideal - base_x
                        step = 0.1 * (1.0 - progress * 0.5)
                        x_new = base_x + step * direction
                        x_new = np.clip(x_new, lo, hi)
                        fx = evaluate(x_new)
                        add_to_archive(x_new, fx)

        return archive

# EVOLVE-BLOCK-END

"""Decomposition-based multi-objective optimizer for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based MOO: reference directions + local search per subproblem."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _reference_directions(self, n_obj, n_dirs):
        """Generate n_dirs reference directions in the simplex (sum=1, all >=0)."""
        dirs = []
        if n_obj == 3:
            # Systematic sampling on the 2-simplex
            # Use a grid approach: for i+j+k = n_sub
            n_sub = int(round((2 * n_dirs) ** 0.5))
            while n_sub < 1:
                n_sub = 1
            for i in range(n_sub + 1):
                for j in range(n_sub + 1 - i):
                    k = n_sub - i - j
                    if i + j + k == n_sub:
                        dirs.append(np.array([i, j, k], dtype=float) / n_sub)
            dirs = np.array(dirs)
            # If we have too many, subsample; if too few, add random
            if len(dirs) > n_dirs:
                idx = np.linspace(0, len(dirs) - 1, n_dirs).astype(int)
                dirs = dirs[idx]
            elif len(dirs) < n_dirs:
                extra = np.random.dirichlet(np.ones(n_obj), size=n_dirs - len(dirs))
                dirs = np.vstack([dirs, extra])
        else:
            dirs = np.random.dirichlet(np.ones(n_obj), size=n_dirs)
        return dirs

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0

        # Choose number of reference directions
        n_dirs = min(24, max(12, self.budget // 10))

        # Generate reference directions
        ref_dirs = self._reference_directions(self.n_obj, n_dirs)

        # Track all evaluated points for hypervolume
        all_points = []

        # Phase 1: Random initialization
        n_init = min(n_dirs * 2, self.budget // 4)
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        all_points.extend([f for f in F_init])
        used = n_init

        # Assign each direction to the closest point in the initial population
        # (using weighted sum as proxy)
        best_per_dir = np.zeros((n_dirs, self.dim))
        best_f_per_dir = np.zeros((n_dirs, self.n_obj))
        for i in range(n_dirs):
            weights = ref_dirs[i]
            scores = F_init @ weights
            idx = np.argmin(scores)
            best_per_dir[i] = X_init[idx]
            best_f_per_dir[i] = F_init[idx]

        # Phase 2: Local search refinement per direction
        # Use coordinate descent / Nelder-Mead style local search
        # Budget per direction: (budget - n_init) / n_dirs
        remaining = self.budget - n_init
        budget_per_dir = max(1, remaining // n_dirs)

        for i in range(n_dirs):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            x_best = best_per_dir[i].copy()
            f_best = best_f_per_dir[i].copy()
            weights = ref_dirs[i]

            # Local search: coordinate descent with adaptive step
            step_sizes = 0.05 * (hi - lo)
            dim = self.dim

            for step in range(budget_per_dir):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Try perturbing each coordinate
                improved = False
                for d in range(dim):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    # Try positive and negative perturbations
                    for sign in [1, -1]:
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break

                        x_trial = x_best.copy()
                        x_trial[d] += sign * step_sizes[d]
                        x_trial = np.clip(x_trial, lo, hi)

                        f_trial = np.array(problem(x_trial))
                        used += 1
                        all_points.append(f_trial)

                        # Check if this improves the weighted sum for this direction
                        ws_best = np.dot(f_best, weights)
                        ws_trial = np.dot(f_trial, weights)

                        if ws_trial < ws_best:
                            x_best = x_trial
                            f_best = f_trial
                            improved = True
                            # Shrink step for this coordinate
                            step_sizes[d] *= 0.7
                        else:
                            # Expand step if not improved (adaptive)
                            step_sizes[d] = min(step_sizes[d] * 1.2, 0.1 * (hi - lo))

                # If no improvement in a full pass, reduce all steps
                if not improved:
                    step_sizes *= 0.8
                    if np.all(step_sizes < 1e-6 * (hi - lo)):
                        break

            best_per_dir[i] = x_best
            best_f_per_dir[i] = f_best

        return best_per_dir
# EVOLVE-BLOCK-END

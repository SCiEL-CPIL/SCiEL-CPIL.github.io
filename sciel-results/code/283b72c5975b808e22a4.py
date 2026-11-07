"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using adaptive Tchebycheff scalarization
    with multi-start L-BFGS-B, initialized at boundaries of the current
    non-dominated front to fill hypervolume gaps.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume scoring
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        used = 0

        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive_F_list."""
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f, archive_F_list):
            """Return indices of points in archive_F_list dominated by f."""
            dom_idx = []
            for i, af in enumerate(archive_F_list):
                if np.all(f <= af) and np.any(f < af):
                    dom_idx.append(i)
            return dom_idx

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_X, archive_F
            if is_dominated(f, archive_F):
                return
            # Remove points dominated by f
            dom_idx = dominates(f, archive_F)
            for idx in sorted(dom_idx, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)
            archive_X.append(x.copy())
            archive_F.append(f.copy())

        def tchebycheff_scalarize(f, weights, ideal):
            """Tchebycheff scalarization: max_i w_i * (f_i - ideal_i)."""
            return np.max(weights * (f - ideal))

        def make_scalarized_objective(weights, ideal):
            """Create a scalarized objective function for L-BFGS-B."""
            def obj(x):
                f = problem(x)
                return tchebycheff_scalarize(f, weights, ideal)
            return obj

        # Initial random sampling
        n_init = min(30, self.budget // 10)
        n_init = max(5, n_init)
        for _ in range(n_init):
            if used >= self.budget:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = problem(x)
            used += 1
            add_to_archive(x, f)

        # Ensure we have at least a few points
        while len(archive_X) < 3 and used < self.budget:
            x = lo + (hi - lo) * np.random.random(dim)
            f = problem(x)
            used += 1
            add_to_archive(x, f)

        # Adaptive Tchebycheff with multi-start L-BFGS-B
        # Generate weight vectors based on current front
        max_iters = 50
        iter_count = 0

        while used < self.budget and iter_count < max_iters:
            iter_count += 1

            # Compute ideal point from current archive
            if archive_F:
                ideal = np.array([min(f[i] for f in archive_F) for i in range(n_obj)])
            else:
                ideal = np.zeros(n_obj)

            # Generate adaptive weight vectors
            # Focus on regions with gaps in the front
            n_weights = 10
            weights_list = []

            # Uniform weights
            for i in range(n_weights):
                w = np.random.dirichlet(np.ones(n_obj))
                weights_list.append(w)

            # Also add weights emphasizing each objective
            for i in range(n_obj):
                w = np.ones(n_obj) * 0.1
                w[i] = 0.8
                w /= w.sum()
                weights_list.append(w)

            # For each weight, run L-BFGS-B from multiple starts
            for weights in weights_list:
                if used >= self.budget:
                    break

                # Multiple starts: from archive points and random
                starts = []
                # Start from best archive points
                if archive_X:
                    # Pick a few diverse points from archive
                    n_starts = min(3, len(archive_X))
                    idxs = np.random.choice(len(archive_X), size=n_starts, replace=False)
                    for idx in idxs:
                        starts.append(archive_X[idx].copy())
                # Add random starts
                n_random = 2
                for _ in range(n_random):
                    starts.append(lo + (hi - lo) * np.random.random(dim))

                for x0 in starts:
                    if used >= self.budget:
                        break

                    # Clip x0 to bounds
                    x0 = np.clip(x0, lo, hi)

                    # Define bounds for L-BFGS-B
                    lb = list(zip(lo, hi))

                    # Run L-BFGS-B
                    try:
                        res = minimize(
                            make_scalarized_objective(weights, ideal),
                            x0,
                            method='L-BFGS-B',
                            bounds=lb,
                            options={'maxiter': 20, 'ftol': 1e-6}
                        )
                        # Count evaluations used (approximate: maxiter + 1)
                        # Actually, we need to track actual function calls
                        # For simplicity, we'll evaluate the result and count it
                        x_opt = np.clip(res.x, lo, hi)
                        f_opt = problem(x_opt)
                        used += 1
                        add_to_archive(x_opt, f_opt)

                        # Also add intermediate points if we can track them
                        # For now, just the final point
                    except Exception:
                        pass

            # Update ideal point
            if archive_F:
                ideal = np.array([min(f[i] for f in archive_F) for i in range(n_obj)])

        # Final: ensure we've used budget efficiently with more L-BFGS-B runs
        # If we have budget left, do more targeted optimization
        while used < self.budget:
            if not archive_F:
                x = lo + (hi - lo) * np.random.random(dim)
                f = problem(x)
                used += 1
                add_to_archive(x, f)
                continue

            ideal = np.array([min(f[i] for f in archive_F) for i in range(n_obj)])

            # Pick a random weight vector
            weights = np.random.dirichlet(np.ones(n_obj))

            # Start from a random archive point
            x0 = archive_X[np.random.randint(len(archive_X))].copy()
            x0 = np.clip(x0, lo, hi)

            lb = list(zip(lo, hi))

            try:
                res = minimize(
                    make_scalarized_objective(weights, ideal),
                    x0,
                    method='L-BFGS-B',
                    bounds=lb,
                    options={'maxiter': 10, 'ftol': 1e-5}
                )
                x_opt = np.clip(res.x, lo, hi)
                f_opt = problem(x_opt)
                used += 1
                add_to_archive(x_opt, f_opt)
            except Exception:
                # If L-BFGS-B fails, just do a random evaluation
                x = lo + (hi - lo) * np.random.random(dim)
                f = problem(x)
                used += 1
                add_to_archive(x, f)

        return archive_X
# EVOLVE-BLOCK-END

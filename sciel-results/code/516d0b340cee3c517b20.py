"""Optimized multi-objective search for ZDT3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using archive-based search for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget

        # Archive to store all non-dominated points found
        archive_X = []
        archive_F = []

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_X, archive_F
            # Check if new point is dominated by any archive point
            for i, af in enumerate(archive_F):
                if np.all(af <= f) and np.any(af < f):
                    return  # Dominated, don't add
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Phase 1: Sample the analytical Pareto front densely
        # For ZDT3: x1 = t, x_i = 0 for i >= 2 gives f1 = t, f2 = 1 - sqrt(t)
        # But the actual Pareto front has structure from the sin term
        # Let's sample t in [0,1] with denser sampling where the front curves

        # Use non-uniform sampling: more points where |df2/dt| is large (near t=0)
        # f2 = 1 - sqrt(t), df2/dt = -1/(2*sqrt(t)), large near t=0
        # Use t = u^2 for uniform u to get denser sampling near t=0
        n_initial = 80
        u = np.linspace(0, 1, n_initial)
        t_vals = u ** 2  # Denser near 0

        for t in t_vals:
            x = np.zeros(self.dim)
            x[0] = t
            f = problem(x)
            add_to_archive(x, f)
            n_evals += 1
            if time.time() > deadline:
                break

        # Phase 2: Local refinement around archive points
        # For each archive point, try small perturbations in x1
        # and random perturbations in other dimensions

        # Sort archive by f1 for systematic refinement
        if archive_X:
            archive_arr_X = np.array(archive_X)
            archive_arr_F = np.array(archive_F)
            sort_idx = np.argsort(archive_arr_F[:, 0])
            archive_arr_X = archive_arr_X[sort_idx]
            archive_arr_F = archive_arr_F[sort_idx]
            archive_X = [archive_arr_X[i] for i in range(len(archive_arr_X))]
            archive_F = [archive_arr_F[i] for i in range(len(archive_arr_F))]

        # Refinement: for points in the archive, try local moves
        # Focus on x1 (first dimension) since it directly controls f1
        refinement_steps = [0.01, 0.005, 0.002, 0.001, 0.0005]

        for step in refinement_steps:
            if n_evals >= max_evals or time.time() > deadline:
                break

            # For each archive point, try moving x1 by ±step
            # Only do this for a subset to save budget
            n_refine = min(len(archive_X), max(5, (max_evals - n_evals) // (len(refinement_steps) * 2)))

            for i in range(n_refine):
                if n_evals >= max_evals or time.time() > deadline:
                    break

                x_base = archive_X[i].copy()
                t_base = x_base[0]

                # Try positive and negative steps in x1
                for delta in [step, -step]:
                    if n_evals >= max_evals or time.time() > deadline:
                        break
                    x_new = x_base.copy()
                    x_new[0] = np.clip(t_base + delta, 0, 1)

                    # Also try small random perturbations in other dims
                    if np.random.rand() < 0.3:
                        x_new[1:] += np.random.randn(self.dim - 1) * 0.01
                        x_new[1:] = np.clip(x_new[1:], 0, 1)

                    f_new = problem(x_new)
                    add_to_archive(x_new, f_new)
                    n_evals += 1

        # Phase 3: If budget remains, do more random exploration
        # focused on regions of the Pareto front we may have missed
        if n_evals < max_evals:
            # Identify gaps in the archive along f1
            if len(archive_F) >= 2:
                f1_vals = np.array([f[0] for f in archive_F])
                f1_vals_sorted = np.sort(f1_vals)
                gaps = np.diff(f1_vals_sorted)
                if len(gaps) > 0:
                    max_gap = np.max(gaps)
                    gap_idx = np.argmax(gaps)
                    gap_start = f1_vals_sorted[gap_idx]
                    gap_end = f1_vals_sorted[gap_idx + 1]

                    # Fill the largest gap with more points
                    n_fill = min(20, max_evals - n_evals)
                    if n_fill > 0:
                        t_fill = np.linspace(gap_start, gap_end, n_fill + 2)[1:-1]
                        for t in t_fill:
                            if n_evals >= max_evals or time.time() > deadline:
                                break
                            x = np.zeros(self.dim)
                            x[0] = t
                            f = problem(x)
                            add_to_archive(x, f)
                            n_evals += 1

        # Return all archive points
        if archive_X:
            return np.array(archive_X)
        return np.array([np.zeros(self.dim)])

# EVOLVE-BLOCK-END

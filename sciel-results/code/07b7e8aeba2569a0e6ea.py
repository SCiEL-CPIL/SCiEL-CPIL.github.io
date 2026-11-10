"""Multi-objective LHS + Nelder-Mead optimizer for WFG7 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS sampling and Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (a is better or equal in all, strictly better in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _nelder_mead(self, problem, x0, lo, hi, max_evals, deadline):
        """
        Run Nelder-Mead on a scalarized objective (sum of objectives).
        Returns (best_x, best_f, n_evals_used).
        """
        dim = self.dim
        n_obj = self.n_obj

        # Simplex: x0 and dim perturbed points
        simplex = [x0.copy()]
        f_vals = [np.sum(problem(x0))]
        n_evals = 1

        # Perturb each dimension
        for i in range(dim):
            xi = x0.copy()
            # Small perturbation: 5% of range
            span = hi[i] - lo[i]
            step = 0.05 * span if span > 0 else 0.1
            xi[i] += step
            xi = np.clip(xi, lo, hi)
            simplex.append(xi)
            f_vals.append(np.sum(problem(xi)))
            n_evals += 1

        # Nelder-Mead iterations
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        while n_evals < max_evals and time.time() < deadline:
            # Sort simplex by objective value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, lo, hi)
            f_r = np.sum(problem(x_r))
            n_evals += 1

            if f_r < f_vals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, lo, hi)
                f_e = np.sum(problem(x_e))
                n_evals += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                if f_r < f_vals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = np.sum(problem(x_c))
                    n_evals += 1
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_vals[i] = np.sum(problem(simplex[i]))
                            n_evals += 1
                        break
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        f_vals[i] = np.sum(problem(simplex[i]))
                        n_evals += 1
                    break

        # Find best
        best_idx = int(np.argmin(f_vals))
        return simplex[best_idx], f_vals[best_idx], n_evals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Budget allocation:
        # - LHS sampling: 60 points
        # - Nelder-Mead refinement on top non-dominated points: remaining budget
        lhs_size = 60
        nm_per_point = 15  # max evaluations per Nelder-Mead run
        max_nm_points = 5  # how many points to refine

        used = 0

        # Latin Hypercube Sampling
        # For each dimension, divide [0,1] into lhs_size equal intervals,
        # pick one random point in each, then permute per dimension.
        rng = np.random.default_rng(42)
        lhs = np.zeros((lhs_size, self.dim))
        for d in range(self.dim):
            # Generate lhs_size random points in [0,1]
            pts = rng.random(lhs_size)
            # Divide into lhs_size bins, pick one from each
            bins = np.linspace(0, 1, lhs_size + 1)
            for i in range(lhs_size):
                # Pick random point in bin i
                u = rng.random()
                lhs[i, d] = bins[i] + u * (bins[i + 1] - bins[i])

        # Map to actual bounds
        X_lhs = lo + span * lhs
        F_lhs = np.array([problem(x) for x in X_lhs])
        used += lhs_size

        # Find non-dominated points from LHS
        archive_X = []
        archive_F = []
        for i in range(lhs_size):
            x, f = X_lhs[i], F_lhs[i]
            # Check if dominated by any existing archive point
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if not dominated:
                # Remove points dominated by this new point
                new_X = []
                new_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_X.append(ax)
                        new_F.append(af)
                new_X.append(x.copy())
                new_F.append(f.copy())
                archive_X = new_X
                archive_F = new_F

        # Select top points for Nelder-Mead refinement
        # Sort by sum of objectives (lower is better)
        if archive_F:
            F_arr = np.array(archive_F)
            sums = np.sum(F_arr, axis=1)
            order = np.argsort(sums)
            # Take top max_nm_points
            refine_indices = order[:max_nm_points]
        else:
            refine_indices = []

        # Run Nelder-Mead on each selected point
        for idx in refine_indices:
            if used >= self.budget or time.time() > deadline:
                break

            x0 = archive_X[idx].copy()
            # Remaining budget for this Nelder-Mead run
            remaining = self.budget - used
            max_evals = min(nm_per_point, remaining)
            if max_evals < 3:
                break

            x_best, f_best, n_used = self._nelder_mead(problem, x0, lo, hi, max_evals, deadline)
            used += n_used

            # Add to archive if non-dominated
            f_best_vec = np.array(problem(x_best)) if False else None  # already computed in NM
            # Actually, we need the vector of objectives, not just the sum
            # Let's re-evaluate to get the vector
            f_vec = problem(x_best)
            used += 1  # this extra evaluation

            dominated = False
            for af in archive_F:
                if np.all(af <= f_vec) and np.any(af < f_vec):
                    dominated = True
                    break
            if not dominated:
                new_X = []
                new_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f_vec <= af) and np.any(f_vec < af)):
                        new_X.append(ax)
                        new_F.append(af)
                new_X.append(x_best.copy())
                new_F.append(f_vec.copy())
                archive_X = new_X
                archive_F = new_F

        # If we still have budget, do more LHS + NM cycles
        # But for simplicity, let's just use the remaining budget for additional random + NM
        while used < self.budget and time.time() < deadline:
            # Generate a random point
            x = lo + span * rng.random(self.dim)
            f = problem(x)
            used += 1

            # Check if non-dominated
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if not dominated:
                new_X = []
                new_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_X.append(ax)
                        new_F.append(af)
                new_X.append(x.copy())
                new_F.append(f.copy())
                archive_X = new_X
                archive_F = new_F

                # If we have budget, do a quick NM refinement
                if used + nm_per_point <= self.budget and time.time() < deadline:
                    x_best, f_best, n_used = self._nelder_mead(problem, x, lo, hi, nm_per_point, deadline)
                    used += n_used
                    f_vec = problem(x_best)
                    used += 1

                    dominated2 = False
                    for af in archive_F:
                        if np.all(af <= f_vec) and np.any(af < f_vec):
                            dominated2 = True
                            break
                    if not dominated2:
                        new_X = []
                        new_F = []
                        for ax, af in zip(archive_X, archive_F):
                            if not (np.all(f_vec <= af) and np.any(f_vec < af)):
                                new_X.append(ax)
                                new_F.append(af)
                        new_X.append(x_best.copy())
                        new_F.append(f_vec.copy())
                        archive_X = new_X
                        archive_F = new_F

        return archive_X
# EVOLVE-BLOCK-END

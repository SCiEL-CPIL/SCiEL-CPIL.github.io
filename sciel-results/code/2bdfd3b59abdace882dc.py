# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Deterministic space-filling initial design (Halton sequence).
      2. Differential Evolution (DE) for broad exploration.
      3. Surrogate-assisted local refinement using RBF interpolation and
         gradient-based (L-BFGS-B) local search on weighted-sum scalarizations.
      4. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim, skip=0):
        """Generate n points in the [0,1]^dim unit cube using the Halton sequence."""
        points = np.zeros((n, dim))
        for i in range(skip, skip + n):
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, ...
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                points[i - skip, d] = r
        return points

    def _is_dominated(self, f, archive_f):
        """Check if f is dominated by any point in archive_f (array of shape (m, n_obj))."""
        if len(archive_f) == 0:
            return False
        all_leq = np.all(archive_f <= f, axis=1)
        any_lt = np.any(archive_f < f, axis=1)
        return np.any(all_leq & any_lt)

    def _add_to_archive(self, x, f, archive_x, archive_f):
        """Add (x, f) to archive if it is not dominated. Remove dominated points."""
        if self._is_dominated(f, archive_f):
            return archive_x, archive_f
        if len(archive_f) > 0:
            all_geq = np.all(archive_f >= f, axis=1)
            any_gt = np.any(archive_f > f, axis=1)
            keep_mask = ~(all_geq & any_gt)
            archive_x = archive_x[keep_mask]
            archive_f = archive_f[keep_mask]
        return np.vstack([archive_x, x.reshape(1, -1)]), np.vstack([archive_f, f.reshape(1, -1)])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        eval_count = 0
        archive_x = np.empty((0, dim))
        archive_f = np.empty((0, n_obj))
        all_points = []
        all_fvals = []

        def evaluate(x):
            nonlocal eval_count
            if time.time() > deadline or eval_count >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            all_points.append(x.copy())
            all_fvals.append(f.copy())
            return x, f

        # Phase 1: Halton sequence initial design
        n_halton = min(60, budget)
        halton_points = self._halton(n_halton, dim, skip=1)
        for i in range(n_halton):
            if time.time() > deadline or eval_count >= budget:
                break
            x = lo + halton_points[i] * (hi - lo)
            res = evaluate(x)
            if res is not None:
                x_eval, f_eval = res
                archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)

        # Phase 2: Differential Evolution (DE) for exploration
        pop_size = min(30, max(10, budget - eval_count - 80))  # Leave budget for local refinement
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        if pop_size > 0:
            # Initialize population from archive or random
            if len(archive_x) >= pop_size:
                # Select diverse points from archive
                idx = np.linspace(0, len(archive_x) - 1, pop_size, dtype=int)
                population = archive_x[idx].copy()
            else:
                population = lo + np.random.random((pop_size, dim)) * (hi - lo)

            # Evaluate initial population
            for i in range(pop_size):
                if time.time() > deadline or eval_count >= budget:
                    break
                x = population[i]
                res = evaluate(x)
                if res is not None:
                    x_eval, f_eval = res
                    archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)

            # DE iterations
            max_de_iters = 30
            for gen in range(max_de_iters):
                if time.time() > deadline or eval_count >= budget:
                    break

                for i in range(pop_size):
                    if time.time() > deadline or eval_count >= budget:
                        break

                    # Select three distinct random indices
                    r1, r2, r3 = np.random.choice(pop_size, size=3, replace=False)
                    while r1 == i or r2 == i or r3 == i or r1 == r2 or r2 == r3 or r1 == r3:
                        r1, r2, r3 = np.random.choice(pop_size, size=3, replace=False)

                    # Create trial vector
                    v = population[r1] + F * (population[r2] - population[r3])
                    v = np.clip(v, lo, hi)

                    # Crossover
                    j_rand = np.random.randint(dim)
                    mask = np.random.random(dim) < CR
                    mask[j_rand] = True
                    trial = np.where(mask, v, population[i])

                    # Evaluate trial
                    res = evaluate(trial)
                    if res is None:
                        continue
                    x_trial, f_trial = res

                    # Update archive if non-dominated
                    if not self._is_dominated(f_trial, archive_f):
                        archive_x, archive_f = self._add_to_archive(x_trial, f_trial, archive_x, archive_f)
                        population[i] = x_trial

        # Phase 3: Surrogate-assisted local refinement using RBF interpolation
        # and gradient-based (L-BFGS-B) local search on weighted-sum scalarizations.
        if len(archive_x) >= 5 and time.time() < deadline:
            # Build RBF surrogate for each objective
            try:
                rbf_interpolators = []
                for obj_idx in range(n_obj):
                    rbf = RBFInterpolator(archive_x, archive_f[:, obj_idx], kernel='thin_plate_spline')
                    rbf_interpolators.append(rbf)

                # Generate diverse weight vectors
                n_weights = 15
                rng = np.random.RandomState(42)
                weights = []
                for _ in range(n_weights):
                    w = rng.dirichlet(np.ones(n_obj))
                    weights.append(w)
                # Add extreme weights
                for i in range(n_obj):
                    w = np.zeros(n_obj)
                    w[i] = 1.0
                    weights.append(w)
                # Add uniform weight
                weights.append(np.ones(n_obj) / n_obj)
                weights = np.array(weights)

                # Select starting points from archive
                n_start = min(len(archive_x), len(weights))
                if n_start < len(archive_x):
                    idx = np.linspace(0, len(archive_x) - 1, n_start, dtype=int)
                    start_points = archive_x[idx]
                else:
                    start_points = archive_x

                for i in range(len(start_points)):
                    if time.time() > deadline or eval_count >= budget:
                        break

                    w = weights[i % len(weights)]
                    x0 = start_points[i].copy()

                    def surrogate_objective(x, w=w):
                        if time.time() > deadline:
                            return 1e10
                        x = np.clip(x, lo, hi)
                        pred = np.array([rbf(x) for rbf in rbf_interpolators])
                        return np.dot(w, pred)

                    def surrogate_gradient(x, w=w):
                        if time.time() > deadline:
                            return np.zeros(dim)
                        x = np.clip(x, lo, hi)
                        grad = np.zeros(dim)
                        for obj_idx in range(n_obj):
                            # Numerical gradient of RBF
                            eps = 1e-6
                            for d in range(dim):
                                xp = x.copy()
                                xm = x.copy()
                                xp[d] += eps
                                xm[d] -= eps
                                grad[d] += w[obj_idx] * (rbf_interpolators[obj_idx](xp) - rbf_interpolators[obj_idx](xm)) / (2 * eps)
                        return grad

                    # L-BFGS-B local search using surrogate
                    try:
                        res = minimize(surrogate_objective, x0, method='L-BFGS-B',
                                       jac=surrogate_gradient,
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                        if res.fun < 1e9:
                            x_best = np.clip(res.x, lo, hi)
                            # Evaluate the true objective at the surrogate's optimum
                            res_eval = evaluate(x_best)
                            if res_eval is not None:
                                x_eval, f_eval = res_eval
                                archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)
                    except Exception:
                        pass
            except Exception:
                pass

        # Phase 4: Use remaining budget for random sampling to fill gaps
        if eval_count < budget:
            n_extra = budget - eval_count
            for _ in range(n_extra):
                if time.time() > deadline:
                    break
                x = lo + np.random.random(dim) * (hi - lo)
                res = evaluate(x)
                if res is not None:
                    x_eval, f_eval = res
                    archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)

        if not all_points:
            return lo

        all_points = np.array(all_points)
        all_fvals = np.array(all_fvals)

        return all_points[0]
# EVOLVE-BLOCK-END

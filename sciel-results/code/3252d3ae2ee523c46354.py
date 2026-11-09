"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer: Sobol init + adaptive scalarization + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        used = 0
        archive_X = []
        archive_F = []

        def eval_point(x):
            nonlocal used
            f = np.asarray(problem(np.asarray(x, dtype=float)), dtype=float)
            used += 1
            return f

        def time_ok(margin=5.0):
            return (time.time() + margin) < deadline

        # Phase 1: Sobol / LHS initialization (diverse coverage)
        n_init = min(40, self.budget // 4)
        # Use scipy qmc if available, else stratified random
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            X_init = sampler.random(n_init)
        except Exception:
            # Stratified random (LHS-like)
            X_init = np.zeros((n_init, self.dim))
            for d in range(self.dim):
                for i in range(n_init):
                    X_init[i, d] = (i + np.random.random()) / n_init

        # Clip to bounds
        X_init = np.clip(X_init, lo, hi)

        for x in X_init:
            if not time_ok(10.0):
                break
            f = eval_point(x)
            archive_X.append(x.copy())
            archive_F.append(f.copy())

        # Helper: check if a point is non-dominated in archive
        def is_non_dominated(f_new, F_archive):
            for f_old in F_archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True

        def add_to_archive(x, f):
            if is_non_dominated(f, archive_F):
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                # Remove any archive points dominated by new point
                keep_X = []
                keep_F = []
                for xi, fi in zip(archive_X, archive_F):
                    if np.all(f <= fi) and np.any(f < fi):
                        continue  # dominated by new point
                    keep_X.append(xi)
                    keep_F.append(fi)
                archive_X.clear()
                archive_F.clear()
                archive_X.extend(keep_X)
                archive_F.extend(keep_F)

        # Rebuild archive properly
        # Actually let's just collect all evaluated points and filter at the end
        # But for intermediate use, maintain a simple non-dominated set

        # Phase 2: Adaptive scalarization with local refinement
        # Use a set of weight vectors to cover the objective space
        # For 3 objectives, use a simplex lattice of weights

        # Generate weight vectors (simplex lattice)
        n_weights = 20
        weight_vectors = []
        # Use Fibonacci sphere or random weights on simplex
        for _ in range(n_weights):
            w = np.random.dirichlet(np.ones(self.n_obj))
            weight_vectors.append(w)
        weight_vectors = np.array(weight_vectors)

        # For each weight vector, run a local optimization
        n_local = min(60, self.budget // 3)
        per_weight = max(1, n_local // n_weights)

        for wi in range(n_weights):
            if not time_ok(10.0):
                break
            w = weight_vectors[wi]

            # Scalarized objective: sum w_j * f_j
            # Start from a random point or from archive
            # Pick starting point: random or from archive
            if archive_X and np.random.random() < 0.5:
                x0 = archive_X[np.random.randint(len(archive_X))].copy()
            else:
                x0 = lo + span * np.random.random(self.dim)

            def scalarized(x):
                f = eval_point(x)
                return np.dot(w, f)

            # Nelder-Mead local optimization
            try:
                res = minimize(
                    scalarized,
                    x0,
                    method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-5}
                )
                if time_ok(5.0):
                    f_best = eval_point(res.x)
                    add_to_archive(np.clip(res.x, lo, hi), f_best)
            except Exception:
                pass

        # Phase 3: Polish the current Pareto front with local searches
        # For each archive point, try small perturbations and local refinement
        n_polish = min(40, self.budget - used - 10)
        if n_polish > 0 and archive_F:
            # Sample weight vectors for polishing
            for _ in range(n_polish):
                if not time_ok(5.0):
                    break
                if used >= self.budget - 5:
                    break

                # Pick a random archive point as seed
                idx = np.random.randint(len(archive_X))
                x_seed = archive_X[idx].copy()

                # Small random perturbation
                x_pert = x_seed + 0.05 * span * np.random.standard_normal(self.dim)
                x_pert = np.clip(x_pert, lo, hi)

                # Random weight for local optimization
                w = np.random.dirichlet(np.ones(self.n_obj))

                def scalarized_polish(x):
                    f = eval_point(x)
                    return np.dot(w, f)

                try:
                    res = minimize(
                        scalarized_polish,
                        x_pert,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 20, 'xatol': 1e-5, 'fatol': 1e-6}
                    )
                    if time_ok(3.0):
                        f_res = eval_point(res.x)
                        add_to_archive(np.clip(res.x, lo, hi), f_res)
                except Exception:
                    pass

        # Final: if we have remaining budget, do a few more random samples + polish
        while used < self.budget - 5 and time_ok(5.0):
            x_rand = lo + span * np.random.random(self.dim)
            f_rand = eval_point(x_rand)
            add_to_archive(x_rand, f_rand)

            # Quick local polish from this point
            if used < self.budget - 10 and time_ok(3.0):
                w = np.random.dirichlet(np.ones(self.n_obj))

                def scalarized_final(x):
                    f = eval_point(x)
                    return np.dot(w, f)

                try:
                    res = minimize(
                        scalarized_final,
                        x_rand,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 15, 'xatol': 1e-5, 'fatol': 1e-6}
                    )
                    if time_ok(2.0):
                        f_res = eval_point(res.x)
                        add_to_archive(np.clip(res.x, lo, hi), f_res)
                except Exception:
                    pass

        # Build final result
        if archive_X:
            X_final = np.array(archive_X)
            F_final = np.array(archive_F)
        else:
            X_final = np.array([])
            F_final = np.array([])

        return X_final

# EVOLVE-BLOCK-END

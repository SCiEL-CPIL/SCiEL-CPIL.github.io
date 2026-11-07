# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 using Tchebycheff decomposition + L-BFGS-B scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        def is_deadline():
            return (deadline - time.time()) < 8.0

        def clip(x):
            return np.clip(x, lo, hi)

        # Phase 1: Sobol quasi-random sampling for initial exploration
        from scipy.stats import qmc

        n_sobol = min(120, budget)
        sampler = qmc.Sobol(d=dim, scramble=True)
        n_points = 1
        while n_points < n_sobol:
            n_points *= 2
        sobol_samples = sampler.random(n_points)[:n_sobol]
        X_sobol = lo + sobol_samples * (hi - lo)

        F_all = []
        X_all = []
        for i in range(n_sobol):
            if is_deadline():
                break
            x = X_sobol[i]
            f = problem(x)
            F_all.append(f)
            X_all.append(x)

        F_all = np.array(F_all)
        X_all = np.array(X_all)
        used = len(F_all)

        if used == 0:
            # Fallback: single random point
            x = lo + np.random.random(dim) * (hi - lo)
            f = problem(x)
            return np.array([f])

        # Helper: check if a point is dominated
        def is_dominated(f_new, F_front):
            if len(F_front) == 0:
                return False
            # Check if any point in front dominates f_new
            leq = np.all(F_front <= f_new, axis=1)
            lt = np.any(F_front < f_new, axis=1)
            return np.any(leq & lt)

        # Helper: get non-dominated front
        def get_non_dominated(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            F_i = F[:, np.newaxis, :]
            F_j = F[np.newaxis, :, :]
            leq = np.all(F_i <= F_j, axis=2)
            lt = np.any(F_i < F_j, axis=2)
            dominates = leq & lt
            dominated = np.any(dominates, axis=0)
            return np.where(~dominated)[0]

        # Phase 2: Tchebycheff decomposition with L-BFGS-B
        # Generate weight vectors on the simplex
        # For 3 objectives, use a structured grid
        n_w = 6  # points per edge
        weight_vectors = []
        for i in range(n_w + 1):
            for j in range(n_w + 1 - i):
                k = n_w - i - j
                w = np.array([i / n_w, j / n_w, k / n_w])
                weight_vectors.append(w)
        W = np.array(weight_vectors)

        # Add some random weights for diversity
        rng = np.random.default_rng(123)
        n_extra = 20
        for _ in range(n_extra):
            w = rng.dirichlet(np.ones(n_obj))
            weight_vectors.append(w)
        W = np.array(weight_vectors)

        # Shuffle weights for diversity
        perm = rng.permutation(len(W))
        W = W[perm]

        # Track the current front
        front_F = F_all[get_non_dominated(F_all)]
        ideal = np.min(front_F, axis=0)
        nadir = np.max(front_F, axis=0)

        # For each weight vector, optimize the Tchebycheff scalarization
        for w in W:
            if is_deadline() or used >= budget:
                break

            # Pick a starting point from the current front (or random if front is empty)
            if len(front_F) > 0:
                # Pick a random point from the front
                idx = rng.integers(0, len(front_F))
                x0 = X_all[get_non_dominated(F_all)][idx].copy()
            else:
                x0 = lo + np.random.random(dim) * (hi - lo)

            # Small perturbation for diversity
            x0 = clip(x0 + rng.normal(0, 0.1, dim))

            def tcheb_obj(x, w=w, ideal=ideal):
                f = problem(x)
                return np.max(w * (f - ideal))

            try:
                res = minimize(
                    tcheb_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x_opt = clip(res.x)
                f_opt = problem(x_opt)
                used += 1

                # Check if this point is useful (not dominated)
                if not is_dominated(f_opt, front_F):
                    # Add to collection
                    F_all = np.vstack([F_all, [f_opt]])
                    X_all = np.vstack([X_all, [x_opt]])

                    # Update front
                    nd_idx = get_non_dominated(F_all)
                    front_F = F_all[nd_idx]
                    ideal = np.min(front_F, axis=0)

            except Exception:
                pass

        # Phase 3: Local refinement around the best front points
        remaining = budget - used
        if remaining > 0 and not is_deadline():
            nd_idx = get_non_dominated(F_all)
            front_idx = nd_idx

            # Refine each front point with small perturbations
            for idx in front_idx:
                if is_deadline() or used >= budget:
                    break

                x_target = X_all[idx].copy()

                # Try a few perturbations
                for _ in range(3):
                    if is_deadline() or used >= budget:
                        break

                    x_new = x_target + rng.normal(0, 0.05, dim)
                    x_new = clip(x_new)

                    # Use a random weight for local optimization
                    w = rng.dirichlet(np.ones(n_obj))

                    def local_tcheb(x, w=w, ideal=ideal):
                        f = problem(x)
                        return np.max(w * (f - ideal))

                    try:
                        res = minimize(
                            local_tcheb,
                            x_new,
                            method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 30, 'ftol': 1e-12}
                        )
                        x_opt = clip(res.x)
                        f_opt = problem(x_opt)
                        used += 1

                        if not is_dominated(f_opt, front_F):
                            F_all = np.vstack([F_all, [f_opt]])
                            X_all = np.vstack([X_all, [x_opt]])

                            nd_idx = get_non_dominated(F_all)
                            front_F = F_all[nd_idx]
                            ideal = np.min(front_F, axis=0)

                    except Exception:
                        pass

        # Return all evaluated points (evaluator records them all)
        return F_all

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


class Optimizer:
    """Multi-objective optimizer for RE34 using Tchebycheff decomposition + SA refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        def is_deadline():
            return (deadline - time.time()) < 5.0

        def clip(x):
            return np.clip(x, lo, hi)

        # Generate Tchebycheff weight vectors (deterministic grid on simplex)
        # For 3 objectives, use a structured grid
        n_w = 5  # points per edge of the simplex
        weight_vectors = []
        for i in range(n_w + 1):
            for j in range(n_w + 1 - i):
                k = n_w - i - j
                w = np.array([i / n_w, j / n_w, k / n_w])
                weight_vectors.append(w)
        W = np.array(weight_vectors)  # 21 weight vectors for n_w=5

        # Phase 1: Evaluate a structured set of initial points
        # Use a combination of:
        # 1. Center point
        # 2. Axis-aligned points (min/max for each objective direction)
        # 3. Random points
        rng = np.random.default_rng(123)

        # Generate diverse initial points
        n_initial = min(30, budget // 6)  # Use ~5% of budget for initialization
        X_init = []

        # Add center
        X_init.append((lo + hi) / 2)

        # Add points along each axis at several levels
        for d in range(dim):
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x = (lo + hi) / 2
                x[d] = lo[d] + frac * (hi[d] - lo[d])
                X_init.append(x)

        # Add random points to fill up to n_initial
        while len(X_init) < n_initial:
            x = lo + rng.random(dim) * (hi - lo)
            X_init.append(x)

        X_init = np.array(X_init[:n_initial])
        F_init = np.array([problem(x) for x in X_init])
        used = n_initial

        # Phase 2: For each weight vector, use the best initial point as starting point
        # and refine with Simulated Annealing on the Tchebycheff scalarized objective
        # Tchebycheff: min max_i w_i * (f_i(x) - z_i*)
        # where z_i* is the ideal point (minimum of each objective)

        # First, estimate ideal point from initial evaluations
        z_star = F_init.min(axis=0)

        # For each weight, find best initial point by Tchebycheff distance
        best_start_idx = np.zeros(len(W), dtype=int)
        for wi, w in enumerate(W):
            # Tchebycheff distance for each initial point
            if np.all(w > 0):
                tcheb = np.max(w * (F_init - z_star) / (w + 1e-10), axis=1)
            else:
                tcheb = np.sum(w * (F_init - z_star), axis=1)
            best_start_idx[wi] = np.argmin(tcheb)

        # Phase 3: Simulated Annealing refinement for each weight
        # We have budget - used evaluations left
        remaining = budget - used

        # Allocate evaluations: ~20 SA steps per weight on average
        # But be adaptive: spend more on weights that seem promising
        n_weights = len(W)
        evals_per_weight = max(1, remaining // (n_weights * 2))  # Leave some for final polish
        # Actually, let's do 2 phases: coarse SA then fine SA
        # Phase 3a: Coarse SA (larger steps)
        # Phase 3b: Fine SA (smaller steps)

        # Track best solution per weight
        best_x_per_w = X_init[best_start_idx].copy()
        best_f_per_w = F_init[best_start_idx].copy()
        best_tcheb_per_w = np.zeros(n_weights)

        # Compute initial Tchebycheff values
        for wi, w in enumerate(W):
            if np.all(w > 0):
                best_tcheb_per_w[wi] = np.max(w * (best_f_per_w[wi] - z_star) / (w + 1e-10))
            else:
                best_tcheb_per_w[wi] = np.sum(w * (best_f_per_w[wi] - z_star))

        # SA parameters
        T0 = 1.0  # Initial temperature (in Tchebycheff units)
        alpha = 0.95  # Cooling rate
        n_sa_iters = max(5, evals_per_weight // 2)

        # For each weight, run SA
        for wi, w in enumerate(W):
            if is_deadline() or used >= budget:
                break

            x_cur = best_x_per_w[wi].copy()
            f_cur = best_f_per_w[wi].copy()
            tcheb_cur = best_tcheb_per_w[wi]

            x_best = x_cur.copy()
            f_best = f_cur.copy()
            tcheb_best = tcheb_cur

            T = T0
            for it in range(n_sa_iters):
                if is_deadline() or used >= budget:
                    break

                # Generate neighbor: Gaussian perturbation
                # Step size decreases with temperature
                step_size = 0.5 * T
                x_new = x_cur + rng.normal(0, step_size, dim)
                x_new = clip(x_new)

                f_new = problem(x_new)
                used += 1

                if used > budget:
                    break

                # Update ideal point
                z_star = np.minimum(z_star, f_new)

                # Compute Tchebycheff
                if np.all(w > 0):
                    tcheb_new = np.max(w * (f_new - z_star) / (w + 1e-10))
                else:
                    tcheb_new = np.sum(w * (f_new - z_star))

                # Accept if better or with probability exp(-delta/T)
                delta = tcheb_new - tcheb_cur
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    x_cur = x_new
                    f_cur = f_new
                    tcheb_cur = tcheb_new

                    if tcheb_new < tcheb_best:
                        x_best = x_new.copy()
                        f_best = f_new.copy()
                        tcheb_best = tcheb_new

                T *= alpha

            # Update global best for this weight
            best_x_per_w[wi] = x_best
            best_f_per_w[wi] = f_best
            best_tcheb_per_w[wi] = tcheb_best

        # Phase 4: Collect all evaluated points and return
        # We need to return the set of non-dominated points
        # But the evaluator records ALL returned objective vectors, so we should
        # return a diverse set that covers the Pareto front well

        # Collect all unique points evaluated
        # We tracked best per weight, but we also have F_init
        # Let's combine F_init and best_f_per_w
        all_F = np.vstack([F_init, best_f_per_w])

        # Remove dominated points to get a clean Pareto front
        def non_dominated_mask(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=bool)
            F_i = F[:, np.newaxis, :]
            F_j = F[np.newaxis, :, :]
            leq = np.all(F_i <= F_j, axis=2)
            lt = np.any(F_i < F_j, axis=2)
            dominates = leq & lt
            dominated = np.any(dominates, axis=0)
            return ~dominated

        nd_mask = non_dominated_mask(all_F)
        nd_F = all_F[nd_mask]

        # Return the non-dominated set
        # The return value is ignored, but we need to not crash
        return nd_F if len(nd_F) > 0 else F_init

# EVOLVE-BLOCK-END

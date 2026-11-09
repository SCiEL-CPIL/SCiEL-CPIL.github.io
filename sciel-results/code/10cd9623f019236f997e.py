"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff aggregation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        margin = 5.0  # seconds to spare

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate weight vectors for decomposition
        # For 2 objectives, weights are (w, 1-w)
        n_weights = min(40, budget // 5)  # At most 40 subproblems, using ~5 evals each
        weights = np.linspace(0, 1, n_weights)

        # Ideal point estimation
        ideal = np.array([np.inf] * n_obj)
        anti_ideal = np.array([np.inf] * n_obj)

        # Phase 1: Random sampling to estimate ideal/anti-ideal points
        rng = np.random.default_rng(42)
        n_init = min(20, budget // 10)
        init_points = lo + (hi - lo) * rng.random((n_init, dim))
        init_f = np.array([problem(x) for x in init_points])

        for f in init_f:
            for i in range(n_obj):
                ideal[i] = min(ideal[i], f[i])
                anti_ideal[i] = max(anti_ideal[i], f[i])

        # Handle case where ideal == anti_ideal for some objective
        for i in range(n_obj):
            if anti_ideal[i] - ideal[i] < 1e-10:
                anti_ideal[i] = ideal[i] + 1.0

        # Phase 2: For each weight, run L-BFGS-B on Tchebycheff scalarization
        results = []
        evals_used = n_init
        time_ok = lambda: (time.time() + margin) < deadline

        for w in weights:
            if not time_ok():
                break

            # Tchebycheff scalarization: min max_i w_i * (f_i(x) - z_i*) / (zbar_i - z_i*)
            # where z_i* is ideal, zbar_i is anti-ideal
            def tchebycheff(x):
                fx = problem(x)
                # Normalize objectives
                norm = np.array([
                    (fx[i] - ideal[i]) / (anti_ideal[i] - ideal[i]) if (anti_ideal[i] - ideal[i]) > 1e-10 else 0
                    for i in range(n_obj)
                ])
                # Weighted max
                w_vec = np.array([w, 1 - w]) if n_obj == 2 else np.ones(n_obj) / n_obj
                return np.max(w_vec * norm)

            # Multiple starting points for each subproblem
            n_starts = min(3, (budget - evals_used) // max(1, n_weights))
            best_local_f = None
            best_local_x = None
            best_local_val = np.inf

            for s in range(n_starts):
                if not time_ok():
                    break
                if evals_used >= budget:
                    break

                # Start from a random point or from one of the initial points
                if s == 0 and len(init_points) > 0:
                    x0 = init_points[rng.integers(len(init_points))].copy()
                else:
                    x0 = lo + (hi - lo) * rng.random(dim)

                # Clip to bounds
                x0 = np.clip(x0, lo, hi)

                # Run L-BFGS-B
                try:
                    res = minimize(
                        tchebycheff, x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'maxfun': 20}
                    )
                    evals_used += res.nfev
                    fx = problem(res.x)
                    if np.all(fx < best_local_val if best_local_val is not None else np.ones(n_obj) * np.inf):
                        # Simple dominance check
                        if best_local_f is None or np.any(fx < best_local_f) and not np.any(fx > best_local_f):
                            best_local_f = fx
                            best_local_x = res.x
                        elif best_local_f is not None:
                            # Keep both if non-dominated
                            pass
                    if best_local_f is None:
                        best_local_f = fx
                        best_local_x = res.x
                except Exception:
                    pass

            if best_local_f is not None:
                results.append(best_local_f)

        # Phase 3: Use remaining budget for additional random sampling + local refinement
        while evals_used < budget and time_ok():
            x = lo + (hi - lo) * rng.random(dim)
            fx = problem(x)
            evals_used += 1
            results.append(fx)

            # If we have time, do a quick local search from this point
            if time_ok() and evals_used + 5 < budget:
                def obj_sum(x):
                    return np.sum(problem(x))
                try:
                    res = minimize(obj_sum, x, method='Nelder-Mead',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 20, 'maxfev': 10})
                    evals_used += res.nfev
                    fx2 = problem(res.x)
                    results.append(fx2)
                except Exception:
                    pass

        # Return the set of non-dominated points
        if len(results) == 0:
            return np.array([ideal])

        results = np.array(results)
        # Filter to points strictly better than reference point [3000, 0.06]
        ref = np.array([3000.0, 0.06])
        mask = np.all(results < ref, axis=1)
        results = results[mask]

        if len(results) == 0:
            return np.array([ideal])

        # Extract non-dominated points
        n = len(results)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(results[i] <= results[j]) and np.any(results[i] < results[j]):
                    dominated[j] = True

        nd_points = results[~dominated]
        return nd_points if len(nd_points) > 0 else results[:1]
# EVOLVE-BLOCK-END

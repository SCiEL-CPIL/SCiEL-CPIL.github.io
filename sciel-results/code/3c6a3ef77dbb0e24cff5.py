"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # Safety margin before 300s limit

        # Track best value seen
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return None
            call_count += 1
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start Nelder-Mead
        rng = np.random.default_rng(42)
        n_starts = 8
        calls_per_start = budget // n_starts

        for start_idx in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break

            # Initialize simplex
            if start_idx == 0:
                center = rng.uniform(-4.0, 4.0, dim)
            else:
                # Perturb around best found so far, or random
                if best_x is not None:
                    center = np.clip(best_x + rng.normal(0, 1.0, dim), -4.5, 4.5)
                else:
                    center = rng.uniform(-4.0, 4.0, dim)

            # Build simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(1, dim + 1):
                simplex[i] = center.copy()
                simplex[i][i - 1] += 0.5  # Initial step size

            # Evaluate simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = safe_call(simplex[i])
                if f is None:
                    f = 1e10
                simplex_f[i] = f

            if call_count >= budget or time.time() > deadline:
                break

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            max_iter = calls_per_start - (dim + 1)
            stall_count = 0
            stall_limit = 20

            for _ in range(max(0, max_iter)):
                if call_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                # Check convergence
                if stall_count >= stall_limit:
                    break

                centroid = np.mean(simplex[1:], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_call(x_ref)
                if f_ref is None:
                    f_ref = 1e10

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - worst)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = safe_call(x_exp)
                    if f_exp is None:
                        f_exp = 1e10
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    stall_count = 0
                elif f_ref < simplex_f[-2]:
                    # Simple reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                    stall_count = 0
                else:
                    # Contraction
                    if f_ref < worst_f:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_call(x_con)
                    if f_con is None:
                        f_con = 1e10
                    if f_con < min(f_ref, worst_f):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                        stall_count = 0
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = safe_call(simplex[i])
                            if f is None:
                                f = 1e10
                            simplex_f[i] = f
                        stall_count += 1

                        # If simplex is too small, break
                        spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                        if np.max(spread) < 1e-6:
                            break

            # After Nelder-Mead, if we have budget left, do local refinement
            if call_count < budget and time.time() < deadline and best_x is not None:
                # Fine-tune around best with small perturbations
                local_budget = min(budget - call_count, 20)
                step = 0.1
                for _ in range(local_budget):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x_try = np.clip(best_x + rng.normal(0, step, dim), -5.0, 5.0)
                    f_try = safe_call(x_try)
                    if f_try is None:
                        break
                    if f_try < best_f:
                        step *= 0.7
                    else:
                        step *= 1.2
                        step = min(step, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END

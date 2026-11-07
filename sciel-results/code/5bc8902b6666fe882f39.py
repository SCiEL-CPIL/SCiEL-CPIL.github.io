"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts for Schaffer's F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 285.0

        def time_ok():
            return time.time() < DEADLINE

        calls_made = [0]

        def safe_call(x):
            """Call func with clipping, return (x_clipped, f_value)."""
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_made[0] += 1
            try:
                fval = float(func(x_clipped))
            except Exception:
                fval = np.inf
            if not np.isfinite(fval):
                fval = np.inf
            return x_clipped, fval

        def nelder_mead(x0, f0, max_calls, deadline_check):
            """Nelder-Mead simplex method with adaptive step size."""
            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            # Create initial simplex around x0
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += 0.25
                x_new = np.clip(x_new, -5, 5)
                x_c, f_c = safe_call(x_new)
                simplex.append(x_c)
                fvals.append(f_c)
                if not deadline_check() or calls_made[0] >= max_calls:
                    break

            best_x = x0.copy()
            best_f = f0
            for x, f in zip(simplex, fvals):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            max_iter = 200
            stagnation = 0
            max_stagnation = 30

            for iteration in range(max_iter):
                if not deadline_check() or calls_made[0] >= max_calls:
                    break

                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > max_stagnation:
                    break

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                x_c, f_reflect = safe_call(x_reflect)
                if not deadline_check() or calls_made[0] >= max_calls:
                    break

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    x_c, f_expand = safe_call(x_expand)
                    if not deadline_check() or calls_made[0] >= max_calls:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    x_c, f_contract = safe_call(x_contract)
                    if not deadline_check() or calls_made[0] >= max_calls:
                        break
                    if f_contract < fvals[-1]:
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            x_c, f_shrink = safe_call(simplex[i])
                            if not deadline_check() or calls_made[0] >= max_calls:
                                break
                            fvals[i] = f_shrink

            return best_x, best_f

        # Stage 1: Coarse random sampling to find a good starting point
        rng = np.random.default_rng(42)
        n_samples = 20
        best_x = None
        best_f = np.inf
        for _ in range(n_samples):
            if not time_ok() or calls_made[0] >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            x_c, f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x_c.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = np.inf

        # Stage 2: Nelder-Mead from best sample
        remaining = budget - calls_made[0]
        if remaining > 0 and time_ok():
            best_x, best_f = nelder_mead(best_x, best_f, remaining, time_ok)

        # Stage 3: Fine-tuning with small perturbations
        if time_ok() and best_f < np.inf and calls_made[0] < budget:
            for coord in range(dim):
                if not time_ok() or calls_made[0] >= budget:
                    break
                for step in [0.05, 0.02, 0.01, 0.005]:
                    if not time_ok() or calls_made[0] >= budget:
                        break
                    for direction in [1, -1]:
                        if not time_ok() or calls_made[0] >= budget:
                            break
                        x_new = best_x.copy()
                        x_new[coord] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        _, f_new = safe_call(x_new)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END

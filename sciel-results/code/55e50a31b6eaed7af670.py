"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer: random restarts + Nelder-Mead + coordinate search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        # Wrap func with a call counter and time guard
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            call_count += 1
            f = float(func(x))
            if not np.isfinite(f):
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Broad random sampling to find promising basins
        n_random = min(50, budget)
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)

        # Phase 2: Nelder-Mead from multiple promising starts
        # Sort by function value and pick top candidates
        # We'll do a few Nelder-Mead runs from the best few points

        # Collect evaluated points
        # Since we don't store all, let's just do NM from the global best and a few random ones

        def nelder_mead(x0, max_iter=200):
            """Run Nelder-Mead locally, using safe_call."""
            nonlocal call_count, best_x, best_f
            x = x0.copy()
            fx = safe_call(x)

            # Initialize simplex
            simplex = [x.copy()]
            fs = [fx]

            for i in range(dim):
                xi = x.copy()
                xi[i] += 0.5 if xi[i] < 4.5 else -0.5
                xi = np.clip(xi, -5, 5)
                fi = safe_call(xi)
                simplex.append(xi)
                fs.append(fi)

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break

                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i].copy() for i in order]
                fs = [fs[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_call(x_r)

                if f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_call(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    # Contraction
                    if f_r < fs[-1]:
                        x_c = centroid + rho * (simplex[-1] - centroid)
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_call(x_c)
                    if f_c < min(f_r, fs[-1]):
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fs[i] = safe_call(simplex[i])
                        if time.time() > deadline:
                            break
                        if call_count >= budget:
                            break

            return simplex[0], fs[0]

        # Do Nelder-Mead from the best point found so far
        if best_x is not None:
            nelder_mead(best_x, max_iter=300)

        # Do Nelder-Mead from a few other random points
        n_nm_extra = 5
        for _ in range(n_nm_extra):
            if time.time() > deadline or call_count >= budget:
                break
            x_start = rng.uniform(-5, 5, dim)
            nelder_mead(x_start, max_iter=150)

        # Phase 3: Fine-tuning with coordinate descent on the best point
        if best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = safe_call(x)
            step = 0.5
            for _ in range(200):
                if time.time() > deadline or call_count >= budget:
                    break
                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = safe_call(x_try)
                        if f_try < fx:
                            x = x_try
                            fx = f_try
                            improved = True
                            break
                    if time.time() > deadline or call_count >= budget:
                        break
                if not improved:
                    step *= 0.7
                    if step < 1e-4:
                        break
                else:
                    step = max(step * 0.9, 1e-4)

        return best_x, best_f
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Two-stage Sobol global search followed by multi-start Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return calls < budget and (time.time() - start_time) < deadline

        def time_ok_early():
            return calls < budget and (time.time() - start_time) < deadline - 10.0

        # Phase 1: Sobol quasi-random global sampling
        n_sobol = min(400, budget - 50)  # reserve some budget for local refinement
        n_sobol = max(n_sobol, 50)

        sampler = qmc.Sobol(d=dim, scramble=True)
        # Use a power of 2 for Sobol efficiency
        n2 = 1
        while n2 < n_sobol:
            n2 *= 2

        samples = sampler.random(n2)
        # Map from [0, 1]^dim to [-5, 5]^dim
        samples = samples * 10.0 - 5.0

        evaluated = []
        for i in range(min(n_sobol, n2)):
            if not time_ok_early():
                break
            x = samples[i]
            f = func(x)
            calls += 1
            evaluated.append((x, f))
            try_update(x, f)

        if len(evaluated) == 0:
            # Fallback: single point
            x = np.zeros(dim)
            best_f = func(x)
            calls += 1
            best_x = x
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead from top candidates
        # Sort by fitness
        evaluated.sort(key=lambda t: t[1])

        # Take top 5 candidates (or fewer if we don't have that many)
        n_starts = min(5, len(evaluated))
        starts = [evaluated[i][0].copy() for i in range(n_starts)]

        # Also add the overall best
        if best_x is not None and best_f < np.inf:
            # Check if best_x is already in starts
            for s in starts:
                if np.allclose(s, best_x):
                    break
            else:
                if len(starts) < 6:
                    starts.append(best_x.copy())

        # Phase 3: Nelder-Mead local optimization from each start
        def nelder_mead(x0, f0, max_iter=200):
            """Deterministic Nelder-Mead with adaptive simplex."""
            nonlocal calls, best_x, best_f

            d = dim
            # Initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]

            # Scale initial simplex based on distance to other candidates
            # Use a moderate initial step size
            init_step = 1.0

            for i in range(d):
                v = x0.copy()
                v[i] += init_step
                v = np.clip(v, -5, 5)
                simplex.append(v)

            # Evaluate remaining simplex vertices
            for i in range(1, d + 1):
                if not time_ok():
                    break
                f = func(simplex[i])
                calls += 1
                simplex_f.append(f)
                try_update(simplex[i], f)

            if len(simplex_f) < d + 1:
                return x0, f0

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            for iteration in range(max_iter):
                if not time_ok():
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                try_update(x_r, f_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    try_update(x_e, f_e)

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)

                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    try_update(x_c, f_c)

                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            if not time_ok():
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = func(simplex[i])
                            calls += 1
                            simplex_f[i] = f_i
                            try_update(simplex[i], f_i)

            # Return best from this local search
            best_local_idx = np.argmin(simplex_f)
            return simplex[best_local_idx], simplex_f[best_local_idx]

        # Run Nelder-Mead from each starting point
        for i, start in enumerate(starts):
            if not time_ok():
                break

            # Get current best fitness at this start (we may have already evaluated it)
            # Find if this start was in our evaluated set
            f_start = None
            for x_eval, f_eval in evaluated:
                if np.allclose(x_eval, start):
                    f_start = f_eval
                    break

            if f_start is None:
                if not time_ok():
                    break
                f_start = func(start)
                calls += 1
                try_update(start, f_start)

            # Allocate more iterations to earlier (better) starts
            remaining_starts = len(starts) - i
            alloc_iter = max(50, 200 * budget // max(1, remaining_starts * 5))
            alloc_iter = min(alloc_iter, 300)

            try:
                local_x, local_f = nelder_mead(start, f_start, max_iter=alloc_iter)
                if local_f < best_f:
                    best_f = local_f
                    best_x = local_x.copy()
            except Exception:
                pass

        # Phase 4: Final coordinate-wise refinement around the best point
        if time_ok() and best_f < np.inf:
            x = best_x.copy()
            fx = best_f
            step = 0.5

            for coord in range(dim):
                if not time_ok():
                    break
                for _ in range(15):
                    if not time_ok():
                        break
                    improved = False
                    for sign in [1, -1]:
                        if not time_ok():
                            break
                        y = x.copy()
                        y[coord] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < fx:
                            x, fx = y, fy
                            try_update(x, fx)
                            improved = True
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END

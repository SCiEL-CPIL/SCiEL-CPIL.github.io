"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble with LHS initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Generate diverse starting points using Latin Hypercube Sampling
        n_starts = 5
        lhs_points = self._latin_hypercube(rng, n_starts, dim, low=-4.0, high=4.0)

        # Initialize multiple Nelder-Mead simplexes concurrently
        # Each simplex needs dim+1 points
        alpha = 0.5  # Initial simplex spread
        active_simplexes = []  # List of (simplex_list, f_list, converged)

        for start_idx, x0 in enumerate(lhs_points):
            if not time_ok() or calls >= budget:
                break

            # Build simplex for this start point
            simplex = [x0.copy()]
            f_simplex = [func(x0)]
            calls += 1
            record(x0, f_simplex[0])

            for i in range(dim):
                if not time_ok() or calls >= budget:
                    break
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                record(x_i, f_i)

            if len(simplex) == dim + 1:
                active_simplexes.append({
                    'simplex': simplex,
                    'f': f_simplex,
                    'converged': False
                })

        if not active_simplexes:
            return best_x, best_f

        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5
        convergence_tol = 1e-10

        # Round-robin over active simplexes
        while calls < budget and time_ok():
            any_active = False

            for s in active_simplexes:
                if s['converged'] or not time_ok() or calls >= budget:
                    continue

                any_active = True
                simplex = s['simplex']
                f_simplex = s['f']

                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                s['simplex'] = simplex
                s['f'] = f_simplex

                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]

                # Check for convergence
                if f_worst - f_best < convergence_tol:
                    s['converged'] = True
                    continue

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_r = func(x_r)
                calls += 1
                record(x_r, f_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not time_ok() or calls >= budget:
                        break
                    f_e = func(x_e)
                    calls += 1
                    record(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)

                    if not time_ok() or calls >= budget:
                        break
                    f_c = func(x_c)
                    calls += 1
                    record(x_c, f_c)

                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok() or calls >= budget:
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            record(simplex[i], f_simplex[i])

                s['simplex'] = simplex
                s['f'] = f_simplex

            if not any_active:
                break

        # Final polish: run a few more NM iterations on the best simplex if budget remains
        if calls < budget and time_ok() and best_x is not None:
            # Find which simplex has the best point
            best_s_idx = None
            best_s_f = np.inf
            for idx, s in enumerate(active_simplexes):
                if s['f'][0] < best_s_f:
                    best_s_f = s['f'][0]
                    best_s_idx = idx

            if best_s_idx is not None:
                s = active_simplexes[best_s_idx]
                simplex = s['simplex']
                f_simplex = s['f']

                # Run additional NM iterations with tighter tolerance
                polish_tol = 1e-12
                for _ in range(min(50, budget - calls)):
                    if not time_ok() or calls >= budget:
                        break

                    indices = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in indices]
                    f_simplex = [f_simplex[i] for i in indices]

                    x_best = simplex[0]
                    f_best = f_simplex[0]
                    x_worst = simplex[-1]
                    f_worst = f_simplex[-1]

                    if f_worst - f_best < polish_tol:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + gamma * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    if not time_ok() or calls >= budget:
                        break
                    f_r = func(x_r)
                    calls += 1
                    record(x_r, f_r)

                    if f_r < f_best:
                        x_e = centroid + rho * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        if not time_ok() or calls >= budget:
                            break
                        f_e = func(x_e)
                        calls += 1
                        record(x_e, f_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_simplex[-2]:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        if f_r < f_worst:
                            x_c = centroid + sigma_nm * (x_r - centroid)
                        else:
                            x_c = centroid - sigma_nm * (x_worst - centroid)
                        x_c = np.clip(x_c, -5, 5)

                        if not time_ok() or calls >= budget:
                            break
                        f_c = func(x_c)
                        calls += 1
                        record(x_c, f_c)

                        if f_c < min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            for i in range(1, len(simplex)):
                                if not time_ok() or calls >= budget:
                                    break
                                simplex[i] = x_best + shrink * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = func(simplex[i])
                                calls += 1
                                record(simplex[i], f_simplex[i])

        return best_x, best_f

    def _latin_hypercube(self, rng, n_samples, dim, low=-4.0, high=4.0):
        """Generate Latin Hypercube sample points."""
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            # Create stratified intervals
            intervals = np.linspace(low, high, n_samples + 1)
            for i in range(n_samples):
                # Sample uniformly within each interval
                points[i, d] = rng.uniform(intervals[i], intervals[i + 1])
        # Shuffle rows to decorrelate dimensions
        rng.shuffle(points)
        return points
# EVOLVE-BLOCK-END

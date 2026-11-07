"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic Nelder-Mead simplex with adaptive random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        dim = self.dim
        total_budget = self.budget
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_made += 1
            return val

        best_x = None
        best_f = np.inf

        def nelder_mead(x0, max_iter, step_scale=1.0):
            nonlocal calls_made, best_f, best_x

            # Initialize simplex with adaptive step
            step = 0.5 * step_scale
            simplex = [x0.copy()]
            for i in range(dim):
                dx = x0.copy()
                dx[i] += step
                simplex.append(np.clip(dx, -5, 5))

            fvals = []
            for s in simplex:
                if calls_made >= total_budget:
                    break
                fvals.append(safe_func(s))

            if len(fvals) < dim + 1:
                return x0.copy(), best_f

            # Track convergence
            prev_best = fvals[0]

            for _ in range(max_iter):
                if calls_made >= total_budget:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Stop criteria
                if abs(fvals[0] - fvals[-1]) < 1e-12:
                    break
                center = np.mean(simplex, axis=0)
                size = np.mean([np.linalg.norm(s - center) for s in simplex])
                if size < 1e-7:
                    break

                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + 1.0 * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_func(x_r)

                if calls_made >= total_budget:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + 2.0 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_func(x_e)
                    if calls_made >= total_budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)

                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_func(x_c)
                    if calls_made >= total_budget:
                        break

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_made >= total_budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])
                        if calls_made >= total_budget:
                            break

                # Early exit if no progress
                if fvals[0] >= prev_best - 1e-10:
                    break
                prev_best = fvals[0]

            return simplex[0].copy(), fvals[0]

        # Phase 1: Diverse random initialization with stratified sampling
        n_init = 15
        # Use stratified random sampling for better coverage
        points = []
        for i in range(n_init):
            if calls_made >= total_budget:
                break
            # Mix of uniform and grid-based points
            if i < dim + 1:
                # Grid-like points for better coverage
                x = np.zeros(dim)
                for d in range(dim):
                    x[d] = rng.choice([-4, -2, 0, 2, 4])
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            points.append(x)

        for x in points:
            if calls_made >= total_budget:
                break
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None, np.inf

        # Phase 2: Run NM from best point with moderate budget
        remaining = total_budget - calls_made
        if remaining > 20:
            nelder_mead(best_x, max_iter=min(100, remaining - 10), step_scale=1.0)

        # Phase 3: Multiple restarts from diverse points
        remaining = total_budget - calls_made
        if remaining > 30:
            n_restarts = min(4, remaining // 40)
            for r in range(n_restarts):
                if calls_made >= total_budget - 20:
                    break
                # Generate restart point near best or random
                if r % 2 == 0:
                    # Perturb best point
                    x = best_x + rng.normal(0, 0.5, dim)
                else:
                    # Random point
                    x = rng.uniform(-4.5, 4.5, dim)
                x = np.clip(x, -5, 5)
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    if calls_made < total_budget - 20:
                        nelder_mead(best_x, max_iter=min(30, (total_budget - calls_made) - 10), step_scale=0.5)
                elif calls_made < total_budget - 20:
                    # Still run NM from this point even if not better
                    nelder_mead(x, max_iter=min(20, (total_budget - calls_made) - 10), step_scale=0.5)

        # Phase 4: Final local refinement around best
        remaining = total_budget - calls_made
        if remaining > 15:
            nelder_mead(best_x, max_iter=min(50, remaining - 5), step_scale=0.2)

        return best_x, best_f
# EVOLVE-BLOCK-END

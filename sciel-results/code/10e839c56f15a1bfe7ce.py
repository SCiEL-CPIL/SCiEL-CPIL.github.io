"""Optimizer for BBOB f19 (Griewank-Rosenbrock) using Sobol quasi-random + L-BFGS-B + Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import comb


class Optimizer:
    """Sobol quasi-random exploration + L-BFGS-B local refinement + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # 300s limit, 10s safety margin

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def eval_point(x):
            nonlocal calls_used, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Sobol quasi-random sequence to map the space
        # Generate Sobol points for 5 dimensions
        n_sobol = 256  # power of 2 for Sobol
        sobol_points = self._sobol_sequence(n_sobol, dim)

        # Evaluate Sobol points
        for i in range(n_sobol):
            if calls_used >= budget or time.time() > deadline:
                break
            x = sobol_points[i]
            eval_point(x)

        # Phase 2: Local refinement from top candidates using L-BFGS-B
        # We don't have stored fvals, so let's track them
        # Let me restructure to store evaluation results

        # Actually, let me redo this properly with stored results
        # Reset and restart with proper tracking

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        eval_results = []  # list of (x, f)

        def eval_point_tracked(x):
            nonlocal calls_used, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_used += 1
            eval_results.append((x_clipped.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Sobol quasi-random exploration
        n_sobol = 200
        sobol_points = self._sobol_sequence(n_sobol, dim)

        for i in range(n_sobol):
            if calls_used >= budget or time.time() > deadline:
                break
            x = sobol_points[i]
            eval_point_tracked(x)

        # Phase 2: L-BFGS-B from top K candidates
        if calls_used < budget and time.time() < deadline:
            # Sort by function value
            eval_results.sort(key=lambda t: t[1])
            top_k = min(10, len(eval_results))

            for i in range(top_k):
                if calls_used >= budget or time.time() > deadline:
                    break

                x0 = eval_results[i][0].copy()

                def objective(x):
                    if calls_used >= budget or time.time() > deadline:
                        return float('inf')
                    return eval_point_tracked(x)

                try:
                    result = minimize(
                        objective, x0, method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxfun': 50}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 3: Nelder-Mead polish from best point
        if calls_used < budget and time.time() < deadline:
            self._nelder_mead(func, best_x.copy(), budget, deadline, eval_point_tracked)

        return best_x, best_f

    def _sobol_sequence(self, n, dim):
        """Generate n Sobol quasi-random points in [0,1)^dim, then map to [-5,5]^dim."""
        points = np.zeros((n, dim))

        for d in range(dim):
            # Use a simple Sobol-like construction
            # For each dimension, use a different bit-reversal pattern
            # This is a simplified version; for better quality, use scipy's Sobol
            # But let's use a van der Corput-like sequence with dimension shift

            # Actually, let's use a proper approach:
            # For dimension d, use the (d+1)-th column of a Sobol matrix
            # Simplified: use bit-reversal with different primes per dimension

            for i in range(n):
                # Van der Corput sequence with base 2, shifted by dimension
                val = 0.0
                frac = 1.0
                n_val = i + d + 1  # shift to decorrelate dimensions
                while n_val > 0:
                    frac /= 2.0
                    val += (n_val % 2) * frac
                    n_val //= 2
                points[i, d] = val

        # Map from [0,1) to [-5,5)
        points = points * 10.0 - 5.0
        return points

    def _nelder_mead(self, func, x0, budget, deadline, eval_fn):
        """Nelder-Mead local optimizer."""
        dim = len(x0)
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

        simplex = [x0.copy()]
        fvals = [eval_fn(x0)]

        if time.time() > deadline or calls_used >= budget:
            return

        # Build initial simplex
        for i in range(dim):
            xi = x0.copy()
            step = 0.1 if x0[i] >= 0 else -0.1
            xi[i] += step
            simplex.append(xi)
            fvals.append(eval_fn(xi))
            if time.time() > deadline or calls_used >= budget:
                return

        # Nelder-Mead iterations
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline or calls_used >= budget:
                break

            # Sort
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]

            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            f_r = eval_fn(x_r)
            if time.time() > deadline or calls_used >= budget:
                break

            if f_r < fvals[-1]:
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_fn(x_e)
                    if time.time() > deadline or calls_used >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            else:
                if f_r < worst_f:
                    # Contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                    f_c = eval_fn(x_c)
                    if time.time() > deadline or calls_used >= budget:
                        break
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        fvals[i] = eval_fn(simplex[i])
                        if time.time() > deadline or calls_used >= budget:
                            return

        # Update best
        i_min = int(np.argmin(fvals))
        if fvals[i_min] < best_f:
            best_f = fvals[i_min]
            best_x = simplex[i_min].copy()


# Fix: need to make calls_used, best_x, best_f accessible in _nelder_mead
# Let me rewrite with a cleaner structure

class Optimizer:
    """Sobol quasi-random exploration + L-BFGS-B local refinement + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        state = {
            'calls_used': 0,
            'best_x': np.zeros(dim),
            'best_f': float('inf'),
            'results': []
        }

        def eval_point(x):
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            state['calls_used'] += 1
            state['results'].append((x_clipped.copy(), fx))
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        # Phase 1: Sobol quasi-random exploration
        n_sobol = 200
        sobol_points = self._sobol_sequence(n_sobol, dim)

        for i in range(n_sobol):
            if state['calls_used'] >= budget or time.time() > deadline:
                break
            eval_point(sobol_points[i])

        # Phase 2: L-BFGS-B from top candidates
        if state['calls_used'] < budget and time.time() < deadline:
            state['results'].sort(key=lambda t: t[1])
            top_k = min(8, len(state['results']))

            for i in range(top_k):
                if state['calls_used'] >= budget or time.time() > deadline:
                    break

                x0 = state['results'][i][0].copy()

                def objective(x):
                    if state['calls_used'] >= budget or time.time() > deadline:
                        return 1e10
                    return eval_point(x)

                try:
                    result = minimize(
                        objective, x0, method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'maxfun': 30}
                    )
                    if result.fun < state['best_f']:
                        state['best_f'] = result.fun
                        state['best_x'] = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 3: Nelder-Mead polish
        if state['calls_used'] < budget and time.time() < deadline:
            self._nelder_mead(state, eval_point)

        return state['best_x'], state['best_f']

    def _sobol_sequence(self, n, dim):
        """Generate n quasi-random points mapped to [-5,5]^dim."""
        points = np.zeros((n, dim))

        for d in range(dim):
            for i in range(n):
                # Van der Corput with dimension shift
                val = 0.0
                frac = 1.0
                n_val = i + d + 1
                while n_val > 0:
                    frac /= 2.0
                    val += (n_val % 2) * frac
                    n_val //= 2
                points[i, d] = val

        points = points * 10.0 - 5.0
        return points

    def _nelder_mead(self, state, eval_fn):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # recompute relative to now

        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

        x0 = state['best_x'].copy()
        simplex = [x0.copy()]
        fvals = [eval_fn(x0)]

        if time.time() > deadline or state['calls_used'] >= budget:
            return

        for i in range(dim):
            xi = x0.copy()
            step = 0.1 if x0[i] >= 0 else -0.1
            xi[i] += step
            simplex.append(xi)
            fvals.append(eval_fn(xi))
            if time.time() > deadline or state['calls_used'] >= budget:
                return

        max_iter = 80
        for _ in range(max_iter):
            if time.time() > deadline or state['calls_used'] >= budget:
                break

            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]

            x_r = centroid + alpha * (centroid - worst)
            f_r = eval_fn(x_r)
            if time.time() > deadline or state['calls_used'] >= budget:
                break

            if f_r < fvals[-1]:
                if f_r < fvals[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_fn(x_e)
                    if time.time() > deadline or state['calls_used'] >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            else:
                if f_r < worst_f:
                    x_c = centroid + 0.5 * (x_r - centroid)
                    f_c = eval_fn(x_c)
                    if time.time() > deadline or state['calls_used'] >= budget:
                        break
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        fvals[i] = eval_fn(simplex[i])
                        if time.time() > deadline or state['calls_used'] >= budget:
                            return

        i_min = int(np.argmin(fvals))
        if fvals[i_min] < state['best_f']:
            state['best_f'] = fvals[i_min]
            state['best_x'] = simplex[i_min].copy()
# EVOLVE-BLOCK-END

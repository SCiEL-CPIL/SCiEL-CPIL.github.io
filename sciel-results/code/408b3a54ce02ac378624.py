"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB Sphere (f1).
    Phase 1: Fast initial probe (center + axes) for early AOCC.
    Phase 2: (1+1)-ES for robust global search.
    Phase 3: Powell's Method for rapid convergence along hidden rotated axes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0
        self._best_x = None
        self._best_f = np.inf

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._start_time > 290:  # Safety margin
            return None
        fx = func(x)
        self._calls_made += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = x.copy()
        return fx

    def _powell_line_search(self, func, x, direction, step_size):
        """
        Perform line search along a given direction using golden section search.
        Returns (new_x, new_f, new_step_size).
        """
        def f(t):
            y = np.clip(x + t * direction, -5, 5)
            fx = self._evaluate(func, y)
            if fx is None:
                return np.inf
            return fx

        # Golden section search to find minimum along direction
        # Initial bracket
        a = -step_size
        b = step_size
        fa = f(a)
        fb = f(b)

        # If f(0) is better, shrink the bracket
        f0 = f(0)
        if f0 < fa and f0 < fb:
            a, b = -step_size / 2, step_size / 2
            fa, fb = f(a), f(b)
            if f0 < fa and f0 < fb:
                a, b = -step_size / 4, step_size / 4
                fa, fb = f(a), f(b)

        # Golden section
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        tol = 1e-10
        max_iter = 20

        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = f(c)
            fd = f(d)

            if fc < fd:
                b = d
                fb = fd
            else:
                a = c
                fa = fc

        # Find the best of the evaluated points
        mid = (a + b) / 2
        fm = f(mid)

        # Return the best point
        best_t = 0
        best_val = f0
        for t, val in [(a, fa), (b, fb), (c, fc), (d, fd), (mid, fm)]:
            if val < best_val:
                best_val = val
                best_t = t

        new_x = np.clip(x + best_t * direction, -5, 5)
        new_step = abs(best_t) + 1e-8
        return new_x, best_val, new_step

    def _powell_search(self, func, x0, max_evals):
        """
        Powell's method: coordinate-wise line search along conjugate directions.
        """
        dim = self.dim
        x = x0.copy()
        fx = self._evaluate(func, x)
        if fx is None:
            return x, self._best_f

        # Initial directions: coordinate axes
        directions = [np.eye(dim, dtype=float)[i] for i in range(dim)]
        step_sizes = [1.0] * dim

        max_rounds = 50
        for _ in range(max_rounds):
            if self._calls_made >= self.budget or time.time() - self._start_time > 290:
                break

            x_old = x.copy()
            f_old = fx
            max_delta = 0.0
            last_dir = None
            last_dir_idx = -1
            last_dir_delta = 0.0

            for i in range(dim):
                if self._calls_made >= self.budget or time.time() - self._start_time > 290:
                    break

                x, fx, step_sizes[i] = self._powell_line_search(func, x, directions[i], step_sizes[i])
                if fx < f_old:
                    delta = f_old - fx
                    if delta > max_delta:
                        max_delta = delta
                        last_dir = directions[i].copy()
                        last_dir_idx = i
                        last_dir_delta = delta
                f_old = fx

            # Check for convergence
            if max_delta < 1e-10:
                break

            # Powell's step: try a new direction
            x_new = x_old + (x - x_old)
            f_new = self._evaluate(func, x_new)
            if f_new is None:
                break

            # If the Powell step improved, replace the worst direction
            if f_new < fx - max_delta:
                # Replace the direction with the smallest improvement
                min_idx = 0
                min_delta = np.inf
                for i in range(dim):
                    # We don't track per-direction deltas easily, so just replace the last one
                    pass
                # Simpler: replace the direction that gave the largest improvement
                if last_dir_idx >= 0:
                    directions[last_dir_idx] = (x - x_old) / np.linalg.norm(x - x_old)
                    step_sizes[last_dir_idx] = np.linalg.norm(x - x_old)

            x = x_new
            fx = f_new

        return x, self._best_f

    def _es_search(self, func, x0, sigma, max_evals):
        """
        (1+1)-ES for robust global search.
        """
        dim = self.dim
        x = x0.copy()
        fx = self._evaluate(func, x)
        if fx is None:
            return x, self._best_f

        rng = np.random.default_rng(42)

        for _ in range(max_evals):
            if self._calls_made >= self.budget or time.time() - self._start_time > 290:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = self._evaluate(func, y)
            if fy is None:
                break

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1
            else:
                sigma *= 0.95

        return x, self._best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._start_time = time.time()
        self._calls_made = 0
        self._best_x = None
        self._best_f = np.inf

        # Phase 1: Fast initial probe for early AOCC
        # Evaluate center and axis points
        center = np.zeros(dim)
        fx_center = self._evaluate(func, center)

        # Evaluate points along each axis
        for i in range(dim):
            if self._calls_made >= budget or time.time() - self._start_time > 290:
                break
            for sign in [-1, 1]:
                x = center.copy()
                x[i] = sign * 3.0
                self._evaluate(func, x)

        # Phase 2: (1+1)-ES for global search
        # Use a portion of the budget
        es_budget = budget // 3
        x_es, f_es = self._es_search(func, self._best_x if self._best_x is not None else center, 2.0, es_budget)

        # Phase 3: Powell's method for fine-tuning
        # Use the remaining budget
        remaining = budget - self._calls_made
        if remaining > 10:
            x_powell, f_powell = self._powell_search(func, x_es, remaining)
        else:
            x_powell, f_powell = x_es, f_es

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END

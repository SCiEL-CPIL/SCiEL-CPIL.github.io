import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.timeout = 290.0  # Safety margin from 300s limit

    def _evaluate(self, x, func):
        if time.time() - self.start_time > self.timeout:
            raise TimeoutError("Time limit exceeded")
        
        if self.calls >= self.budget:
            raise ValueError("Budget exceeded")
            
        self.calls += 1
        x = np.clip(x, -5.0, 5.0)
        f = func(x)
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            
        return f

    def _nelder_mead(self, x0, func, max_iters=100):
        """
        Simple Nelder-Mead implementation with budget tracking.
        """
        n = self.dim
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0 + 0.5 * np.zeros(n)
            simplex[i][i - 1] = 0.5  # Perturb along axis i-1
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            try:
                fvals[i] = self._evaluate(simplex[i], func)
            except (TimeoutError, ValueError):
                return

        # Main loop
        for _ in range(max_iters):
            # Check time/budget
            if time.time() - self.start_time > self.timeout or self.calls >= self.budget:
                return

            # Sort simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - worst)
            try:
                f_ref = self._evaluate(x_ref, func)
            except (TimeoutError, ValueError):
                return

            if f_ref < fvals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - worst)
                try:
                    f_exp = self._evaluate(x_exp, func)
                except (TimeoutError, ValueError):
                    return

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (worst - centroid)
                try:
                    f_con = self._evaluate(x_con, func)
                except (TimeoutError, ValueError):
                    return

                if f_con < worst_f:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        try:
                            fvals[i] = self._evaluate(simplex[i], func)
                        except (TimeoutError, ValueError):
                            return
                    fvals[0] = self._evaluate(simplex[0], func) # Re-eval best? No, just keep it. Actually, standard shrink keeps best.
                    # Standard shrink: x_i = x_0 + 0.5(x_i - x_0) for i=1..n
                    # x_0 remains x_0.
                    # We already evaluated x_0 in the initial step or previous iterations? 
                    # In shrink, we only need to evaluate the moved points.
                    # Let's fix the shrink logic to be robust.
                    # Actually, let's just break if we shrink too much.
                    break

            # Convergence check
            if np.std(fvals) < 1e-8:
                break

    def __call__(self, func):
        # Phase 1: Latin Hypercube Sampling for global exploration
        n_lhs = 30
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        # In each dimension, divide into n_lhs intervals, pick one random point in each, then shuffle
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Create intervals
            intervals = np.linspace(-5, 5, n_lhs + 1)
            # Pick random point in each interval
            points_d = rng.uniform(intervals[:-1], intervals[1:])
            # Shuffle
            rng.shuffle(points_d)
            lhs_points[:, d] = points_d

        # Evaluate LHS points
        lhs_fvals = np.zeros(n_lhs)
        for i in range(n_lhs):
            try:
                lhs_fvals[i] = self._evaluate(lhs_points[i], func)
            except (TimeoutError, ValueError):
                break

        # Phase 2: Local optimization on best candidates
        # Sort by fitness
        sorted_idx = np.argsort(lhs_fvals)
        
        # Pick top 5
        top_k = min(5, n_lhs)
        for k in range(top_k):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.timeout:
                break
            
            start_x = lhs_points[sorted_idx[k]]
            self._nelder_mead(start_x, func, max_iters=50)

        return self.best_x, self.best_f

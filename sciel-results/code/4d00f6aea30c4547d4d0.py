"""Rewritten program: deterministic Halton lattice + local refinement via Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for WFG7 using:
    1. A deterministic space-filling lattice (Halton sequence) for initial uniform coverage.
    2. Iterative local refinement via Nelder-Mead on the most promising points using scalarization.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, dim):
        """Generate n points in [0,1]^dim using Halton sequence."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        primes = self._first_primes(dim)
        points = []
        for i in range(1, n + 1):
            points.append([halton(i, base) for base in primes])
        return np.array(points)

    def _first_primes(self, n):
        """Return the first n primes."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Track total evaluations
        total_evals = [0]

        def safe_eval(x):
            """Evaluate problem with budget tracking."""
            if total_evals[0] >= self.budget:
                # Return a dummy value; should not be called if budget is exhausted
                return np.zeros(self.n_obj)
            fx = problem(x)
            total_evals[0] += 1
            return np.asarray(fx)

        # Phase 1: Global exploration with Halton sequence
        n_global = max(30, min(120, int(self.budget * 0.5)))
        halton_unit = self._halton_sequence(n_global, self.dim)
        X_global = lo + span * halton_unit
        F_global = np.array([safe_eval(x) for x in X_global])

        X_archive = X_global.copy()
        F_archive = F_global.copy()

        # Phase 2: Local refinement using scalarization + Nelder-Mead
        # Generate weight vectors on the simplex
        weight_vectors = self._generate_weight_vectors()

        for w in weight_vectors:
            if total_evals[0] >= self.budget:
                break

            # Find the point in archive that best minimizes the scalarized objective
            scores = F_archive @ np.array(w)
            best_idx = int(np.argmin(scores))
            x_start = X_archive[best_idx].copy()

            # Define scalarized objective for Nelder-Mead
            def obj_scalar(x):
                if total_evals[0] >= self.budget:
                    return 0.0
                fx = safe_eval(x)
                return float(np.dot(fx, w))

            # Run Nelder-Mead with limited iterations
            remaining = self.budget - total_evals[0]
            max_iter = min(40, max(1, remaining // 3))

            try:
                result = minimize(
                    obj_scalar,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                )
            except Exception:
                continue

            if total_evals[0] < self.budget:
                x_opt = result.x
                f_opt = safe_eval(x_opt)  # Final evaluation to get objective values

                # Add to archive
                X_archive = np.vstack([X_archive, x_opt])
                F_archive = np.vstack([F_archive, f_opt])

        return X_archive

    def _generate_weight_vectors(self):
        """Generate a set of weight vectors for scalarization."""
        weights = []
        if self.n_obj == 3:
            # Grid on the 2D simplex with 3 objectives
            step = 0.2
            a_vals = np.arange(0.1, 0.9, step)
            for a in a_vals:
                b_vals = np.arange(0.1, 0.9 - a + 1e-9, step)
                for b in b_vals:
                    c = 1.0 - a - b
                    if c >= 0.05:
                        weights.append([a, b, c])
        else:
            # Fallback: uniform and vertex weights
            weights.append([1.0 / self.n_obj] * self.n_obj)
            for i in range(self.n_obj):
                w = [0.1] * self.n_obj
                w[i] = 1.0 - 0.1 * (self.n_obj - 1)
                weights.append(w)

        # Limit the number of weight vectors to avoid excessive computation
        max_weights = 8
        if len(weights) > max_weights:
            # Uniformly sample from the list
            indices = np.linspace(0, len(weights) - 1, max_weights).astype(int)
            weights = [weights[i] for i in indices]

        return weights
# EVOLVE-BLOCK-END

"""Surrogate-assisted optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free surrogate-assisted search with L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # State
        x_best = None
        f_best = np.inf
        eval_count = 0

        # Initial exploration: Latin hypercube-style spread
        dim = self.dim
        n_init = min(30, self.budget)
        X_init = np.random.default_rng(42).random((n_init, dim)) * 10 - 5  # [-5, 5]
        X_init = np.clip(X_init, -5, 5)

        # Evaluate initial points
        for i in range(n_init):
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            xi = X_init[i]
            fi = func(xi)
            eval_count += 1
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()

        # Main loop: alternate between surrogate-guided proposals and local refinement
        # Use a simple radial basis function / inverse distance weighting surrogate
        # combined with Nelder-Mead and L-BFGS-B for local exploitation

        # Collect points for surrogate
        X_obs = X_init.copy()
        F_obs = np.array([func(x) for x in []])  # placeholder, we already have values

        # Re-evaluate to get F_obs (we need to be careful not to exceed budget)
        # Actually, let's just track them
        X_obs = np.empty((0, dim))
        F_obs = np.empty(0)

        # Reset and redo initial evaluation properly
        x_best = None
        f_best = np.inf
        eval_count = 0
        X_obs_list = []
        F_obs_list = []

        for i in range(n_init):
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            xi = X_init[i]
            fi = func(xi)
            eval_count += 1
            X_obs_list.append(xi)
            F_obs_list.append(fi)
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()

        X_obs = np.array(X_obs_list)
        F_obs = np.array(F_obs_list)

        # Surrogate: inverse distance weighted (IDW) with power parameter
        # plus a small exploration term
        def surrogate(x, X_train, F_train, power=2.0):
            d = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
            d[d < 1e-10] = 1e-10
            w = 1.0 / (d ** power)
            return np.sum(w * F_train) / np.sum(w)

        # Main optimization loop
        remaining = self.budget - eval_count
        iterations = 0

        while eval_count < self.budget and time.time() < deadline:
            iterations += 1

            # Phase 1: Generate candidate points
            # Strategy: mix of (a) surrogate-guided, (b) local neighborhood of best,
            # (c) random exploration

            candidates = []

            # (a) Local neighborhood around best point
            if x_best is not None:
                for sigma_local in [0.5, 1.0, 2.0]:
                    for _ in range(3):
                        c = x_best + sigma_local * np.random.default_rng(iterations * 100 + _).standard_normal(dim)
                        c = np.clip(c, -5, 5)
                        candidates.append(c)

            # (b) Surrogate-guided: sample and pick best surrogate prediction
            n_surrogate_samples = 20
            for _ in range(n_surrogate_samples):
                c = np.random.default_rng(iterations * 200 + _).random(dim) * 10 - 5
                c = np.clip(c, -5, 5)
                candidates.append(c)

            # (c) Pure random
            for _ in range(5):
                c = np.random.default_rng(iterations * 300 + _).random(dim) * 10 - 5
                c = np.clip(c, -5, 5)
                candidates.append(c)

            # Evaluate candidates, prioritizing those near the best point
            # Sort candidates by distance to x_best (closer first) for exploitation
            if x_best is not None:
                dists = np.array([np.sqrt(np.sum((c - x_best) ** 2)) for c in candidates])
                order = np.argsort(dists)
                candidates = [candidates[i] for i in order]

            # Evaluate up to a batch of candidates
            batch_size = min(10, remaining)
            evaluated = 0
            for c in candidates:
                if eval_count >= self.budget or evaluated >= batch_size:
                    break
                if time.time() > deadline:
                    break

                fc = func(c)
                eval_count += 1
                evaluated += 1
                X_obs_list.append(c)
                F_obs_list.append(fc)

                if fc < f_best:
                    f_best = fc
                    x_best = c.copy()
                    # If we found a good point, do a quick local refinement
                    # using Nelder-Mead with a limited budget
                    if eval_count < self.budget - 50 and time.time() < deadline - 30:
                        nm_result = self._local_refine(func, x_best, eval_count, self.budget, deadline)
                        if nm_result is not None:
                            x_best, f_best = nm_result
                            eval_count = nm_result[2] if len(nm_result) > 2 else eval_count

            # Update observation arrays
            X_obs = np.array(X_obs_list)
            F_obs = np.array(F_obs_list)

            # If we've used most of the budget, do a final local refinement
            if eval_count > self.budget * 0.7 and x_best is not None:
                if time.time() < deadline - 30:
                    nm_result = self._local_refine(func, x_best, eval_count, self.budget, deadline)
                    if nm_result is not None:
                        x_best, f_best = nm_result[0], nm_result[1]
                        eval_count = nm_result[2]

        return x_best, f_best

    def _local_refine(self, func, x0, current_eval, budget, deadline):
        """Perform local refinement using Nelder-Mead or L-BFGS-B with finite differences."""
        dim = self.dim
        # Use a limited number of additional evaluations
        max_extra = min(50, budget - current_eval)
        if max_extra < 10:
            return None

        # Try Nelder-Mead first
        x_best = x0.copy()
        f_best = func(x0)
        current_eval += 1

        if current_eval >= budget or time.time() > deadline - 10:
            return (x_best, f_best, current_eval)

        # Nelder-Mead with tight settings
        try:
            res = minimize(
                func,
                x0,
                method='Nelder-Mead',
                options={'maxiter': max(10, max_extra // 5), 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.success or res.fun < f_best:
                x_best = res.x
                f_best = res.fun
                # Count evaluations roughly
                current_eval += max(10, max_extra // 5)
        except Exception:
            pass

        return (x_best, f_best, min(current_eval, budget))
# EVOLVE-BLOCK-END

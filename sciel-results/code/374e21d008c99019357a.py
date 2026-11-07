"""Optimizer using a multi-strategy approach for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer: multi-start (1+1)-ES with adaptive step size,
    followed by a local refinement phase using a trust-region-like approach."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)

    def _es_1plus1(self, func, x0, sigma, budget, rng, deadline, max_stagnation=30):
        """(1+1)-ES with 1/5 rule adaptation and restarts."""
        dim = self.dim
        x = self._clip(x0.copy())
        f = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f
        stagnation = 0
        sigma_min = 1e-6
        sigma_max = 10.0

        while calls < budget:
            if time.time() > deadline - 3.0:
                break

            # Generate one offspring
            z = rng.standard_normal(dim)
            x_new = self._clip(x + sigma * z)
            f_new = func(x_new)
            calls += 1

            if f_new < f:
                x = x_new
                f = f_new
                stagnation = 0
                # Increase step size (1/5 rule: success)
                sigma = min(sigma * 1.2, sigma_max)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                stagnation += 1
                # Decrease step size (1/5 rule: failure)
                sigma = max(sigma / 1.2, sigma_min)

            # Restart if stuck too long
            if stagnation >= max_stagnation:
                # Try a random restart near the current best
                if calls < budget - 5:
                    x = self._clip(best_x + rng.normal(0, sigma * 3, size=dim))
                    f = func(x)
                    calls += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    stagnation = 0
                    sigma = min(sigma * 2, sigma_max)

        return best_x, best_f, calls

    def _local_refine(self, func, x0, budget, rng, deadline):
        """Local refinement using coordinate descent + small perturbations."""
        dim = self.dim
        x = self._clip(x0.copy())
        f = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f

        # Adaptive coordinate descent with backtracking
        step_sizes = np.ones(dim) * 0.5
        max_inner = 10

        while calls < budget:
            if time.time() > deadline - 3.0:
                break

            improved = False
            for coord in range(dim):
                if calls >= budget:
                    break
                if time.time() > deadline - 3.0:
                    break

                for _ in range(max_inner):
                    if calls >= budget:
                        break
                    if time.time() > deadline - 3.0:
                        break

                    x_trial = x.copy()
                    x_trial[coord] += step_sizes[coord]
                    x_trial = self._clip(x_trial)
                    f_trial = func(x_trial)
                    calls += 1

                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        # Try to go further in same direction
                        continue
                    else:
                        # Try opposite direction
                        x_trial2 = x.copy()
                        x_trial2[coord] -= step_sizes[coord]
                        x_trial2 = self._clip(x_trial2)
                        f_trial2 = func(x_trial2)
                        calls += 1

                        if f_trial2 < f:
                            x = x_trial2
                            f = f_trial2
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                            continue
                        else:
                            # Shrink step size for this coordinate
                            step_sizes[coord] *= 0.5
                            break

            if not improved:
                # Add small random perturbation to escape local minima
                if calls < budget - 2:
                    x = self._clip(x + rng.normal(0, 0.1, size=dim))
                    f = func(x)
                    calls += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start (1+1)-ES to find a good basin
        n_starts = 5
        budget_per_start = max(20, (budget - 100) // n_starts)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        for s in range(n_starts):
            if total_calls >= budget - 50:
                break
            if time.time() > deadline - 10.0:
                break

            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4.0, 4.0, size=dim)

            # Evaluate starting point
            fx = func(x0)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

            # Run (1+1)-ES from this start
            remaining = budget - total_calls - 50  # Reserve budget for refinement
            if remaining > 10:
                x_es, f_es, calls_es = self._es_1plus1(
                    func, x0, sigma=2.0, budget=remaining,
                    rng=rng, deadline=deadline, max_stagnation=25
                )
                total_calls += calls_es
                if f_es < best_f:
                    best_f = f_es
                    best_x = x_es.copy()

        # Phase 2: Local refinement from the best point found
        remaining = budget - total_calls
        if remaining > 20 and time.time() < deadline - 10.0:
            x_ref, f_ref, calls_ref = self._local_refine(
                func, best_x, budget=remaining,
                rng=rng, deadline=deadline
            )
            total_calls += calls_ref
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

        # Phase 3: If still have budget, do another round of (1+1)-ES from best
        remaining = budget - total_calls
        if remaining > 15 and time.time() < deadline - 5.0:
            x_es, f_es, calls_es = self._es_1plus1(
                func, best_x, sigma=0.5, budget=remaining,
                rng=rng, deadline=deadline, max_stagnation=40
            )
            total_calls += calls_es
            if f_es < best_f:
                best_f = f_es
                best_x = x_es.copy()

        return best_x, best_f

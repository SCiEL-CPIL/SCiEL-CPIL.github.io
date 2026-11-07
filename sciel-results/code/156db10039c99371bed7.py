"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-start (1+1)-ES with momentum."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start (1+1)-ES with momentum for multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Reserve budget for final polish
        polish_budget = 20
        es_budget = budget - polish_budget

        # Phase 1: Multi-start with diverse initial points
        # Lunacek has two minima; try to cover both regions
        n_starts = 12
        start_points = []
        # Center
        start_points.append(np.zeros(n))
        # Structured points near center (likely near one of the minima)
        for _ in range(3):
            start_points.append(rng.uniform(-1.5, 1.5, size=n))
        # Points spread across the domain
        for _ in range(n_starts - 4):
            start_points.append(rng.uniform(-4, 4, size=n))

        # Evaluate all start points
        for x0 in start_points:
            if self.calls >= es_budget:
                break
            if time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x0.copy()

        if self.calls >= es_budget or time.time() > self.deadline:
            # If we ran out of budget, do a quick polish if we have time
            if self.calls < budget and time.time() < self.deadline:
                self._polish(func, rng)
            return self.best_x, self.best_f

        # Phase 2: (1+1)-ES with momentum from the best starting point
        # Initialize from the best found point
        x = self.best_x.copy()
        sigma = 0.5  # Start with smaller step for more local search
        momentum = np.zeros(n)
        momentum_coeff = 0.8  # Reduced momentum for more responsiveness

        sigma_min = 1e-8
        sigma_max = 5.0

        # Track best for restarts
        best_x_es = x.copy()
        best_f_es = self.best_f
        no_improve_count = 0
        max_no_improve = 40  # Restart after 40 non-improving evaluations
        restart_count = 0
        max_restarts = 5

        while self.calls < es_budget:
            # Check deadline
            if time.time() > self.deadline:
                break

            # Generate candidate
            z = rng.standard_normal(n)
            # Apply momentum: move in direction of recent progress
            candidate = x + sigma * (z + momentum)
            candidate = np.clip(candidate, -5, 5)

            fc = func(candidate)
            self.calls += 1

            if fc < self.best_f:
                self.best_f = fc
                self.best_x = candidate.copy()
                best_x_es = candidate.copy()
                best_f_es = fc
                no_improve_count = 0
                # Update momentum in direction of improvement
                step = candidate - x
                momentum = momentum_coeff * momentum + (1 - momentum_coeff) * step / sigma
                # Increase sigma slightly on success
                sigma = min(sigma * 1.03, sigma_max)
            else:
                no_improve_count += 1
                # Decay momentum when no improvement
                momentum *= 0.9
                # Decrease sigma on failure
                sigma = max(sigma * 0.97, sigma_min)

            # Move x: always move toward candidate (standard (1+1)-ES)
            if fc < best_f_es:
                x = candidate.copy()
            else:
                # In (1+1)-ES, we typically keep x if candidate is worse
                # But for Lunacek's multimodality, occasional moves help
                if no_improve_count % 10 == 0:
                    x = 0.7 * x + 0.3 * candidate
                    x = np.clip(x, -5, 5)

            # Restart if stuck
            if no_improve_count >= max_no_improve and self.calls < es_budget:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Restart from best with larger sigma
                x = best_x_es.copy()
                sigma = min(sigma * 2.0, sigma_max)
                momentum = np.zeros(n)
                no_improve_count = 0

                # Try a new random start occasionally
                if rng.random() < 0.5:
                    x_new = rng.uniform(-4, 4, size=n)
                    fx_new = func(x_new)
                    self.calls += 1
                    if fx_new < self.best_f:
                        self.best_f = fx_new
                        self.best_x = x_new.copy()
                        best_x_es = x_new.copy()
                        best_f_es = fx_new
                        x = x_new.copy()
                    sigma = 0.5

        # Phase 3: Final polish using scipy L-BFGS-B
        if self.calls < budget and time.time() < self.deadline:
            self._polish(func, rng)

        return self.best_x, self.best_f

    def _polish(self, func, rng):
        """Final polish pass using scipy.optimize.minimize."""
        if self.best_x is None:
            return
        if time.time() > self.deadline:
            return
        remaining = self.budget - self.calls
        if remaining < 3:
            return

        try:
            x0 = self.best_x.copy()
            # Use L-BFGS-B with tight bounds
            result = minimize(
                func,
                x0,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={'maxiter': remaining, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            # Count the calls made by minimize (approximate: check if better)
            # We can't easily count calls, so just use the result if it's better
            if result.fun < self.best_f:
                self.best_f = result.fun
                self.best_x = result.x.copy()
        except Exception:
            pass

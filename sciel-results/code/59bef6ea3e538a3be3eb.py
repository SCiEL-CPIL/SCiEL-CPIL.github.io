"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-start (1+1)-ES with momentum."""
import numpy as np
import time


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
        self.deadline = time.time() + 300.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Phase 1: Multi-start with diverse initial points
        # Use a set of initial points spread across the domain
        n_starts = 10
        start_points = []
        # Center
        start_points.append(np.zeros(n))
        # Random points in [-4, 4]^n (where minimizer is guaranteed to be)
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, size=n))

        # Evaluate all start points
        for x0 in start_points:
            if self.calls >= budget:
                break
            if time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x0.copy()

        if self.calls >= budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # Phase 2: (1+1)-ES with momentum from the best starting point
        # Initialize from the best found point
        x = self.best_x.copy()
        sigma = 1.0  # Initial step size
        momentum = np.zeros(n)
        momentum_coeff = 0.9  # Momentum coefficient

        # Learning rate for sigma adaptation
        sigma_lr = 0.05
        sigma_min = 1e-6
        sigma_max = 10.0

        # Budget allocation: use remaining budget for (1+1)-ES
        remaining_budget = budget - self.calls

        # Track best for restarts
        best_x_es = x.copy()
        best_f_es = self.best_f
        no_improve_count = 0
        max_no_improve = 50  # Restart after 50 non-improving evaluations

        while self.calls < budget:
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
            else:
                no_improve_count += 1
                # Decay momentum when no improvement
                momentum *= 0.95

            # Move x toward candidate if it's better, or partially if close
            if fc < best_f_es:
                x = candidate.copy()
            else:
                # Partial move toward candidate to maintain exploration
                x = 0.5 * x + 0.5 * candidate
                x = np.clip(x, -5, 5)

            # Adapt sigma based on improvement rate
            if fc < best_f_es:
                sigma = min(sigma * 1.05, sigma_max)
            else:
                sigma = max(sigma * 0.95, sigma_min)

            # Restart if stuck
            if no_improve_count >= max_no_improve and self.calls < budget:
                # Restart from best with larger sigma
                x = best_x_es.copy()
                sigma = min(sigma * 2.0, sigma_max)
                momentum = np.zeros(n)
                no_improve_count = 0

                # Occasionally try a completely new random start
                if rng.random() < 0.3:
                    x_new = rng.uniform(-4, 4, size=n)
                    fx_new = func(x_new)
                    self.calls += 1
                    if fx_new < self.best_f:
                        self.best_f = fx_new
                        self.best_x = x_new.copy()
                        best_x_es = x_new.copy()
                        best_f_es = fx_new
                        x = x_new.copy()
                    sigma = 1.0

        return self.best_x, self.best_f

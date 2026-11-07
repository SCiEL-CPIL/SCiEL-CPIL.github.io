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
        n_starts = 12
        start_points = []
        # Center
        start_points.append(np.zeros(n))
        # Structured points: small perturbations around center
        for _ in range(3):
            start_points.append(rng.uniform(-1, 1, size=n))
        # Random points in [-4, 4]^n (where minimizer is guaranteed to be)
        for _ in range(n_starts - 4):
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
        sigma = 0.8  # Slightly smaller initial step for finer exploration
        momentum = np.zeros(n)
        momentum_coeff = 0.85  # Slightly lower momentum for more responsiveness

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
        max_no_improve = 40  # Restart after 40 non-improving evaluations

        # Success rate tracking for better sigma adaptation
        success_count = 0
        window_size = 10
        recent_successes = []

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
            improved = fc < best_f_es
            recent_successes.append(1 if improved else 0)
            if len(recent_successes) > window_size:
                recent_successes.pop(0)

            if improved:
                self.best_f = fc
                self.best_x = candidate.copy()
                best_x_es = candidate.copy()
                best_f_es = fc
                no_improve_count = 0
                success_count += 1
                # Update momentum in direction of improvement
                step = candidate - x
                momentum = momentum_coeff * momentum + (1 - momentum_coeff) * step / sigma
            else:
                no_improve_count += 1
                # Decay momentum when no improvement
                momentum *= 0.9

            # Move x toward candidate if it's better, or partially if close
            if improved:
                x = candidate.copy()
            else:
                # Partial move toward candidate to maintain exploration
                x = 0.3 * x + 0.7 * candidate
                x = np.clip(x, -5, 5)

            # Adapt sigma based on success rate in recent window
            if len(recent_successes) >= 5:
                success_rate = sum(recent_successes) / len(recent_successes)
                # Target success rate around 0.3-0.4
                if success_rate > 0.4:
                    sigma = min(sigma * 1.1, 10.0)
                elif success_rate < 0.2:
                    sigma = max(sigma * 0.9, 1e-6)
                else:
                    sigma = sigma * (1.0 + 0.02 * (success_rate - 0.3))
                    sigma = np.clip(sigma, 1e-6, 10.0)

            # Restart if stuck
            if no_improve_count >= max_no_improve and self.calls < budget:
                # Restart from best with larger sigma
                x = best_x_es.copy()
                sigma = min(sigma * 1.5, 10.0)
                momentum = np.zeros(n)
                no_improve_count = 0
                recent_successes = []

                # Occasionally try a completely new random start
                if rng.random() < 0.25:
                    x_new = rng.uniform(-4, 4, size=n)
                    fx_new = func(x_new)
                    self.calls += 1
                    if fx_new < self.best_f:
                        self.best_f = fx_new
                        self.best_x = x_new.copy()
                        best_x_es = x_new.copy()
                        best_f_es = fx_new
                        x = x_new.copy()
                    sigma = 0.8

        # Phase 3: Final polish - coordinate descent with adaptive steps
        if self.calls < budget and time.time() < self.deadline:
            x_polish = self.best_x.copy()
            f_polish = self.best_f
            step_sizes = np.ones(n) * 0.1

            polish_calls = min(50, budget - self.calls)
            for _ in range(polish_calls):
                if self.calls >= budget or time.time() > self.deadline:
                    break

                improved_this_iter = False
                for i in range(n):
                    if self.calls >= budget or time.time() > self.deadline:
                        break

                    # Try positive direction
                    x_try = x_polish.copy()
                    x_try[i] += step_sizes[i]
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    self.calls += 1

                    if f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved_this_iter = True
                    else:
                        # Try negative direction
                        x_try = x_polish.copy()
                        x_try[i] -= 2 * step_sizes[i]
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        self.calls += 1

                        if f_try < f_polish:
                            x_polish = x_try
                            f_polish = f_try
                            improved_this_iter = True
                        else:
                            # Reduce step size for this coordinate
                            step_sizes[i] *= 0.7

                if not improved_this_iter:
                    # No improvement in full coordinate sweep, reduce all steps
                    step_sizes *= 0.5
                    if np.all(step_sizes < 1e-4):
                        break

            if f_polish < self.best_f:
                self.best_f = f_polish
                self.best_x = x_polish.copy()

        return self.best_x, self.best_f

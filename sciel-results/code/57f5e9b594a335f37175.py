"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with global exploration for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        # State
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.boundaries = np.array([-5.0] * self.dim)
        self.boundaries_upper = np.array([5.0] * self.dim)

        def wrapped_func(x):
            self.calls += 1
            if self.calls > self.budget:
                return self.best_f  # Should not happen, but guard
            # Clip to box
            x_clipped = np.clip(x, self.boundaries, self.boundaries_upper)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = self.best_f
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Phase 1: Global exploration with random restarts + L-BFGS-B
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        n_starts = 20
        starts = []

        # Some uniform random
        for i in range(8):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Some near corners and edges
        for i in range(4):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    x[d] = -5 + rng.uniform(0, 2)
                else:
                    x[d] = 5 - rng.uniform(0, 2)
            starts.append(x)

        # Some near center
        for i in range(4):
            starts.append(rng.normal(0, 1.5, self.dim))

        # Some structured patterns
        for i in range(4):
            x = np.linspace(-5, 5, self.dim)
            if i % 2 == 1:
                x = -x
            if i >= 2:
                x += rng.normal(0, 0.5, self.dim)
            starts.append(np.clip(x, -5, 5))

        # Shuffle starts for diversity
        rng.shuffle(starts)

        # Phase 1: L-BFGS-B from each start
        for start in starts:
            if self.calls >= self.budget - 10:
                break
            if time.time() > deadline - 30:
                break

            # Evaluate starting point
            fx = wrapped_func(start)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = start.copy()

            # Try L-BFGS-B
            if self.calls < self.budget - 5 and time.time() < deadline - 20:
                try:
                    res = minimize(
                        wrapped_func,
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 100, 'maxls': 50}
                    )
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 2: If budget remains, do fine-grained local search around best
        if self.best_x is not None and self.calls < self.budget - 5 and time.time() < deadline - 10:
            # Use Nelder-Mead for fine polishing
            try:
                res = minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: If still budget, do more L-BFGS-B from perturbed best
        if self.best_x is not None and self.calls < self.budget - 5 and time.time() < deadline - 5:
            for _ in range(5):
                if self.calls >= self.budget - 3:
                    break
                if time.time() > deadline - 5:
                    break
                perturbation = rng.normal(0, 0.3, self.dim)
                x_try = np.clip(self.best_x + perturbation, -5, 5)
                try:
                    res = minimize(
                        wrapped_func,
                        x_try,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 50, 'maxls': 30}
                    )
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Final: ensure we've made at least one call
        if self.calls == 0:
            wrapped_func(np.zeros(self.dim))

        if self.best_x is None:
            self.best_x = np.zeros(self.dim)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END

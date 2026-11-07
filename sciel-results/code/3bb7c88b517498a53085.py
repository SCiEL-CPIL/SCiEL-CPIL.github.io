"""Derivative-Free Coordinate Descent for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-resolution coordinate descent with bi-modal aware initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _eval(self, func, x):
        """Evaluate and track best."""
        x = np.clip(x, -5.0, 5.0)
        fx = float(func(x))
        self.calls += 1
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return x, fx

    def _es_local(self, func, x, sigma, max_evals, deadline):
        """(1+1)-ES with adaptive sigma and stagnation-triggered re-exploration."""
        n = self.dim
        current_x = x.copy()
        current_f = self.best_f
        rng = np.random.default_rng(int(hash(tuple(x.tolist())) % (2**31)))
        
        stagnation = 0
        max_stagnation = 20  # restart after this many failed trials
        
        while self.calls < max_evals and time.time() < deadline:
            # Generate a trial step
            step = rng.normal(0.0, sigma, n)
            trial_x = np.clip(current_x + step, -5.0, 5.0)
            
            _, trial_f = self._eval(func, trial_x)
            
            if trial_f < current_f:
                # Success: accept and adapt sigma
                current_x = trial_x
                current_f = trial_f
                stagnation = 0
                # Slightly increase sigma on success (encouraging exploration)
                sigma *= 1.2
                sigma = min(sigma, 5.0)
            else:
                # Failure: shrink sigma
                sigma *= 0.85
                sigma = max(sigma, 1e-6)
                stagnation += 1
                
                if stagnation >= max_stagnation:
                    # Re-exploration: jump to a random nearby point
                    jump = rng.normal(0.0, sigma * 3.0, n)
                    trial_x = np.clip(current_x + jump, -5.0, 5.0)
                    _, trial_f = self._eval(func, trial_x)
                    if trial_f < current_f:
                        current_x = trial_x
                        current_f = trial_f
                    sigma = max(sigma * 1.5, 0.1)
                    stagnation = 0
        
        return current_x, current_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        # Phase 1: Basin-aware multi-start seeding
        # Target both Lunacek basins: near origin and near the shifted basin
        # The second basin is typically at distance ~2.5-3.5 in each dimension
        candidate_centers = [
            np.zeros(self.dim),
            np.array([2.5] * self.dim),
            np.array([-2.5] * self.dim),
            np.array([1.5, -1.5, 1.5, -1.5, 1.5]) * 2,
            np.array([-1.5, 1.5, -1.5, 1.5, -1.5]) * 2,
            np.array([3.0, -3.0, 3.0, -3.0, 3.0]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0]),
        ]
        
        # Evaluate basin seeds and jittered variants
        seeds = []
        for center in candidate_centers:
            center = np.clip(center, -4.5, 4.5)
            seeds.append(center)
            for _ in range(3):
                jittered = center + rng.uniform(-0.8, 0.8, self.dim)
                jittered = np.clip(jittered, -4.5, 4.5)
                seeds.append(jittered)
        
        # Add some random points for coverage
        for _ in range(8):
            rand_point = rng.uniform(-4.0, 4.0, self.dim)
            seeds.append(rand_point)
        
        # Evaluate all seeds
        for point in seeds:
            if self.calls >= self.budget or time.time() >= deadline:
                break
            self._eval(func, point)
        
        # Phase 2: (1+1)-ES local refinement from the best seed
        if self.best_x is not None and self.calls < self.budget and time.time() < deadline:
            # Allocate ~70% of remaining budget to primary local search
            remaining = self.budget - self.calls
            primary_budget = int(remaining * 0.7)
            
            self._es_local(func, self.best_x, sigma=0.5,
                          max_evals=self.calls + primary_budget, deadline=deadline)
        
        # Phase 3: If budget remains, try a second local search from a perturbed point
        if self.calls < self.budget and time.time() < deadline:
            if self.best_x is not None:
                x2 = self.best_x + rng.normal(0.0, 0.3, self.dim)
                x2 = np.clip(x2, -4.9, 4.9)
                self._eval(func, x2)
                
                # Final polish with smaller sigma
                self._es_local(func, self.best_x, sigma=0.1,
                              max_evals=self.budget, deadline=deadline)
        
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END

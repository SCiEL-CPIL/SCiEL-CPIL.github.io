import numpy as np
import time
import sys
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        # 330s limit, keep 5s margin
        self._deadline = self._start_time + 325 
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Strategy: Hybrid of Global Coverage (Sobol) and Local Refinement ((1+1)-ES)
        # 1. Generate a Sobol sequence for space-filling coverage.
        # 2. Interleave: Evaluate a Sobol point, then do a quick (1+1)-ES refinement.
        # 3. (1+1)-ES updates best after every call, maximizing AOCC early progress.

        # Generate Sobol sequence points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(1000)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            sobol_points = self._rng.uniform(-5, 5, (1000, self.dim))

        sobol_idx = 0
        total_budget = self.budget

        # (1+1)-ES parameters
        sigma = 1.0  # initial step size
        alpha = 0.5  # step size reduction factor on failure
        beta = 1.0   # step size increase factor on success

        def one_plus_one_es(start_x, start_f, max_steps):
            """Run (1+1)-ES from start_x for at most max_steps evaluations."""
            nonlocal sigma
            x = start_x.copy()
            f = start_f
            for _ in range(max_steps):
                if self._call_count >= total_budget or time.time() >= self._deadline:
                    break
                # Generate trial point
                trial = x + sigma * self._rng.standard_normal(self.dim)
                # Clip to box
                trial = np.clip(trial, -5.0, 5.0)
                trial_f = safe_call(trial)
                if trial_f < f:
                    x = trial
                    f = trial_f
                    sigma = min(sigma * beta, 5.0)  # increase step size
                else:
                    sigma = max(sigma * alpha, 1e-4)  # decrease step size
            return x, f

        # Phase 1: Global Exploration
        # Use first 30 calls to find a decent starting point
        for i in range(min(30, total_budget)):
            if self._call_count >= total_budget or time.time() >= self._deadline:
                break
            x = sobol_points[sobol_idx]
            sobol_idx += 1
            f = safe_call(x)

        # Phase 2: Hybrid Exploration/Exploitation
        last_improve_call = self._call_count
        sigma = 1.0

        while self._call_count < total_budget - 30 and time.time() < self._deadline:
            # Get next global point
            if sobol_idx >= len(sobol_points):
                x = self._rng.uniform(-5, 5, self.dim)
            else:
                x = sobol_points[sobol_idx]
                sobol_idx += 1

            f = safe_call(x)

            # Decide whether to refine
            should_refine = False
            refine_x = self._best_x
            refine_f = self._best_f

            if f < self._best_f:
                should_refine = True
                refine_x = x
                refine_f = f
                last_improve_call = self._call_count
                sigma = 1.0  # reset step size for new basin
            elif self._call_count - last_improve_call > 8:
                # Stagnation: try a larger jump via Sobol, then refine
                # First, jump to a new Sobol point
                if sobol_idx < len(sobol_points):
                    x_jump = sobol_points[sobol_idx]
                    sobol_idx += 1
                    f_jump = safe_call(x_jump)
                    if f_jump < self._best_f:
                        refine_x = x_jump
                        refine_f = f_jump
                        last_improve_call = self._call_count
                        sigma = 1.0
                        should_refine = True
                else:
                    x_jump = self._rng.uniform(-5, 5, self.dim)
                    f_jump = safe_call(x_jump)
                    if f_jump < self._best_f:
                        refine_x = x_jump
                        refine_f = f_jump
                        last_improve_call = self._call_count
                        sigma = 1.0
                        should_refine = True
                last_improve_call = self._call_count

            if should_refine and self._call_count < total_budget - 15:
                # Run a short (1+1)-ES burst
                refine_x, refine_f = one_plus_one_es(refine_x, refine_f, 10)

        # Final Polish: tighter (1+1)-ES
        if self._call_count < total_budget and time.time() < self._deadline:
            sigma = 0.3
            one_plus_one_es(self._best_x, self._best_f, 25)

        return self._best_x, self._best_f

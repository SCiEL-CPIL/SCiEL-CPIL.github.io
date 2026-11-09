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

        # Strategy: Basin Hopping
        # 1. Start with a random point
        # 2. Local Minimize (Nelder-Mead)
        # 3. Jump
        # 4. Local Minimize
        # 5. Repeat until budget/time exhausted

        # Initial point
        x0 = self._rng.uniform(-5, 5, self.dim)
        safe_call(x0)
        
        current_x = x0.copy()
        current_f = self._best_f
        
        # We will perform a series of "Jumps + Local Refinements"
        # Each local refinement (Nelder-Mead) typically costs ~20-50 evaluations in dim 5.
        # We have 500 budget. So we can afford ~10-20 such cycles.
        
        jump_size = 2.0 # Initial jump size
        no_improve_count = 0
        
        while self._call_count < self.budget - 30 and time.time() < self._deadline:
            # Local Refinement
            try:
                res = minimize(
                    safe_call,
                    current_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                )
                # Update current state to the result of minimization
                if res.fun < 1e9: # Check if minimization succeeded
                    current_x = res.x.copy()
                    current_f = res.fun
                else:
                    if self._best_f < 1e9:
                        current_x = self._best_x.copy()
                        current_f = self._best_f
            except Exception:
                break

            # Check if we have enough budget for a jump and another refinement
            if self._call_count > self.budget - 40:
                break

            # Global Jump
            noise = self._rng.normal(0, jump_size, self.dim)
            next_x = current_x + noise
            next_x = np.clip(next_x, -5, 5)
            
            # Evaluate the jump point
            f_jump = safe_call(next_x)
            
            # Update current state to the jumped point for the next iteration
            current_x = next_x
            current_f = f_jump
            
            # Adapt jump size: increase if no improvement in best, decrease if improving
            if self._best_f < 1e9 and current_f > self._best_f:
                no_improve_count += 1
                if no_improve_count > 3:
                    jump_size *= 1.5
                    no_improve_count = 0
            else:
                no_improve_count = 0
                jump_size *= 0.9
                jump_size = max(jump_size, 0.5)

        # Final polish
        if self._call_count < self.budget and time.time() < self._deadline:
            try:
                res = minimize(
                    safe_call,
                    self._best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
            except Exception:
                pass

        return self._best_x, self._best_f

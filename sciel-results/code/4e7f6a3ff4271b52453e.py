"""Derivative-Free Trust-Region (DFTR) optimizer for BBOB f18.

Uses local quadratic surrogates built from directional derivative approximations
to navigate the ill-conditioned Schaffers F7 landscape. This structural change
moves from population-based CMA-ES to a model-based local search that explicitly
fits local geometry, potentially providing faster convergence on smooth basins.
"""
import time
import numpy as np


class Optimizer:
    """DFTR with local quadratic surrogates and restart mechanism."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 290.0                      # safety margin before 330s kill
        calls = [0]
        n = self.dim

        def time_ok():
            return (time.time() - t0) < WALL

        def safe(x):
            """One recorded, exception-safe, budget/time-guarded evaluation."""
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        rng = np.random.default_rng(42)
        
        # Initial exploration: coarse grid/random sampling to find a good starting point
        best_x = None
        best_f = np.inf
        
        # Phase 1: Initial random sampling to locate promising basin
        n_init = min(30, self.budget // 2)
        for i in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-4.0, 4.0, n)
            f = safe(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe(best_x) or 1e18
        
        # Phase 2: DFTR local optimization around best point
        def dftr_local_search(x0, f0, remaining_budget):
            """Run DFTR from x0 with initial function value f0."""
            x = x0.copy()
            fx = f0
            delta = 1.0  # trust region radius
            max_iter = min(remaining_budget // (2 * n + 1), 50)
            
            for it in range(max_iter):
                if not time_ok() or calls[0] >= self.budget:
                    break
                
                # Build local quadratic model using coordinate directions
                # Evaluate x +/- delta * e_i for each dimension
                H = np.zeros((n, n))
                g = np.zeros(n)
                
                # Central difference for gradient and Hessian
                for i in range(n):
                    if calls[0] >= self.budget - 1 or not time_ok():
                        break
                    
                    xp = x.copy()
                    xp[i] += delta
                    xp = np.clip(xp, -5.0, 5.0)
                    fp = safe(xp)
                    if fp is None:
                        break
                    
                    xm = x.copy()
                    xm[i] -= delta
                    xm = np.clip(xm, -5.0, 5.0)
                    fm = safe(xm)
                    if fm is None:
                        break
                    
                    # Gradient component
                    g[i] = (fp - fm) / (2.0 * delta)
                    
                    # Hessian diagonal approximation
                    H[i, i] = (fp - 2.0 * fx + fm) / (delta ** 2)
                
                # Check if we got enough evaluations
                if calls[0] >= self.budget or not time_ok():
                    break
                
                # Ensure H is positive definite by adding regularization
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < 1e-8:
                    H += (1e-8 - np.min(eigvals)) * np.eye(n)
                
                # Solve for step: minimize 0.5 * d^T H d + g^T d
                try:
                    step = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    step = -g / (np.linalg.norm(g) + 1e-12)
                
                # Limit step to trust region
                step_norm = np.linalg.norm(step)
                if step_norm > delta:
                    step = step * (delta / step_norm)
                
                # Try the step
                x_new = x + step
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = safe(x_new)
                
                if f_new is None:
                    break
                
                # Accept if improvement
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    delta *= 1.5  # expand trust region
                    delta = min(delta, 5.0)
                else:
                    delta *= 0.5  # shrink trust region
                    if delta < 1e-4:
                        break
            
            return x, fx
        
        # Run DFTR from the best point found in Phase 1
        remaining = self.budget - calls[0]
        if remaining > 2 * n + 1:
            best_x, best_f = dftr_local_search(best_x, best_f, remaining)
        
        # Phase 3: If budget remains, do multi-restart with smaller neighborhoods
        if calls[0] < self.budget and time_ok():
            n_restarts = min(5, (self.budget - calls[0]) // (2 * n + 1))
            for r in range(n_restarts):
                if not time_ok() or calls[0] >= self.budget:
                    break
                
                # Restart from a point near current best
                x_start = best_x + 0.5 * rng.standard_normal(n)
                x_start = np.clip(x_start, -5.0, 5.0)
                f_start = safe(x_start)
                if f_start is None:
                    break
                
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()
                
                remaining = self.budget - calls[0]
                if remaining > 2 * n + 1:
                    x_local, f_local = dftr_local_search(x_start, f_start, remaining)
                    if f_local < best_f:
                        best_f = f_local
                        best_x = x_local.copy()
        
        return best_x, float(best_f)

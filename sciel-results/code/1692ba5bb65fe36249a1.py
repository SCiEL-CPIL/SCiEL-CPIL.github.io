"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class Optimizer:
    """
    Optimizer for rotated Rosenbrock in 5D.
    
    Strategy:
    1. Structured global sampling (Halton sequence + corners) to identify the basin.
    2. Nelder-Mead with adaptive restarts to navigate the narrow valley robustly.
    3. L-BFGS-B for high-precision polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf
        self.all_evals = []  # List of (fx, x) for tracking

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit (evaluator kills at 330s, but we want to be safe)
        
        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)
        
        # Wrapper to track calls and enforce budget/deadline
        def tracked_func(x):
            if not time_ok() or self.call_count >= self.budget:
                raise ValueError("Budget or time exceeded")
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            self.call_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            
            self.all_evals.append((fx, x.copy()))
            return fx

        # Phase 1: Global Exploration
        # Use a Halton sequence for low-discrepancy sampling in 5D.
        # This is much more efficient than random sampling for covering the space.
        n_global = 100
        halton_points = self._halton_sequence(n_global, self.dim)
        
        # Add corners to ensure boundary coverage
        corners = []
        for i in range(2**self.dim):
            corner = np.array([(i >> j) & 1 for j in range(self.dim)], dtype=float)
            corner = corner * 10.0 - 5.0
            corners.append(corner)
        
        # Combine and shuffle slightly to avoid order bias
        global_starts = list(halton_points) + corners
        rng.shuffle(global_starts)
        
        # Evaluate global starts
        for x0 in global_starts:
            if not time_ok() or self.call_count >= self.budget - 50:
                break
            try:
                tracked_func(x0)
            except Exception:
                pass

        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Select top K candidates from global phase for local refinement
        # Sort by function value
        self.all_evals.sort(key=lambda t: t[0])
        top_k = 5
        candidates = [x for fx, x in self.all_evals[:top_k]]
        
        # Phase 2: Local Optimization with Nelder-Mead
        # Nelder-Mead is derivative-free and robust to the narrow valley of Rosenbrock.
        # We run it from the best candidates.
        for x_start in candidates:
            if not time_ok() or self.call_count >= self.budget - 20:
                break
            
            # Run Nelder-Mead with adaptive restarts
            self._nm_with_restarts(x_start, func, tracked_func, time_ok)
            
            # If we found a significantly better point, we might want to restart NM from there
            # But for now, just move to the next candidate.

        # Phase 3: Polishing with L-BFGS-B
        # L-BFGS-B is very fast for smooth functions and can get high precision.
        if not time_ok() or self.call_count >= self.budget - 5:
            return self.best_x, self.best_fx

        bounds = [(-5.0, 5.0)] * self.dim
        try:
            res = minimize(
                tracked_func,
                self.best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 100,
                    'maxfun': min(100, self.budget - self.call_count - 2),
                    'ftol': 1e-15,
                    'gtol': 1e-12
                }
            )
        except Exception:
            pass

        # Ensure we return a valid solution
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

    def _halton_sequence(self, n, dim):
        """Generate a Halton sequence in 'dim' dimensions."""
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._halton(i + 1, j + 2)  # Start with prime 2, 3, 5, 7, 11...
        # Map to [-5, 5]
        points = points * 10.0 - 5.0
        return points

    def _halton(self, index, base):
        """Compute the Halton sequence value for a given index and base."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _nm_with_restarts(self, x_start, func, tracked_func, time_ok):
        """Run Nelder-Mead with adaptive restarts if it stagnates."""
        bounds = [(-5.0, 5.0)] * self.dim
        x = x_start.copy()
        
        # Initial simplex size
        initial_simplex = 0.5 * np.ones(self.dim)
        
        for restart in range(3):
            try:
                # Create initial simplex
                simplex = [x.copy()]
                for i in range(self.dim):
                    new_x = x.copy()
                    new_x[i] += initial_simplex[i]
                    new_x = np.clip(new_x, -5.0, 5.0)
                    simplex.append(new_x)
                
                # Evaluate initial simplex
                simplex_f = []
                for sx in simplex:
                    if not time_ok() or self.call_count >= self.budget - 10:
                        return
                    try:
                        fx = tracked_func(sx)
                    except Exception:
                        return
                    simplex_f.append(fx)
                
                # Run Nelder-Mead
                res = minimize(
                    tracked_func,
                    x,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfev': min(100, self.budget - self.call_count - 10),
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # Check for improvement
                if res.fun < self.best_fx:
                    self.best_x = res.x.copy()
                    self.best_fx = res.fun
                    x = res.x.copy()
                
                # Check for stagnation: if no significant improvement in the last few restarts
                # We don't have easy access to the history, so we just check if we are still improving
                # For now, we just break if we hit the time/budget limit.
                
            except Exception:
                break

# EVOLVE-BLOCK-END

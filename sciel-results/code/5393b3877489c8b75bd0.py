# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-Start Derivative-Free Broyden (DFB) optimizer for BBOB f7 (Step Ellipsoidal).
    
    Strategy:
    1. Coarse random search to identify the non-flat side of the step function.
    2. Multiple parallel DFB local optimizations from diverse starting points.
    3. DFB uses finite-difference gradients and a Broyden Hessian approximation
       to achieve super-linear convergence within the smooth convex basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def _count_call(self):
        self._call_count += 1

    def _evaluate(self, func, x):
        """Evaluate function with call counting."""
        self._count_call()
        return func(x)

    def _dfb_local_search(self, func, x0, max_iter=100, tol=1e-6):
        """
        Derivative-Free Broyden local optimization.
        
        Uses finite-difference gradients and a Broyden update for the inverse Hessian.
        This is a quasi-Newton method that achieves super-linear convergence
        on smooth convex functions like the ellipsoidal part of f7.
        """
        if self._call_count >= self.budget:
            return x0, self._best_f

        x = np.array(x0, dtype=float)
        d = self.dim
        
        # Initial gradient estimate via central differences
        h = 1e-5
        f0 = self._evaluate(func, x)
        
        if f0 < self._best_f:
            self._best_f = f0
            self._best_x = x.copy()

        # Estimate gradient
        grad = np.zeros(d)
        for i in range(d):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            fp = self._evaluate(func, x_plus)
            fm = self._evaluate(func, x_minus)
            grad[i] = (fp - fm) / (2 * h)
        
        # If we've exceeded budget, return
        if self._call_count >= self.budget:
            return x, self._best_f

        # Broyden inverse Hessian approximation: H = I initially
        H = np.eye(d)
        
        for iteration in range(max_iter):
            if self._call_count >= self.budget:
                break
            
            # Check convergence
            if np.linalg.norm(grad) < tol:
                break
            
            # Newton step: s = -H @ grad
            s = -H @ grad
            
            # Limit step size to avoid large jumps
            step_norm = np.linalg.norm(s)
            if step_norm > 1.0:
                s = s / step_norm
            
            # Try the step
            x_new = x + s
            x_new = np.clip(x_new, -5, 5)
            f_new = self._evaluate(func, x_new)
            
            if f_new < self._best_f:
                self._best_f = f_new
                self._best_x = x_new.copy()
            
            # Estimate new gradient at x_new
            grad_new = np.zeros(d)
            for i in range(d):
                x_plus = x_new.copy()
                x_plus[i] += h
                x_minus = x_new.copy()
                x_minus[i] -= h
                fp = self._evaluate(func, x_plus)
                fm = self._evaluate(func, x_minus)
                grad_new[i] = (fp - fm) / (2 * h)
            
            if self._call_count >= self.budget:
                return x_new, self._best_f
            
            # Broyden update for inverse Hessian
            # s = x_new - x, y = grad_new - grad
            s_vec = x_new - x
            y_vec = grad_new - grad
            
            # Avoid division by zero
            sy = s_vec @ y_vec
            if abs(sy) < 1e-10:
                H = H + (np.outer(s_vec, s_vec) / (s_vec @ s_vec + 1e-10)) * 0.5
            else:
                # Standard Broyden update for inverse Hessian
                H = H + np.outer(s_vec, s_vec) / sy - np.outer(H @ y_vec, y_vec) / (y_vec @ H @ y_vec + 1e-10)
            
            x = x_new
            grad = grad_new
        
        return x, self._best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        rng = np.random.default_rng(12345)
        
        # Phase 1: Coarse random search to find the non-flat side
        # The step function is flat on one side of a hyperplane through the optimum.
        # We need to find points on the non-flat (convex) side.
        
        # Start with a diverse set of points
        n_coarse = min(30, self.budget // 10)
        
        # Include origin and some structured points
        test_points = [np.zeros(self.dim)]
        # Add points along each axis
        for i in range(self.dim):
            test_points.append(np.full(self.dim, 2.0) * np.eye(self.dim)[i])
            test_points.append(np.full(self.dim, -2.0) * np.eye(self.dim)[i])
        
        # Add random points
        while len(test_points) < n_coarse:
            test_points.append(rng.uniform(-4, 4, self.dim))
        
        best_coarse_x = None
        best_coarse_f = np.inf
        flat_points = []  # Points that seem to be on the flat side
        
        for x in test_points:
            if self._call_count >= self.budget or time.time() > deadline:
                break
            
            x = np.clip(x, -5, 5)
            fx = self._evaluate(func, x)
            
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            
            if fx < best_coarse_f:
                best_coarse_f = fx
                best_coarse_x = x.copy()
            else:
                # If this point is worse than the best, it might be on the flat side
                # or just far from optimum
                flat_points.append(x.copy())
        
        if self._call_count >= self.budget or time.time() > deadline:
            return self._best_x, self._best_f
        
        # Phase 2: Multi-start DFB from diverse starting points
        # Use the best coarse point and several other points as starting points
        
        start_points = [best_coarse_x.copy()] if best_coarse_x is not None else []
        
        # Add some points near the best found so far
        if self._best_x is not None:
            for scale in [0.5, 1.0, 2.0]:
                x_start = self._best_x + scale * rng.standard_normal(self.dim)
                x_start = np.clip(x_start, -5, 5)
                start_points.append(x_start)
        
        # Add some random starting points
        n_starts = min(5, (self.budget - self._call_count) // 50)
        for _ in range(n_starts):
            x_start = rng.uniform(-4, 4, self.dim)
            start_points.append(x_start)
        
        # Remove duplicates and out-of-bounds
        unique_starts = []
        for sp in start_points:
            sp = np.clip(sp, -5, 5)
            if not any(np.allclose(sp, u, atol=1e-3) for u in unique_starts):
                unique_starts.append(sp)
        
        # Run DFB from each starting point
        for x_start in unique_starts:
            if self._call_count >= self.budget or time.time() > deadline:
                break
            
            # Estimate remaining budget for this local search
            remaining = self.budget - self._call_count
            if remaining < 20:  # Need at least enough for gradient estimation
                break
            
            # Run DFB with limited iterations
            max_iter = min(30, remaining // 10)  # Each iteration costs ~2*d+1 calls
            self._dfb_local_search(func, x_start, max_iter=max_iter)
        
        # Phase 3: Final polish with a more careful DFB from the best point
        if self._best_x is not None and self._call_count < self.budget - 30:
            if time.time() < deadline:
                remaining = self.budget - self._call_count
                max_iter = min(20, remaining // 10)
                self._dfb_local_search(func, self._best_x, max_iter=max_iter)
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END

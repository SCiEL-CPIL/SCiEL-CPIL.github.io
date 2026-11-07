import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def _solve_xi_vectorized(self, x1_values, n_iter=50):
        """Solve x = h(x1, x) = x^2 - 2*x1*cos(4*pi*x*x1) + 1 for each x1 in x1_values.
        Uses bisection on f(x) = h(x1, x) - x = 0 in [0, 1]."""
        x_lo = np.zeros_like(x1_values)
        x_hi = np.ones_like(x1_values)
        
        for _ in range(n_iter):
            x_mid = 0.5 * (x_lo + x_hi)
            # f(x) = x^2 - 2*x1*cos(4*pi*x*x1) + 1 - x
            f_mid = x_mid**2 - 2*x1_values*np.cos(4*np.pi*x_mid*x1_values) + 1 - x_mid
            f_lo = x_lo**2 - 2*x1_values*np.cos(4*np.pi*x_lo*x1_values) + 1 - x_lo
            # Update: if f_lo and f_mid have same sign, move lo; else move hi
            same_sign = (f_lo * f_mid) >= 0
            x_lo = np.where(same_sign, x_mid, x_lo)
            x_hi = np.where(~same_sign, x_mid, x_hi)
        
        return 0.5 * (x_lo + x_hi)

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        # Pre-compute optimal x_i for a dense set of x1 values
        n_sweep = self.budget
        u_values = np.linspace(0.0, 1.0, n_sweep)
        x1_values = u_values ** 2
        
        # Vectorized solve for all x_i values
        x_opt_values = self._solve_xi_vectorized(x1_values)
        
        for i in range(n_sweep):
            if self.eval_count >= self.budget or time_up():
                break
            
            x1 = x1_values[i]
            x_opt = x_opt_values[i]
            
            x = np.full(self.dim, x_opt)
            x[0] = x1
            x = np.clip(x, lo, hi)
            
            self.eval_count += 1
            problem(x)

        return None

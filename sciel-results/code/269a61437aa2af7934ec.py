"""Combined optimizer: Sphere-specific quadratic fit + (1+1)-ES with 1/5 rule."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sphere-specific optimizer: quadratic fit init + (1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_fx = np.inf

        def make_call(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline - 2.0:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sphere-specific quadratic fit using 2*dim+1 points
        # f(x) = a*||x-c||^2 + offset
        # Fit: f(x) = a*||x||^2 + w^T x + b
        # Minimum at x* = -w/(2a)
        
        r = 3.0
        points = [np.zeros(dim)]
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = r
            points.append(e_i)
            points.append(-e_i)
        
        # Evaluate all points
        evals = []
        for p in points:
            fx = make_call(p)
            if fx is None:
                break
            evals.append(fx)
        
        # If we couldn't evaluate all points, fallback to random search
        if len(evals) < 2 * dim + 1:
            for _ in range(budget - call_count):
                if time.time() > deadline - 2.0:
                    break
                x = rng.uniform(-5, 5, dim)
                make_call(x)
            return best_x, best_fx
        
        # Fit f(x) = a*||x||^2 + w^T x + b
        # At x=0: f(0) = b
        # At x=e_i*r: f = a*r^2 + w_i*r + b
        # At x=-e_i*r: f = a*r^2 - w_i*r + b
        # So: w_i = (f(e_i*r) - f(-e_i*r)) / (2*r)
        #     a = (f(e_i*r) + f(-e_i*r) - 2*b) / (2*r^2)
        
        b = evals[0]
        w = np.zeros(dim)
        a_sum = 0.0
        for i in range(dim):
            w[i] = (evals[1 + 2*i] - evals[2 + 2*i]) / (2.0 * r)
            a_sum += (evals[1 + 2*i] + evals[2 + 2*i] - 2.0 * b) / (2.0 * r**2)
        a = a_sum / dim
        
        # If a <= 0, the fit is degenerate; fall back to best evaluated point
        if a > 1e-10:
            x0 = -w / (2.0 * a)
        else:
            best_idx = np.argmin(evals)
            x0 = points[best_idx].copy()
        
        # Clip to box
        x0 = np.clip(x0, -5.0, 5.0)
        fx0 = make_call(x0)
        if fx0 is None:
            return best_x, best_fx
        
        # Phase 2: (1+1)-ES with 1/5 success rule
        sigma = 0.3
        x_cur = x0.copy()
        f_cur = fx0
        
        # 1/5 success rule parameters
        sigma_up = 1.05
        sigma_down = 0.95
        window = 10
        successes = 0
        
        while call_count < budget:
            if time.time() > deadline - 2.0:
                break
            
            # Generate perturbation
            z = rng.standard_normal(dim)
            delta = sigma * z
            x_new = x_cur + delta
            x_new = np.clip(x_new, -5, 5)
            
            f_new = make_call(x_new)
            if f_new is None:
                break
            
            # Update best
            if f_new < best_fx:
                best_fx = f_new
                best_x = x_new.copy()
            
            # 1/5 rule
            if f_new < f_cur:
                successes += 1
                x_cur = x_new.copy()
                f_cur = f_new
            
            # Adjust sigma every 'window' steps
            if (call_count % window) == 0:
                if successes / window > 0.2:
                    sigma *= sigma_up
                else:
                    sigma *= sigma_down
                successes = 0
                sigma = np.clip(sigma, 1e-10, 10.0)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END

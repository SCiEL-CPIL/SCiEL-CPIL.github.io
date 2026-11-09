"""BFGS-based optimizer with finite-difference gradients and line search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """BFGS-based optimizer with finite-difference gradients and line search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 275.0          # safety margin vs. 330 s kill
        max_calls = budget

        calls = [0]

        def f(x):
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                return 1e18
            return val

        def remaining():
            return deadline - time.time()

        def finite_diff_gradient(x, h=1e-5):
            """Estimate gradient using central differences."""
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2 * h)
            return g

        def bfgs(x0, fx0, max_iter=50):
            """Run BFGS optimization starting from x0."""
            n = dim
            x = x0.copy()
            fx = fx0
            W = np.eye(n)  # Inverse Hessian approximation
            g = finite_diff_gradient(x)
            
            for _ in range(max_iter):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                
                # Check for convergence
                if np.linalg.norm(g) < 1e-8:
                    break
                
                # Compute step direction
                p = -W @ g
                
                # Line search (Backtracking)
                alpha = 1.0
                c1 = 1e-4
                x_new = x
                fx_new = fx
                success = False
                for _ in range(50):
                    if calls[0] >= max_calls or remaining() < 5.0:
                        break
                    x_trial = x + alpha * p
                    fx_trial = f(x_trial)
                    
                    # Armijo condition
                    if fx_trial <= fx + c1 * alpha * (g @ p):
                        x_new = x_trial
                        fx_new = fx_trial
                        success = True
                        break
                    alpha *= 0.5
                
                if not success or alpha < 1e-8:
                    break
                
                # Compute new gradient only if we accept the step
                g_new = finite_diff_gradient(x_new)
                
                # Update BFGS matrix
                s = x_new - x
                y = g_new - g
                sy = s @ y
                if sy > 1e-10:
                    I = np.eye(n)
                    W = (I - np.outer(s, y) / sy) @ W @ (I - np.outer(y, s) / sy) + np.outer(s, s) / sy
                
                x = x_new
                fx = fx_new
                g = g_new
                
            return x, fx

        best_x = None
        best_fx = 1e18

        # Multi-start: a few deterministic starting points
        rng = np.random.default_rng(0)
        starts = []
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(np.array([4.0]*dim))
        starts.append(np.array([-4.0]*dim))

        # Budget split
        per_start = max(20, budget // len(starts))

        for si, s in enumerate(starts):
            if calls[0] >= max_calls or remaining() < 10.0:
                break
            s = np.clip(s, -5, 5)
            fx = f(s)
            
            # Run BFGS
            x, fx = bfgs(s, fx, max_iter=30)
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END

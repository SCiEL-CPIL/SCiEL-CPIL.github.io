"""Quasi-Newton (BFGS) optimizer for BBOB f12 (Bent Cigar) with adaptive line search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """BFGS with finite-difference gradients for smooth functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin
        max_calls = budget

        calls = [0]
        best_x = None
        best_fx = 1e18

        def f(x):
            nonlocal best_x, best_fx
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e18
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        def remaining():
            return deadline - time.time()

        def gradient(x, h=1e-5):
            """Central difference gradient."""
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2 * h)
            return g

        def bfgs(x0, max_iter=50):
            n = dim
            x = x0.copy()
            fx = f(x)
            g = gradient(x)
            
            # Inverse Hessian approximation
            W = np.eye(n)
            
            for it in range(max_iter):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                
                # Check convergence
                if np.linalg.norm(g) < 1e-8:
                    break
                
                # Search direction
                p = -W @ g
                
                # Backtracking line search
                alpha = 1.0
                c1 = 1e-4
                rho = 0.5
                fx_old = fx
                g_old = g
                x_old = x.copy()
                
                # Limit line search iterations to save budget
                success = False
                for _ in range(15):
                    x_new = x + alpha * p
                    fx_new = f(x_new)
                    
                    # Check sufficient decrease
                    if fx_new < fx_old - c1 * alpha * np.dot(g_old, p):
                        success = True
                        break
                    alpha *= rho
                    if alpha < 1e-10:
                        break
                
                if not success:
                    # If line search fails, try steepest descent with small step
                    p = -g
                    alpha = 1e-3
                    x_new = x + alpha * p
                    fx_new = f(x_new)
                    if fx_new < fx_old:
                        success = True
                    else:
                        break

                if not success:
                    break

                # Update
                s = x_new - x
                g_new = gradient(x_new)
                y = g_new - g
                
                # BFGS update
                sy = np.dot(s, y)
                if sy > 1e-12:
                    rho_val = 1.0 / sy
                    I = np.eye(n)
                    term1 = I - np.outer(rho_val * s, y)
                    term2 = I - np.outer(rho_val * y, s)
                    W = term1 @ W @ term2 + rho_val * np.outer(s, s)
                    
                    # Ensure symmetry and positive definiteness roughly
                    W = (W + W.T) / 2.0
                    # Simple check: if eigenvalues are bad, reset to identity
                    try:
                        eigvals = np.linalg.eigvalsh(W)
                        if np.min(eigvals) < 1e-6:
                            W = np.eye(n)
                    except:
                        W = np.eye(n)
                
                x = x_new
                fx = fx_new
                g = g_new

            return x, fx

        # Multi-start BFGS
        rng = np.random.default_rng(42)
        # Bent Cigar is convex, so one good start is enough. 
        # But rotation/shift might move the minimum. 
        # The minimum is in [-4, 4]^5.
        starts = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim)
        ]

        for s in starts:
            if calls[0] >= max_calls or remaining() < 10.0:
                break
            s = np.clip(s, -5, 5)
            fx = f(s) # Consume 1 call to initialize best
            x, fx = bfgs(s, max_iter=50)
            
            if fx < best_fx:
                 best_fx = fx
                 best_x = x

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END

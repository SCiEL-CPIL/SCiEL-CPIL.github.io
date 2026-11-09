"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: LHS spread + CMA-ES + L-BFGS-B polish.
    
    Designed for BBOB-style Sphere function with hidden shift/rotation/offset.
    AOCC rewards early progress, so we start with a cheap global spread.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples * 10.0 - 5.0  # Map to [-5, 5]

    def _coordinate_descent(self, func, x0, budget, deadline, dim, call_count):
        """Deterministic multi-scale coordinate descent for separable quadratic (Sphere).
        
        Sweeps each axis with exponentially decreasing step sizes from coarse to fine.
        For Sphere f(x) = ||x - x*||^2 + offset, each 1D slice is a parabola,
        so coordinate descent converges linearly.
        """
        x = x0.copy()
        fx = func(x)
        n = dim
        
        # Multi-scale: start with large steps, halve each pass
        # Initial step size covers the box
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125]
        
        for step in step_sizes:
            if time.time() > deadline or call_count[0] >= budget:
                break
            
            # Multiple passes per step size for convergence
            n_passes = 3 if step > 0.1 else 5
            for _ in range(n_passes):
                if time.time() > deadline or call_count[0] >= budget:
                    break
                for d in range(n):
                    if time.time() > deadline or call_count[0] >= budget:
                        break
                    # Try moving in positive and negative direction
                    for sign in [1, -1]:
                        if call_count[0] >= budget:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                        f_trial = func(trial)
                        if f_trial < fx:
                            x = trial
                            fx = f_trial
        
        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = [0]
        
        def wrapped(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))
        
        # Phase 1: Latin Hypercube Sampling spread (cheap global exploration)
        # This gives us a good initial best quickly for AOCC
        n_lhs = 31
        rng = np.random.default_rng(42)
        lhs_samples = self._lhs_sample(n_lhs, dim, rng)
        
        for x in lhs_samples:
            if time.time() > deadline or call_count[0] >= budget:
                break
            fx = wrapped(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)
        
        # Phase 2: Multi-scale coordinate descent (exploits separable quadratic structure)
        if time.time() < deadline - 20 and call_count[0] < budget - 30:
            try:
                x_cd, f_cd = self._coordinate_descent(wrapped, best_x, budget - call_count[0], deadline - 20, dim, call_count)
                if f_cd < best_f:
                    best_f = f_cd
                    best_x = x_cd
            except Exception:
                pass
        
        # Phase 3: L-BFGS-B final polish (convex Sphere function)
        if time.time() < deadline - 5 and call_count[0] < budget - 5:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped, best_x, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 100, "ftol": 1e-15, "gtol": 1e-10})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END

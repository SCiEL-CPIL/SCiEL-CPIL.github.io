"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        # Wrapper to track calls and enforce budget
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with a stratified random grid
        # Use a moderate number of initial points to cover the space
        n_initial = 50
        np.random.seed(42)
        
        # Generate stratified points: divide each dimension into segments
        # For 5D, 50 points is not enough for full stratification, so use random with some structure
        # Actually, let's use a simple random start plus a few structured ones
        
        # 1. Center
        starts = [np.zeros(dim)]
        
        # 2. Random uniform points
        for _ in range(n_initial - 1):
            p = np.random.uniform(-5, 5, dim)
            starts.append(p)

        # Evaluate all initial points
        for p in starts:
            if calls >= budget or time.time() > deadline:
                break
            f = wrapped_func(p)

        # Phase 2: Local refinement using L-BFGS-B from the best starting points
        # Sort starts by their function value (we need to track which was which)
        # Since wrapped_func updates best_f, we need to store the individual results
        # Let's re-evaluate or store them properly.
        
        # Actually, let's just run L-BFGS-B from the current best and a few other good candidates.
        # But we don't have the list of (x, f) pairs easily accessible from wrapped_func unless we store them.
        
        # Let's restructure: store all evaluated points in a list.
        # However, the current wrapped_func doesn't return a list.
        # Let's just use the global best as the primary start for L-BFGS-B.
        # To improve, we can also try L-BFGS-B from a few other random points if budget allows.
        
        # Let's collect the top few candidates from the initial phase.
        # Since we didn't store them, let's just use the best_x found so far.
        # A better approach: re-run the initial evaluation storing results, or just trust the best.
        
        # Let's do a second phase: run L-BFGS-B from the current best_x.
        # Also, run it from a few other random points if we have budget left.
        
        # To be more robust, let's identify a few diverse starting points for L-BFGS-B.
        # We can generate a few new random points and evaluate them, then pick the best among all.
        
        # Let's just run L-BFGS-B from the current best.
        # And maybe from 3 other random points to escape local minima.
        
        n_local_starts = 4
        local_starts = [best_x.copy()]
        
        # Add 3 random points for local optimization
        for _ in range(n_local_starts - 1):
            if calls + 100 > budget:  # L-BFGS-B might use many calls
                break
            p = np.random.uniform(-5, 5, dim)
            local_starts.append(p)

        for start_p in local_starts:
            if calls >= budget or time.time() > deadline:
                break
            
            # Use L-BFGS-B with bounds
            # maxiter is not directly a call count, but we can limit it.
            # L-BFGS-B typically uses a few dozen to a few hundred calls.
            # We'll rely on the wrapped_func budget check.
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_p, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 100}
                )
                # wrapped_func updates best_f and best_x automatically
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

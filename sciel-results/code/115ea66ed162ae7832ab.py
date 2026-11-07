"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Deterministic Nelder-Mead Simplex Optimizer with multi-start restarts.
    
    Strategy:
    - Uses Nelder-Mead (a Hessian-free, gradient-free method) which maintains an implicit
      search direction basis via a simplex of points.
    - This allows rapid exploitation of local convexity without the high per-iteration
      computational cost or population overhead of CMA-ES.
    - Multi-start approach to handle the 21 peaks (non-convexity) of Gallagher f22.
    - Budget is divided into several independent Nelder-Mead runs.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Number of restarts. Nelder-Mead is efficient, so we can afford a few restarts.
        # Each run needs at least dim+1 points for the simplex.
        # Let's try 5 restarts.
        n_restarts = 5
        
        # Allocate budget per restart
        # We need at least dim+1 calls to initialize the simplex.
        # Let's reserve a minimum of 50 calls per restart if possible, but not exceed total.
        budget_per_restart = self.budget // n_restarts
        if budget_per_restart < self.dim + 1:
            budget_per_restart = self.dim + 1
            n_restarts = self.budget // budget_per_restart
            if n_restarts == 0:
                n_restarts = 1
                budget_per_restart = self.budget

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break

            # Initialize starting point for this restart
            if r == 0:
                # Start near center, as minimizer is often near center in BBOB
                x0 = rng.uniform(-2, 2, self.dim)
            else:
                # Random start within the box
                x0 = rng.uniform(-5, 5, self.dim)

            # Initialize Simplex
            # The simplex has dim+1 points.
            # Point 0 is x0.
            # Points 1..dim are x0 + delta * e_i
            delta = 0.5 # Initial step size for simplex expansion
            
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x0
            for i in range(1, self.dim + 1):
                simplex[i] = x0
                simplex[i][i-1] += delta
            
            # Evaluate initial simplex
            fvals = np.zeros(self.dim + 1)
            for i in range(self.dim + 1):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                fvals[i] = func(simplex[i])
                calls_made += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = simplex[i].copy()
            
            if calls_made >= self.budget or time.time() > deadline:
                break

            # Nelder-Mead Parameters
            alpha = 1.0    # Reflection coefficient
            gamma = 2.0    # Expansion coefficient
            rho = 0.5      # Contraction coefficient
            sigma = 0.5    # Shrinkage coefficient
            
            # Main Loop
            while calls_made < self.budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check for convergence: if the spread of function values is small
                # and the spread of points is small, we can stop this restart.
                if np.max(fvals) - np.min(fvals) < 1e-8 * (1 + abs(np.min(fvals))):
                    break
                    
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                worst_fval = fvals[-1]
                
                # 1. Reflection
                xr = centroid + alpha * (centroid - worst_point)
                fr = func(xr)
                calls_made += 1
                if fr < best_fx:
                    best_fx = fr
                    best_x = xr.copy()
                
                if fr < fvals[0]:
                    # 2. Expansion
                    xe = centroid + gamma * (centroid - worst_point)
                    fe = func(xe)
                    calls_made += 1
                    if fe < best_fx:
                        best_fx = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Just accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # 3. Contraction
                    # Check if it's an outside or inside contraction
                    if fr < worst_fval:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (worst_point - centroid)
                    
                    fc = func(xc)
                    calls_made += 1
                    if fc < best_fx:
                        best_fx = fc
                        best_x = xc.copy()
                    
                    if fc < min(fr, worst_fval):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # 4. Shrinkage
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            # Re-evaluate shrunken points
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_fx:
                                best_fx = fvals[i]
                                best_x = simplex[i].copy()
                        
                        if calls_made >= self.budget or time.time() > deadline:
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END

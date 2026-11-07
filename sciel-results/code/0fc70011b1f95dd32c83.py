import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        
        # Phase 1: (1+1)-ES for global exploration
        # Schwefel is multimodal. Large steps help escape local minima.
        sigma = 2.0
        target_budget_phase1 = 300
        
        while calls < target_budget_phase1 and time.time() < deadline:
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1 # Grow step size on success
            else:
                sigma *= 0.85 # Shrink step size on failure
            
            # Prevent sigma from getting too small or too large
            if sigma < 1e-4:
                sigma = 1e-4
            if sigma > 10.0:
                sigma = 10.0

        # Phase 2: Nelder-Mead for local refinement
        # Initialize simplex around the best point found
        # Initial simplex size should be small but non-zero
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x
        for i in range(self.dim):
            simplex[i + 1] = x.copy()
            simplex[i + 1][i] += 0.1 # Small initial step
            
        # Evaluate initial simplex
        f_simplex = np.array([func(simplex[i]) for i in range(self.dim + 1)])
        calls += self.dim + 1
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma_nm = 0.5 # Shrinkage
        
        # Run NM until budget or time runs out
        while calls < self.budget and time.time() < deadline:
            # Sort simplex by function value
            idx = np.argsort(f_simplex)
            simplex = simplex[idx]
            f_simplex = f_simplex[idx]
            
            # Check convergence: if the difference between best and worst is small
            if f_simplex[-1] - f_simplex[0] < 1e-8:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            
            if fr < f_simplex[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                
                if fc < f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        f_simplex[i] = func(simplex[i])
                        calls += 1
                    if calls >= self.budget:
                        break

        # Find best in simplex
        best_idx = np.argmin(f_simplex)
        x_best = simplex[best_idx]
        f_best = f_simplex[best_idx]
        
        return x_best, f_best

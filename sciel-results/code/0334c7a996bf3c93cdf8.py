"""Rewritten optimizer: (1+1)-ES with restarts, then Nelder-Mead polish."""
import time
import numpy as np


class Optimizer:
    """Global (1+1)-ES search with restarts, followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0
        dim = self.dim
        budget = self.budget
        self.calls = 0

        def _eval(x):
            if self.calls >= budget:
                return np.inf
            self.calls += 1
            return func(x)

        best_x = np.zeros(dim)
        best_f = _eval(best_x)
        
        # Phase 1: (1+1)-ES
        # Use 80% of budget for global search
        n_es = int(budget * 0.8)
        sigma = 2.0
        no_improve_count = 0
        restart_interval = 50
        
        for i in range(n_es):
            if self.calls >= budget or time.time() > deadline - 10.0:
                break
            
            # Generate candidate
            y = best_x + sigma * np.random.randn(dim)
            y = np.clip(y, -5, 5)
            
            fy = _eval(y)
            
            if fy < best_f:
                best_f = fy
                best_x = y
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Restart logic: if stuck for restart_interval steps, pick a new random center
            if no_improve_count >= restart_interval:
                # Pick a new random point
                new_center = np.random.uniform(-5, 5, dim)
                f_new = _eval(new_center)
                if f_new < best_f:
                    best_f = f_new
                    best_x = new_center
                no_improve_count = 0
                # Increase sigma to escape local minima
                sigma = min(sigma * 1.2, 4.0)
                
        # Phase 2: Nelder-Mead
        remaining_budget = budget - self.calls
        if remaining_budget > (dim + 1):
            # Build initial simplex
            simplex = [best_x.copy()]
            for d in range(dim):
                perturbed = best_x.copy()
                step = 0.1
                # Ensure we stay in bounds
                if perturbed[d] + step > 5.0:
                    step = -step
                if perturbed[d] + step < -5.0:
                    step = -step
                perturbed[d] += step
                simplex.append(perturbed)
            
            # Evaluate simplex
            fvals = []
            for s in simplex:
                if self.calls >= budget:
                    break
                fvals.append(_eval(s))
            
            if len(fvals) < dim + 1:
                return best_x, best_f

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            
            # Estimate max iterations based on remaining budget
            # Each iteration costs at least 1 eval, usually 2-6.
            # Let's just loop until budget is exhausted.
            
            while self.calls < budget and time.time() < deadline:
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    
                # Check convergence (optional, but budget is the main driver)
                if len(fvals) < 2:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                
                if self.calls >= budget: break
                fr = _eval(xr)
                
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    if self.calls >= budget: 
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = _eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if self.calls >= budget: break
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = _eval(xc)
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if self.calls >= budget: break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = _eval(simplex[i])

        return best_x, best_f

import numpy as np

class Optimizer:
    """Nelder-Mead optimizer on the box [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        # Initialize simplex
        # Center at 0, step size 1.0
        step = 1.0
        simplex = np.zeros((n + 1, n))
        for i in range(n):
            simplex[i+1, i] = step
        # Evaluate initial simplex
        fvals = np.array([func(x) for x in simplex])
        count = n + 1
        
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = np.min(fvals)
        
        max_iter = budget
        no_improve_count = 0
        threshold = 1e-10
        
        while count < budget:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check for convergence or stagnation
            if fvals[0] < best_f - 1e-12:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # If stuck, try a random restart or jump
            if no_improve_count > 20:
                # Perturb the best point
                # Or restart with a new random simplex around the best point
                center = best_x.copy()
                new_simplex = np.zeros((n + 1, n))
                for i in range(n + 1):
                    new_simplex[i] = center + np.random.randn(n) * 0.5
                    new_simplex[i] = np.clip(new_simplex[i], -5, 5)
                new_fvals = np.array([func(x) for x in new_simplex])
                count += n + 1
                if count >= budget:
                    break
                simplex = new_simplex
                fvals = new_fvals
                no_improve_count = 0
                continue

            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            count += 1
            
            if fr < fvals[0]:
                # Expansion
                if count >= budget:
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                    break
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                count += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Simple reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if count >= budget:
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                    break
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                count += 1
                
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrinkage
                    # Shrink all points except the best towards the best
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        count += 1
                        if count >= budget:
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                            # Re-sort and break
                            order = np.argsort(fvals)
                            simplex = simplex[order]
                            fvals = fvals[order]
                            break
                    
                    if count >= budget:
                        break

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f

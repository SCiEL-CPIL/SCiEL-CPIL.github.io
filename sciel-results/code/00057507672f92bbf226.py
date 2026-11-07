"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Nelder-Mead for robust exploration of multimodal landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        # 1. Initialization: Spread starts across the domain
        # Use a grid-like or random spread. For 5D, 50 starts is reasonable.
        # We need to ensure we don't exceed budget.
        # Let's use a stratified random sampling for initial points.
        rng = np.random.default_rng(42)
        
        # Determine number of restarts. 
        # Nelder-Mead in 5D takes ~20-50 evaluations to converge or stagnate.
        # Let's aim for ~10-20 restarts to allow deep local search in each.
        # If budget is 300, 15 restarts * 20 evals = 300.
        n_restarts = max(1, self.budget // 20)
        
        # Generate initial points. To cover the space well, we can use
        # a combination of random points and maybe some structured points.
        # Simple random uniform is often sufficient for BBOB if we have enough restarts.
        starts = rng.uniform(-5, 5, size=(n_restarts, self.dim))
        
        best_x = None
        best_f = np.inf
        
        # We will run Nelder-Mead from each start.
        # To manage budget, we allocate a sub-budget to each restart.
        # However, NM can stop early if it converges.
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            
            x0 = starts[i]
            x, f = self._nelder_mead(func, x0, self.budget // n_restarts, deadline)
            
            if f < best_f:
                best_f = f
                best_x = x
        
        # If we have time left, we might do a few more random restarts or refine best_x
        # But for now, returning the best found is the standard protocol.
        
        return best_x, best_f

    def _nelder_mead(self, func, x0, max_iter, deadline):
        """
        Standard Nelder-Mead implementation.
        """
        dim = self.dim
        x = x0.copy()
        f0 = func(x)
        
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x
        fvals = np.zeros(dim + 1)
        fvals[0] = f0
        
        # Step size for initial simplex expansion
        # Standard heuristic: 0.05 * |x| + 0.00025 if x==0
        # Here we use a fixed relative step based on domain size [-5, 5]
        step = 0.5 
        
        for i in range(1, dim + 1):
            simplex[i] = x.copy()
            simplex[i][i-1] += step
            # Clip to domain
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
            
            if time.time() > deadline:
                break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        best_f = fvals[0]
        best_x = simplex[0].copy()
        
        # NM Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        iter_count = 0
        while iter_count < max_iter:
            if time.time() > deadline:
                break
            
            # Check convergence: spread of function values is small
            # Or spread of points is small
            if np.max(fvals) - np.min(fvals) < 1e-8 * (1 + abs(np.min(fvals))):
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Reflection accepted
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                # Outside contraction if fr < fvals[-1], else inside
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
            
            # Re-sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            iter_count += 1
            
            # Early termination if simplex points are very close
            if np.max(simplex, axis=0) - np.min(simplex, axis=0) < 1e-8:
                break
                
        return best_x, best_f

# EVOLVE-BLOCK-END

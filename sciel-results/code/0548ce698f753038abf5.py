"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead simplex method for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # We will use a global evaluation counter
        eval_count = 0

        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            return fx

        # Strategy:
        # 1. Use first 10 evaluations for diverse initial centroids (one per simplex).
        # 2. Use remaining evaluations for aggressive Nelder-Mead refinement on the best basins.

        # Phase 1: Diverse initial sampling
        # Use stratified sampling to cover the space better
        centroids = []
        
        # Create a grid of initial points with some randomness
        n_initial = min(15, budget - 20)
        
        for i in range(n_initial):
            if eval_count >= budget:
                break
            
            if i < 5:
                # Center region - likely where optimum is
                x = rng.uniform(-1.0, 1.0, dim)
            elif i < 10:
                # Mid-range
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                # Outer region for coverage
                x = rng.uniform(-4.5, 4.5, dim)
            
            x = np.clip(x, -5.0, 5.0)
            fx = safe_eval(x)
            centroids.append((fx, x.copy()))

        # Sort by function value
        centroids.sort(key=lambda t: t[0])
        best_fx = centroids[0][0]
        best_x = centroids[0][1].copy()

        # Phase 2: Adaptive Nelder-Mead refinement
        remaining = budget - eval_count
        
        # Select top candidates for refinement
        top_k = min(6, len(centroids))
        
        # Allocate budget with emphasis on best candidates
        # Use exponential decay for allocation
        weights = [np.exp(-0.5 * i) for i in range(top_k)]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        for idx in range(top_k):
            if eval_count >= budget - 5:
                break
            
            center = centroids[idx][1].copy()
            alloc = int(remaining * weights[idx])
            alloc = max(alloc, dim + 1)
            
            # Adaptive step size: start smaller for better candidates
            if idx == 0:
                delta = 0.5
            elif idx == 1:
                delta = 0.8
            else:
                delta = 1.2
            
            # Initialize simplex
            simplex = [center.copy()]
            for j in range(dim):
                v = center.copy()
                v[j] += delta
                v = np.clip(v, -5.0, 5.0)
                simplex.append(v)
            
            # Evaluate simplex vertices
            fvals = []
            for v in simplex:
                if eval_count >= budget:
                    break
                fvals.append(safe_eval(v))
            
            if len(fvals) < dim + 1:
                continue
            
            # Run Nelder-Mead with adaptive parameters
            # Use more aggressive parameters for step functions
            gamma = 1.5  # Reflection parameter
            rho = 0.5    # Contraction parameter
            sigma = 0.5  # Shrink parameter
            
            max_iter = alloc // 2  # More iterations allowed
            
            for it in range(max_iter):
                if eval_count >= budget - 3:
                    break
                
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + gamma * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_eval(xr)
                
                if fr < fvals[0]:
                    # Expansion - try to go further in good direction
                    xe = centroid + 2.0 * gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    
                    if eval_count < budget:
                        fe = safe_eval(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    
                    if eval_count < budget:
                        fc = safe_eval(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink simplex
                            for j in range(1, dim + 1):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5.0, 5.0)
                                if eval_count < budget:
                                    fvals[j] = safe_eval(simplex[j])
                            if eval_count >= budget:
                                break
                    else:
                        break
                
                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_fx:
                    best_fx = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
                
                # Early termination if converged
                spread = max(fvals) - min(fvals)
                if spread < 1e-8 * (1 + abs(min(fvals))):
                    break
                
                # Also terminate if simplex is very small
                if np.max(np.abs(simplex[0] - simplex[-1])) < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END

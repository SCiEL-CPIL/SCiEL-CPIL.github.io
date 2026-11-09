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
        rng = np.random.default_rng(42)

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
        # 1. Use first 12 evaluations for diverse initial centroids (one per simplex).
        # 2. Use remaining evaluations for aggressive Nelder-Mead refinement on the best basins.

        # Phase 1: Seed 12 diverse initial centroids
        centroids = []
        n_seeds = 12
        for i in range(n_seeds):
            if eval_count >= budget:
                break
            # Spread initial guesses across the box
            # Mix of scales to cover different regions
            if i < 4:
                # Very close to center
                x = rng.uniform(-0.3, 0.3, dim)
            elif i < 8:
                # Moderate spread
                x = rng.uniform(-1.5, 1.5, dim)
            else:
                # Wide spread
                x = rng.uniform(-4.5, 4.5, dim)
            x = np.clip(x, -5.0, 5.0)
            fx = safe_eval(x)
            centroids.append((fx, x.copy()))

        # Sort by function value
        centroids.sort(key=lambda t: t[0])
        best_fx = centroids[0][0]
        best_x = centroids[0][1].copy()

        # Phase 2: Aggressive Nelder-Mead refinement on top candidates
        # We have 12 evals used, so 488 remaining.
        # For each centroid, we'll run NM with a limited number of iterations.
        # Allocate budget among top 6 centroids (most promising basins).
        
        top_k = min(6, len(centroids))
        remaining = budget - eval_count
        # Give more budget to better centroids
        # Distribute: 35%, 20%, 15%, 12%, 10%, 8% for top 6
        weights = [0.35, 0.20, 0.15, 0.12, 0.10, 0.08][:top_k]
        # Normalize weights to sum to 1
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        for idx in range(top_k):
            if eval_count >= budget - 15:  # Leave buffer
                break
            
            center = centroids[idx][1].copy()
            alloc = int(remaining * weights[idx])
            alloc = max(alloc, dim + 1)  # At least enough for one simplex
            
            # Initialize simplex around center with adaptive step size
            # Start with smaller step for finer search
            delta = 0.5
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
            
            # Run Nelder-Mead iterations with improved parameters
            # Each iteration uses at least 1 eval (reflection), up to 1+dim (shrink)
            # Estimate max iterations conservatively
            max_iter = alloc // 2  # More aggressive iteration count
            
            # Adaptive parameters for step ellipsoidal
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for it in range(max_iter):
                if eval_count >= budget - 2:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_eval(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
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
                    fc = safe_eval(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5.0, 5.0)
                            if eval_count >= budget:
                                break
                            fvals[j] = safe_eval(simplex[j])
                        if eval_count >= budget:
                            break
                
                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_fx:
                    best_fx = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
                
                # Check convergence - use tighter threshold for step functions
                if np.max(fvals) - np.min(fvals) < 1e-12:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: Rastrigin-aware lattice exploration, then Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_used = 0

        def timed_call(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        # Phase 1: Rastrigin-aware lattice exploration
        # Rastrigin f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Period is 2, so minima of each term are at even integers (0, ±2, ±4, ...)
        # With hidden shift/rotation, the lattice won't be perfectly aligned,
        # but evaluating at lattice points gives good basin coverage.
        
        best_x = None
        best_f = float('inf')
        
        # Generate lattice points: each coord in {-4, -2, 0, 2, 4}
        # 5^5 = 3125 points, but we only have ~150 calls for this phase
        # Use a subset: evaluate in order of "centrality" (closer to 0 first)
        
        # Actually, with shift/rotation, the optimal lattice may be offset.
        # Let's use a smarter approach: evaluate at a structured set of points.
        
        # Phase 1a: Evaluate at origin and single-axis lattice points
        lattice_points = []
        
        # Origin
        lattice_points.append(np.zeros(dim))
        
        # Single-axis: ±2, ±4 in each dimension
        for i in range(dim):
            for offset in [-4, -2, 2, 4]:
                p = np.zeros(dim)
                p[i] = offset
                lattice_points.append(p)
        
        # Two-axis combinations: ±2 in pairs
        for i in range(dim):
            for j in range(i + 1, dim):
                for oi in [-2, 2]:
                    for oj in [-2, 2]:
                        p = np.zeros(dim)
                        p[i] = oi
                        p[j] = oj
                        lattice_points.append(p)
        
        # Add some random points for diversity
        for _ in range(20):
            p = rng.uniform(-5, 5, dim)
            lattice_points.append(p)
        
        # Evaluate lattice points
        for p in lattice_points:
            if time.time() > deadline or calls_used >= 150:
                break
            f_val = timed_call(p)
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        # If no good point found, use random
        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = timed_call(best_x)
        
        # Phase 2: Nelder-Mead simplex method for local refinement
        # This is more efficient than (1+1)-ES for smooth local minima
        
        def nelder_mead(x0, fx0, max_iter=200):
            """Nelder-Mead simplex optimization."""
            nonlocal calls_used
            nonlocal deadline
            
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            
            # Create other vertices by perturbing in each dimension
            for i in range(n):
                v = x0.copy()
                if abs(x0[i]) > 0.5:
                    v[i] += 0.1 * x0[i]
                else:
                    v[i] += 0.1
                v = np.clip(v, -5, 5)
                simplex.append(v)
                fvals.append(timed_call(v))
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink
            
            for iteration in range(max_iter):
                if calls_used >= budget - 10 or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if fvals[-1] - fvals[0] < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = timed_call(xr)
                
                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = timed_call(xe)
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
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = timed_call(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = timed_call(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = timed_call(xc)
                        if fc <= fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = timed_call(simplex[i])
            
            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]
        
        # Run Nelder-Mead from the best point found in Phase 1
        x, fx = nelder_mead(best_x, best_f, max_iter=200)
        
        if fx < best_f:
            best_x, best_f = x, fx
        
        # Phase 3: Fine-tuning with small-step (1+1)-ES
        if calls_used < budget - 1 and time.time() < deadline:
            sigma = 0.05
            x = best_x.copy()
            fx = best_f
            
            for i in range(budget - calls_used):
                if calls_used >= budget - 1 or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = timed_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_x, best_f = x, fx
                else:
                    sigma *= 0.9
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END

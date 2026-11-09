"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive simplex for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285  # safety margin before 300s limit

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if calls >= budget:
                return np.inf
            f = func(x)
            if not np.isfinite(f):
                f = np.inf
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start exploration to find a good region
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Typical Rosenbrock minimum region
        start_points.append(np.ones(dim))
        
        # Random points
        for _ in range(12):
            start_points.append(rng.uniform(-5, 5, size=dim))
        
        # Points near (1,1,...,1) with small perturbations
        for _ in range(5):
            start_points.append(np.ones(dim) + rng.uniform(-0.5, 0.5, size=dim))
        
        # Evaluate starting points and keep track of best
        evaluated_starts = []
        for x0 in start_points:
            if calls >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = safe_call(x0)
            if f0 < np.inf:
                evaluated_starts.append((f0, x0.copy()))
        
        if not evaluated_starts:
            return best_x, best_f
        
        # Sort by function value
        evaluated_starts.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead local optimization from best starting points
        # Use the best few starting points for Nelder-Mead
        n_local_starts = min(3, len(evaluated_starts))
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage

        for i in range(n_local_starts):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = evaluated_starts[i][1]
            f0 = safe_call(x0) if evaluated_starts[i][0] == np.inf else evaluated_starts[i][0]
            
            # If we already have a value, use it; otherwise evaluate
            if f0 == np.inf:
                continue
            
            # Create initial simplex around the starting point
            # Use adaptive step size based on the local landscape
            step = 0.5
            
            simplex = [x0.copy()]
            fvals = [f0]
            
            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                x_new = x0.copy()
                if x0[d] + step <= 5.0:
                    x_new[d] += step
                else:
                    x_new[d] -= step
                f_new = safe_call(x_new)
                if f_new == np.inf:
                    break
                simplex.append(x_new)
                fvals.append(f_new)
            
            # If we couldn't build a full simplex, skip this start
            if len(simplex) < dim + 1:
                continue
            
            # Sort simplex by function value
            sorted_indices = np.argsort(fvals)
            simplex = [simplex[idx] for idx in sorted_indices]
            fvals = [fvals[idx] for idx in sorted_indices]
            
            # Nelder-Mead iteration
            max_iters = 150
            for _ in range(max_iters):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Compute centroid of all points except the worst
                centroid = np.sum(simplex[:dim], axis=0) / dim
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[dim])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_call(x_r)
                if f_r == np.inf:
                    break
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[dim])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_call(x_e)
                    if f_e == np.inf:
                        # Expansion failed, use reflection
                        simplex[dim] = x_r
                        fvals[dim] = f_r
                    elif f_e < f_r:
                        simplex[dim] = x_e
                        fvals[dim] = f_e
                    else:
                        simplex[dim] = x_r
                        fvals[dim] = f_r
                elif f_r < fvals[dim - 1]:
                    # Accept reflection
                    simplex[dim] = x_r
                    fvals[dim] = f_r
                else:
                    # Contraction
                    if f_r < fvals[dim]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_call(x_c)
                    if f_c == np.inf:
                        break
                    
                    if f_c < min(f_r, fvals[dim]):
                        simplex[dim] = x_c
                        fvals[dim] = f_c
                    else:
                        # Shrinkage
                        shrink_ok = True
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            f_j = safe_call(simplex[j])
                            if f_j == np.inf:
                                shrink_ok = False
                                break
                            fvals[j] = f_j
                        if not shrink_ok:
                            break
                
                # Re-sort simplex
                sorted_indices = np.argsort(fvals)
                simplex = [simplex[idx] for idx in sorted_indices]
                fvals = [fvals[idx] for idx in sorted_indices]
                
                # Check if we're stuck
                if np.max(fvals) - np.min(fvals) < 1e-14:
                    break
        
        # Phase 3: Fine-tuning with small random perturbations around best
        if calls < budget and time.time() < deadline:
            sigma = 0.05
            for _ in range(30):
                if calls >= budget or time.time() > deadline:
                    break
                x_try = best_x + sigma * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = safe_call(x_try)
                if f_try < best_f:
                    sigma *= 1.3
                else:
                    sigma *= 0.7
                    if sigma < 1e-6:
                        break

        return best_x, best_f

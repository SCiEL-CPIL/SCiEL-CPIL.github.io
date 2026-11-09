"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with model-based initialization and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Coarse grid/random sampling to build a local model ---
        # Sample points to estimate the landscape. We'll use a mix of random and grid.
        n_samples = 30  # keep small to allow many optimization steps
        
        # Start with a few random points and a small grid
        points = []
        # Random
        for _ in range(10):
            points.append(rng.uniform(-4, 4, dim))
        # Simple grid around center
        # For dim=5, a full grid is expensive. Use a sparse set.
        # Center is (0,0,0,0,0)
        points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for d in [1, 2, -1, -2]:
                p = np.zeros(dim)
                p[i] = d
                points.append(p.copy())
        
        # Evaluate these
        evals = 0
        values = []
        for p in points:
            if evals >= budget:
                break
            fp = func(np.clip(p, -5, 5))
            values.append(fp)
            evals += 1
        
        if evals >= budget:
            return np.array(points[0]), values[0] if values else 0.0

        # Find best so far
        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx])
        best_f = values[best_idx]

        # --- Phase 2: Local optimization using a simple CMA-ES-like approach ---
        # Use the best point as center, estimate sigma from the spread of good points
        
        # Sort points by fitness
        sorted_idx = np.argsort(values)
        # Use top few points to estimate local scale
        top_k = min(5, len(values))
        top_points = np.array([points[i] for i in sorted_idx[:top_k]])
        
        # Estimate sigma as the average distance from best point to other top points
        if len(top_points) > 1:
            distances = np.linalg.norm(top_points[1:] - top_points[0], axis=1)
            sigma = max(0.1, np.mean(distances) * 0.5)
        else:
            sigma = 1.0

        # Adaptive step size CMA-ES-like loop
        lambda_pop = 6  # population size per iteration
        # We'll do a simple (lambda+1)-ES with step-size adaptation
        
        x_mean = best_x.copy()
        c = 0.9  # learning rate for step size
        # Track success rate for 1/5 rule
        success_count = 0
        window = 10
        successes = []

        while evals < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_pop):
                if evals >= budget:
                    break
                # Sample
                y = x_mean + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)
            
            if not offspring:
                break
            
            # Select best offspring
            best_off_idx = int(np.argmin(offspring_f))
            best_off_x = offspring[best_off_idx]
            best_off_f = offspring_f[best_off_idx]
            
            # Update success
            if best_off_f < best_f:
                successes.append(1)
                # Accept
                x_mean = best_off_x
                best_f = best_off_f
                # Adapt sigma up
                sigma *= 1.1
            else:
                successes.append(0)
                # Adapt sigma down
                sigma *= 0.9
            
            # Trim window
            if len(successes) > window:
                successes.pop(0)
            
            # 1/5 rule adjustment (applied less frequently to avoid oscillation)
            if len(successes) == window:
                rate = np.mean(successes)
                if rate > 0.2:
                    sigma *= 1.05
                elif rate < 0.1:
                    sigma *= 0.95

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)

            # Check time again
            if time.time() > deadline:
                break

        # --- Phase 3: Final high-precision polish pass ---
        # Use remaining budget for a tighter local search
        remaining_budget = budget - evals
        if remaining_budget > 10 and time.time() < deadline - 5:
            # Nelder-Mead-like local refinement
            x_best = x_mean.copy()
            f_best = best_f
            
            # Create initial simplex
            simplex = [x_best.copy()]
            for i in range(dim):
                p = x_best.copy()
                # Small step in each direction
                step = 0.01 * max(1.0, abs(x_best[i]))
                p[i] += step
                simplex.append(p.copy())
            
            # Evaluate simplex
            simplex_f = []
            for p in simplex:
                if evals >= budget:
                    break
                fp = func(np.clip(p, -5, 5))
                simplex_f.append(fp)
                evals += 1
            
            if len(simplex_f) == dim + 1 and evals < budget:
                # Nelder-Mead iterations
                max_nm_iters = min(50, (budget - evals) // (dim + 1))
                for _ in range(max_nm_iters):
                    if evals >= budget or time.time() > deadline:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    alpha = 1.0
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    if evals < budget:
                        f_r = func(x_r)
                        evals += 1
                    else:
                        break
                    
                    if f_r < simplex_f[0]:
                        # Expansion
                        gamma = 2.0
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5, 5)
                        if evals < budget:
                            f_e = func(x_e)
                            evals += 1
                        else:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contraction
                        rho = 0.5
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        if evals < budget:
                            f_c = func(x_c)
                            evals += 1
                        else:
                            break
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                if evals < budget:
                                    simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                                    evals += 1
                                else:
                                    break
                            if evals >= budget:
                                break
                    
                    # Update best
                    best_in_simplex = int(np.argmin(simplex_f))
                    if simplex_f[best_in_simplex] < f_best:
                        f_best = simplex_f[best_in_simplex]
                        x_best = simplex[best_in_simplex].copy()
                
                # Update global best
                if f_best < best_f:
                    best_f = f_best
                    x_mean = x_best.copy()

        return x_mean, best_f
# EVOLVE-BLOCK-END

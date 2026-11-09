"""Multi-start Nelder-Mead optimizer for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse Initialization
        n_init = 40
        init_results = []
        
        # Halton sequence for low discrepancy
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init + 1):
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d]))
            point = np.array(point) * 10.0 - 5.0
            # Add small jitter to avoid exact duplicates or issues
            point += rng.uniform(-0.01, 0.01, dim)
            point = np.clip(point, -5, 5)
            
            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(point)
            init_results.append((point, f))

        # Sort by fitness
        init_results.sort(key=lambda x: x[1])
        
        # Phase 2: Local Optimization with Nelder-Mead
        # Take top K starts
        K = min(3, len(init_results))
        if K == 0:
            return best_x, best_f
            
        top_starts = init_results[:K]
        remaining_budget = budget - calls_made
        budget_per_start = max(10, remaining_budget // K)

        for x0, f0 in top_starts:
            if time.time() > deadline or calls_made >= budget:
                break
                
            # Run Nelder-Mead
            # Simplex initialization
            # Center: x0
            # Other vertices: x0 + delta * e_i
            delta = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate simplex (we already have f0 for x0, need dim more)
            # But evaluate() updates global best, so it's fine.
            # However, we need the fitness values for the simplex points.
            # Let's evaluate the new points.
            simplex_f = [f0]
            for i in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    break
                f_new = evaluate(simplex[i+1])
                simplex_f.append(f_new)
            
            # If we ran out of budget during simplex init, break
            if len(simplex_f) < dim + 1:
                continue

            # Nelder-Mead Loop
            max_iter = budget_per_start // (dim + 1) # Approx iterations
            max_iter = min(max_iter, 50) # Cap iterations to avoid infinite loops if budget is large
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            no_improve_count = 0
            min_f = min(simplex_f)
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break
                    
                # Sort simplex by fitness
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Check convergence: max distance in simplex
                max_dist = 0
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        d = np.linalg.norm(simplex[i] - simplex[j])
                        if d > max_dist:
                            max_dist = d
                if max_dist < 1e-6:
                    break
                    
                # Centroid of best N points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    no_improve_count = 0
                elif fr < simplex_f[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    no_improve_count = 0
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (centroid - simplex[-1])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        no_improve_count = 0
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])
                        no_improve_count += 1
                        
                if no_improve_count > 10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END

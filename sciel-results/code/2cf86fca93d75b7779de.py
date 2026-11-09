import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            # Ensure x is a valid numpy array of correct shape
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape((dim,))
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(func, x0, max_iter=50, timeout_check=None):
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = evaluate(x0)
            simplex_f = [f0]
            
            # Create initial simplex
            for i in range(dim):
                x_new = x0.copy()
                # Step size: 0.05 if x0[i] != 0 else 0.05
                step = 0.05 if x0[i] == 0 else 0.05 * x0[i]
                x_new[i] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                simplex.append(x_new)
                simplex_f.append(f_new)
            
            if calls_made >= budget:
                return x0, f0

            # Nelder-Mead Parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                if timeout_check and time.time() > timeout_check:
                    break
                
                # Check convergence
                if abs(simplex_f[-1] - simplex_f[0]) < 1e-12:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = evaluate(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = evaluate(xe)
                    
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = evaluate(xc)
                        
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrinkage
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = evaluate(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = evaluate(xc)
                        
                        if fc <= simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrinkage
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = evaluate(simplex[i])
                
                # Re-sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                if calls_made >= budget:
                    break

            return simplex[0], simplex_f[0]

        # Multi-start Strategy
        # Phase 1: Race diverse starts to find a good basin
        # We use a portion of the budget for this.
        # Let's try 5 starts, using 20% of budget each? Or just let them run until time/budget.
        # Given 500 budget, 5 starts * 100 calls each is reasonable.
        # Or, run a few quick starts, pick best, then refine.
        
        num_starts = 5
        # Randomly generate starting points in [-4, 4] to stay within bounds comfortably
        # But f7 is often centered or shifted. Uniform in [-4,4] is safe.
        starts = []
        for _ in range(num_starts):
            x_start = np.random.uniform(-4, 4, size=dim)
            starts.append(x_start)
        
        # Add origin as a start (common for these functions)
        starts.append(np.zeros(dim))
        
        # Evaluate starts
        start_results = []
        for x_s in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            f_s = evaluate(x_s)
            start_results.append((x_s, f_s))
            
        if not start_results:
            return best_x, best_f

        # Sort starts by fitness
        start_results.sort(key=lambda x: x[1])
        
        # Refine the best starts
        # Allocate remaining budget to refining the top few starts
        # Since Nelder-Mead is fast, we can try refining the top 3 starts.
        # However, we must be careful with budget.
        
        # Let's just refine the single best start found so far with the remaining budget.
        # This is a "Race then Refine" strategy.
        
        best_start_x, best_start_f = start_results[0]
        
        # If we have enough budget left, run Nelder-Mead on the best start
        # Nelder-Mead uses ~dim+1 calls per iteration.
        # With 500 calls, and 6 calls for starts, we have ~490 left.
        # 490 / 6 = ~80 iterations.
        
        if calls_made < budget:
            final_x, final_f = nelder_mead(func, best_start_x, max_iter=100, timeout_check=deadline)
        
        # Optionally, if we have budget left, try the second best start?
        # For now, stick to one refinement to avoid wasting budget on a potentially worse basin.
        # But if the first one stagnates, the second might be better.
        # Given the "Step" nature, finding the right basin is key.
        # The race (6 starts) should help.
        
        return best_x, best_f

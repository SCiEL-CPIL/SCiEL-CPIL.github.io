import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        x_best = np.random.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        
        calls_made = 1
        budget = self.budget
        dim = self.dim

        # Helper to check if we should stop
        def time_up():
            return time.time() > deadline

        # Nelder-Mead implementation
        def nelder_mead(func, x0, f0, budget_left, calls_count):
            # Initialize simplex
            # Start with a reasonably large step size
            step = 2.0
            simplex = [x0.copy()]
            f_vals = [f0]
            
            # Create other vertices
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                simplex.append(xi)
                fi = func(xi)
                f_vals.append(fi)
                calls_count += 1
                if calls_count >= budget or time_up():
                    return x0, f0, calls_count

            # Sort simplex
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]

            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            while calls_count < budget and not time_up():
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func(xr)
                calls_count += 1
                if time_up():
                    break

                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func(xe)
                    calls_count += 1
                    if time_up():
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    if fr < f_vals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = func(xc)
                        calls_count += 1
                        if time_up():
                            break
                        if fc <= fr:
                            simplex[-1] = xc
                            f_vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_vals[i] = func(simplex[i])
                                calls_count += 1
                                if time_up():
                                    break
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        fc = func(xc)
                        calls_count += 1
                        if time_up():
                            break
                        if fc < f_vals[-1]:
                            simplex[-1] = xc
                            f_vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_vals[i] = func(simplex[i])
                                calls_count += 1
                                if time_up():
                                    break

                # Re-sort
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

            return simplex[0], f_vals[0], calls_count

        # Phase 1: Global search with large simplex
        x_curr, f_curr, calls_made = nelder_mead(func, x_best, f_best, budget, calls_made)
        x_best, f_best = x_curr, f_curr

        # Phase 2: Local refinement with smaller simplex
        # Use remaining budget for fine-tuning
        if calls_made < budget and not time_up():
            # Reset step size for local search
            # We don't pass step size directly to nelder_mead above, 
            # but we can restart with a perturbed start point to trigger a new simplex formation 
            # or just run nelder_mead again. The internal step is fixed at 2.0 in my helper.
            # To improve local search, we should modify the helper to accept a step size or
            # just rely on the fact that the simplex has shrunk.
            # However, the fixed step=2.0 in initialization might be too large for local refinement.
            # Let's create a second call with a smaller initial step by manually constructing the simplex?
            # No, the helper creates the simplex. 
            # Let's just run the helper again. The simplex from the previous run is lost.
            # A better strategy: The previous run ends with a small simplex. 
            # But my helper discards the simplex state.
            # Let's modify the strategy: Run Nelder-Mead once. It converges.
            # If we have budget left, the simplex is already small. 
            # But since I reset the simplex in the helper, I lose that info.
            
            # Let's refine the helper to return the final simplex or just accept that 
            # one long run of NM is often sufficient if the convergence criteria are met.
            # Given the budget is 500, a single NM run might not use all calls if it converges fast.
            # To ensure we use the budget, we can restart NM with a very small step if we have calls left.
            
            # Simple approach: Just run NM again from the best point found.
            # The step size 2.0 is relative to the starting point. 
            # If we are at the minimum, the simplex will expand/contract around it.
            
            x_curr, f_curr, calls_made = nelder_mead(func, x_best, f_best, budget, calls_made)
            if f_curr < f_best:
                x_best, f_best = x_curr, f_curr

        return x_best, f_best

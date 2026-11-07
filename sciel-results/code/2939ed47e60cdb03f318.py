import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f11 (Discus).
    Strategy:
    1. Derivative-free Nelder-Mead (from Inspiration 1) for robust global search.
       This avoids the budget explosion of CMA-ES (Inspiration 2) and the
       high evaluation cost of Sobol sampling (Current Program).
    2. Nelder-Mead is used as the primary engine because:
       - It is derivative-free.
       - It adapts to the local landscape (valleys) effectively.
       - It has a predictable, low evaluation count per step (1-4 calls).
       - It is robust to shifts and rotations.
    3. We implement a manual Nelder-Mead with strict budget and time guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink

        # Initial simplex
        # Start at 0, or random? 0 is often good for centered functions.
        # f11 (Discus) is usually centered but shifted/rotated.
        # Random start within box might be safer if shift is large, but 0 is in [-4,4] usually.
        # Let's use a small random perturbation from 0 to avoid symmetry issues if any.
        rng = np.random.default_rng(42)
        
        # We will try a few different starting points if the first one fails to converge quickly?
        # No, let's stick to one robust run. The Nelder-Mead simplex itself provides exploration.
        # However, to be safe against bad local minima, we can start with a slightly larger spread.
        
        x0 = rng.uniform(-2, 2, dim)
        
        # Build simplex
        simplex = []
        f_values = []
        
        # First point
        x_curr = x0.copy()
        f_curr = func(x_curr)
        simplex.append(x_curr)
        f_values.append(f_curr)
        calls += 1
        if f_curr < best_f:
            best_f = f_curr
            best_x = x_curr.copy()
            
        # Other points
        # Standard Nelder-Mead initial step size is 0.05 or 0.25
        # Box is [-5, 5]. Let's use 0.5 step.
        step = 0.5
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            x_new = x0.copy()
            x_new[i] += step
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            simplex.append(x_new)
            f_values.append(f_new)
            calls += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                
        if calls >= budget or time.time() > deadline:
            if best_x is None:
                best_x = np.zeros(dim)
            return best_x, best_f

        # Nelder-Mead Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            # We maintain simplex and f_values sorted
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Check convergence (optional, but good for stopping)
            # If spread is small, stop
            if np.std(f_values) < 1e-10:
                break
            # Also stop if the simplex is very small in space
            if np.linalg.norm(simplex[0] - simplex[-1]) < 1e-8:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_idx = len(simplex) - 1
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[worst_idx])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
                
            if calls >= budget:
                break
                
            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[worst_idx])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                    
                if calls >= budget:
                    break
                    
                if f_e < f_r:
                    simplex[worst_idx] = x_e
                    f_values[worst_idx] = f_e
                else:
                    simplex[worst_idx] = x_r
                    f_values[worst_idx] = f_r
            elif f_r < f_values[worst_idx - 1]:
                # Simple Reflection
                simplex[worst_idx] = x_r
                f_values[worst_idx] = f_r
            else:
                # Contraction
                # Outside contraction
                x_c = centroid + rho * (simplex[worst_idx] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                    
                if calls >= budget:
                    break
                    
                if f_c < f_values[worst_idx]:
                    simplex[worst_idx] = x_c
                    f_values[worst_idx] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if calls >= budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if f_values[i] < best_f:
                            best_f = f_values[i]
                            best_x = simplex[i].copy()
                    if calls >= budget:
                        break

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls < budget:
                best_f = func(best_x)

        return best_x, best_f

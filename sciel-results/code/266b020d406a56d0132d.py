import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Global Random Search followed by Nelder-Mead Local Refinement.
    Designed for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: Evaluator kills at 330s. We target finishing well before.
        deadline = time.time() + 290.0
        
        best_x = None
        best_f = float('inf')
        
        calls = 0
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            try:
                return func(x)
            except:
                return None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Global Exploration with Random Sampling
        # Use a significant portion of the budget to find a good starting point for local search.
        # For f13 (Sharp Ridge), the global landscape can be deceptive. 
        # A good initial point is crucial for Nelder-Mead to converge to the global minimum.
        n_global = budget // 2  # Use 50% of budget for global search
        
        for i in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            # Sample points uniformly in the box
            x = rng.uniform(-5, 5, size=dim)
            f = safe_call(x)
            if f is None:
                break
            update_best(x, f)
            
        if calls >= budget or best_x is None:
            return best_x, best_f

        # Phase 2: Local Refinement with Nelder-Mead
        # Start Nelder-Mead from the best point found in Phase 1.
        # We need to construct a simplex around best_x.
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Initialize simplex around the best point
        # We need dim+1 points. One is best_x. The others are best_x +/- step along axes.
        # We need to be careful with budget. We have budget - calls left.
        # Simplex initialization costs dim evaluations (one for each axis direction).
        
        # Choose a step size. Since we did global search, we can start smaller.
        # But if the best point is near a boundary or on a ridge, we might need to be careful.
        # Let's start with a moderate step size, say 1.0.
        step = 1.0
        
        simplex = [best_x.copy()]
        f_values = [best_f]
        
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = best_x.copy()
            # Alternate directions to stay within bounds if possible, or just clip
            # Since func clips, we can just add step and let it clip if needed, 
            # but it's better to stay inside.
            # If best_x[i] + step > 5, we might go out of bounds. 
            # The problem statement says func clips to [-5, 5]. So it's safe.
            xi[i] += step
            fi = safe_call(xi)
            if fi is None:
                break
            simplex.append(xi)
            f_values.append(fi)
            update_best(xi, fi)
            
        if len(simplex) < dim + 1:
            return best_x, best_f
            
        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # Main Nelder-Mead loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Update global best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                
            # Check for convergence/stagnation
            # If simplex is too small or values are very close
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < 1e-6 or np.max(f_values) - np.min(f_values) < 1e-10:
                # Restart with a larger step or random restart
                # Shrink the simplex towards the best point and then expand?
                # Or just restart with a new random point near best
                # Let's try expanding the simplex from the best point
                center = simplex[0].copy()
                simplex_new = [center]
                f_values_new = [f_values[0]]
                step = 0.5 # Smaller step for refinement
                for i in range(dim):
                    xi = center.copy()
                    xi[i] += step
                    fi = safe_call(xi)
                    if fi is None:
                        break
                    simplex_new.append(xi)
                    f_values_new.append(fi)
                if len(simplex_new) < dim + 1:
                    break
                simplex = np.array(simplex_new)
                f_values = np.array(f_values_new)
                continue

            # Standard Nelder-Mead steps
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                # Reflection accepted
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

        return best_x, best_f

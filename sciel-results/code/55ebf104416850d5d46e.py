import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for BBOB-style functions:
    - Phase 1: Adaptive random search with restarts
    - Phase 2: Nelder-Mead local refinement
    Designed for ill-conditioned functions like Schaffers F7.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Adaptive random search with multi-restart
        # Use a decreasing step size strategy
        base_sigma = 3.0  # Start wide
        min_sigma = 0.01
        n_restarts = 3
        calls_per_restart = max(10, budget // (n_restarts + 2))
        
        for restart in range(n_restarts):
            if time.time() > deadline or calls >= budget - 50:  # Reserve calls for Nelder-Mead
                break
            
            # New random center
            x_center = rng.uniform(-4, 4, d)
            f_center = evaluate(x_center)
            
            # Adaptive step size: start large, shrink if no improvement
            sigma = base_sigma * (0.7 ** restart)
            no_improve_count = 0
            max_no_improve = 15
            
            while calls < budget - 50 and time.time() < deadline:
                # Generate candidate
                x_trial = x_center + sigma * rng.standard_normal(d)
                f_trial = evaluate(x_trial)
                
                if f_trial < f_center:
                    # Accept: move center, keep or slightly increase sigma
                    x_center = x_trial.copy()
                    f_center = f_trial
                    sigma *= 1.1  # Expand a bit
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Shrink sigma if stuck
                    if no_improve_count >= max_no_improve:
                        sigma *= 0.7
                        no_improve_count = 0
                        if sigma < min_sigma:
                            break  # This restart is exhausted
            
            # Check deadline
            if time.time() > deadline:
                break
        
        # Phase 2: Nelder-Mead local refinement from best point
        if best_x is not None and calls < budget - 10 and time.time() < deadline:
            x_nm = best_x.copy()
            f_nm = best_f
            
            # Initialize simplex
            simplex = [x_nm.copy()]
            for i in range(d):
                x_temp = x_nm.copy()
                step = 0.1 if abs(x_nm[i]) > 1e-8 else 0.1
                x_temp[i] += step
                simplex.append(x_temp)
            
            # Evaluate simplex vertices
            for i, x_v in enumerate(simplex):
                if calls >= budget or time.time() > deadline:
                    break
                f_v = evaluate(x_v)
                simplex[i] = (f_v, x_v.copy())
            
            # Sort by function value
            simplex.sort(key=lambda t: t[0])
            
            # Nelder-Mead iterations
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma_nm = 0.5  # Shrink
            
            while calls < budget and time.time() < deadline:
                # Compute centroid of best d vertices (excluding worst)
                centroid = np.mean([s[1] for s in simplex[:-1]], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1][1])
                f_ref = evaluate(x_ref)
                
                if f_ref < simplex[0][0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = (f_exp, x_exp.copy())
                    else:
                        simplex[-1] = (f_ref, x_ref.copy())
                elif f_ref < simplex[-2][0]:
                    # Accept reflection
                    simplex[-1] = (f_ref, x_ref.copy())
                else:
                    # Contraction
                    if f_ref < simplex[-1][0]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1][1] - centroid)
                    f_con = evaluate(x_con)
                    
                    if f_con < min(f_ref, simplex[-1][0]):
                        simplex[-1] = (f_con, x_con.copy())
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i][1] = simplex[0][1] + sigma_nm * (simplex[i][1] - simplex[0][1])
                            if calls < budget and time.time() < deadline:
                                simplex[i][0] = evaluate(simplex[i][1])
                        simplex.sort(key=lambda t: t[0])
                
                # Check convergence
                if len(simplex) > 1:
                    f_range = abs(simplex[0][0] - simplex[-1][0])
                    if f_range < 1e-10:
                        break
        
        return best_x, best_f

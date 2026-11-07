import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0
        eval_history = []  # Track (fx, x) for all evaluations

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            eval_history.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim
        budget = self.budget
        dim = self.dim

        # Phase 1: Global exploration - diverse starting points
        # Mix of structured and random points to identify basins
        starts = []
        starts.append(np.zeros(dim))
        # Structured points at key coordinates
        for val in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
            pt = np.full(dim, val)
            starts.append(pt)
        # Random points in [-4, 4]^5
        for _ in range(15):
            starts.append(rng.uniform(-4.0, 4.0, dim))
        # Points near the expected minimizer region
        for _ in range(5):
            starts.append(rng.uniform(-2.0, 2.0, dim))
        
        # Evaluate all starting points (but keep ~400 calls for optimization)
        for x0 in starts:
            if call_count >= budget - 400:
                break
            tracked_func(x0)
        
        # Sort evaluated points by function value
        eval_history.sort(key=lambda t: t[0])
        
        # Phase 2: Adaptive local optimization with Nelder-Mead
        # Use Nelder-Mead which is efficient for Rosenbrock's narrow valley
        # with adaptive restarts when progress stalls
        
        def nelder_mead_local(x0, max_calls, tol=1e-12):
            """Run Nelder-Mead with call budget and stagnation detection."""
            nonlocal call_count, best_fx, best_x
            if call_count >= budget - 5 or not time_ok():
                return
            
            # Initial simplex
            x0 = np.asarray(x0, dtype=float)
            x0 = np.clip(x0, -5.0, 5.0)
            fx0 = tracked_func(x0)
            
            if call_count >= budget - 5 or not time_ok():
                return
            
            # Build initial simplex
            simplex = [x0.copy()]
            simplex_fx = [fx0]
            
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 if x0[i] >= 0 else -0.1
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fxi = tracked_func(xi)
                simplex.append(xi.copy())
                simplex_fx.append(fxi)
                
                if call_count >= budget - 5 or not time_ok():
                    return
            
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            prev_best = best_fx
            stall_count = 0
            max_stall = 30  # Restart if no improvement for this many iterations
            
            while call_count < budget - 5 and time_ok():
                # Sort simplex by function value
                order = np.argsort(simplex_fx)
                simplex = [simplex[i] for i in order]
                simplex_fx = [simplex_fx[i] for i in order]
                
                # Check for convergence
                if abs(simplex_fx[0] - simplex_fx[-1]) < tol:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = tracked_func(x_ref)
                
                if call_count >= budget - 5 or not time_ok():
                    break
                
                if f_ref < simplex_fx[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = tracked_func(x_exp)
                    
                    if call_count >= budget - 5 or not time_ok():
                        break
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_fx[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_fx[-1] = f_ref
                elif f_ref < simplex_fx[-2]:
                    simplex[-1] = x_ref
                    simplex_fx[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = tracked_func(x_con)
                    
                    if call_count >= budget - 5 or not time_ok():
                        break
                    
                    if f_con < simplex_fx[-1]:
                        simplex[-1] = x_con
                        simplex_fx[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_fx[i] = tracked_func(simplex[i])
                            if call_count >= budget - 5 or not time_ok():
                                return
                        
                        # Update best
                        min_idx = np.argmin(simplex_fx)
                        if simplex_fx[min_idx] < best_fx:
                            best_fx = simplex_fx[min_idx]
                            best_x = simplex[min_idx].copy()
                        
                        break  # Shrink means we're converging or stuck
            
            # Update best
            if simplex_fx[0] < best_fx:
                best_fx = simplex_fx[0]
                best_x = simplex[0].copy()
            
            # Detect stagnation
            if best_fx < prev_best - 1e-10:
                stall_count = 0
            else:
                stall_count += 1
            
            return stall_count

        # Run local optimization from top-k starting points
        top_k = min(5, len(eval_history))
        used_starts = set()
        
        for k in range(top_k):
            if call_count >= budget - 20 or not time_ok():
                break
            
            # Get the k-th best starting point
            fx_k, x_k = eval_history[k]
            x_k_key = tuple(np.round(x_k, 6))
            
            if x_k_key in used_starts:
                continue
            used_starts.add(x_k_key)
            
            # Run Nelder-Mead from this point
            max_calls_for_this = min(budget - call_count - 10, 150)
            if max_calls_for_this < 20:
                break
            
            stall = nelder_mead_local(x_k, max_calls_for_this)
            
            # If stalled and we have budget, try a perturbation
            if stall >= 20 and call_count < budget - 30 and time_ok():
                # Perturb the best point
                x_pert = best_x + rng.normal(0, 0.5, dim)
                x_pert = np.clip(x_pert, -4.0, 4.0)
                tracked_func(x_pert)
                
                if call_count < budget - 10 and time_ok():
                    nelder_mead_local(x_pert, budget - call_count - 5)
        
        # Phase 3: Final polish with L-BFGS-B for high precision
        if time_ok() and call_count < budget - 3:
            remaining = budget - call_count - 3
            if remaining > 10:
                try:
                    minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 200,
                            'maxfun': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-12,
                            'maxls': 30
                        }
                    )
                except Exception:
                    pass
        
        # Phase 4: One more Nelder-Mead polish if budget remains
        if time_ok() and call_count < budget - 5:
            remaining = budget - call_count - 5
            if remaining > 15:
                nelder_mead_local(best_x, remaining, tol=1e-14)
        
        return best_x, best_fx

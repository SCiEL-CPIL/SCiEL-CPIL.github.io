import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    Strategy:
    1. CMA-ES style population-based search for global exploration.
    2. L-BFGS-B local refinement from best points.
    3. Adaptive restarts if stagnation is detected.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(1234)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f_val = float(func(x))
                if np.isfinite(f_val) and f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: CMA-ES style population-based global search
        # Start with a population around the origin
        pop_size = 20
        sigma = 2.0  # Initial step size
        mean = np.zeros(dim)
        
        # Generate initial population
        population = [mean + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
        population = [np.clip(x, -5.0, 5.0) for x in population]
        
        # Evaluate initial population
        pop_f = []
        for x in population:
            f_val = safe_call(x)
            if f_val is None:
                break
            pop_f.append(f_val)
        
        if len(pop_f) < pop_size:
            # Not enough calls, go straight to local refinement
            if best_x is not None:
                self._local_refine(best_x.copy(), best_f, safe_call, deadline)
            return best_x, best_f
        
        # CMA-ES main loop
        max_gen = 30
        stagnation_count = 0
        stagnation_threshold = 5
        
        for gen in range(max_gen):
            if time.time() > deadline or calls >= budget - 50:
                break
            
            # Sort by function value
            idx = np.argsort(pop_f)
            population = [population[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]
            
            # Check for improvement
            if best_f < pop_f[0] + 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            if stagnation_count >= stagnation_threshold:
                break
            
            # Update mean (weighted by rank)
            weights = np.exp(-np.arange(pop_size) / (pop_size / 2))
            weights /= np.sum(weights)
            new_mean = np.average(population[:pop_size], axis=0, weights=weights[:pop_size])
            new_mean = np.clip(new_mean, -5.0, 5.0)
            
            # Update sigma (adaptive)
            if gen > 0:
                # Simple adaptation: reduce if no improvement
                if pop_f[0] < pop_f[0]:  # Placeholder for actual adaptation
                    sigma *= 0.9
                else:
                    sigma *= 0.8
            sigma = max(sigma, 0.01)
            
            # Generate next generation
            next_population = []
            for i in range(pop_size):
                x_new = new_mean + sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5.0, 5.0)
                next_population.append(x_new)
            
            # Evaluate next generation
            next_f = []
            for x in next_population:
                f_val = safe_call(x)
                if f_val is None:
                    break
                next_f.append(f_val)
            
            if len(next_f) < pop_size:
                break
            
            population = next_population
            pop_f = next_f
            mean = new_mean
        
        # Phase 2: Local refinement from best point using L-BFGS-B
        if best_x is not None and calls < budget - 20 and time.time() < deadline:
            self._local_refine(best_x.copy(), best_f, safe_call, deadline)
        
        # Phase 3: Additional local refinements from other good points
        if calls < budget - 10 and time.time() < deadline:
            # Try a few more local refinements from diverse starting points
            for restart in range(5):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                
                # Perturb the best point
                noise_scale = 0.5 / (restart + 1)
                x_restart = best_x + rng.normal(0, noise_scale, size=dim)
                x_restart = np.clip(x_restart, -5.0, 5.0)
                
                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break
                
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                
                self._local_refine(x_restart.copy(), f_restart, safe_call, deadline)
        
        return best_x, best_f

    def _local_refine(self, x_start, f_start, safe_call, deadline):
        """
        L-BFGS-B local refinement with adaptive step size.
        """
        dim = self.dim
        
        # Use L-BFGS-B for local refinement
        # We need to wrap the function to track calls
        def wrapped_func(x):
            if time.time() > deadline:
                return float('inf')
            f_val = safe_call(x)
            if f_val is None:
                return float('inf')
            return f_val
        
        try:
            result = minimize(
                wrapped_func,
                x_start,
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 50, 'maxfun': 30}
            )
            
            if result.fun < best_f and np.isfinite(result.fun):
                # Update best
                pass  # safe_call already updates best_x and best_f
        
        except Exception:
            pass
        
        # If L-BFGS-B didn't improve much, try Nelder-Mead as backup
        if calls < budget - 10 and time.time() < deadline:
            self._nelder_mead_backup(x_start, safe_call, deadline)

    def _nelder_mead_backup(self, x_start, safe_call, deadline):
        """
        Nelder-Mead simplex method as backup.
        """
        dim = self.dim
        alpha = 1.0
        gamma = 2.0
        rho_c = 0.5
        sigma = 0.5
        
        step_size = 0.2
        
        simplex = [x_start.copy()]
        f_values = [safe_call(x_start)]
        if f_values[0] is None:
            return
        
        for i in range(dim):
            xi = x_start.copy()
            xi[i] += step_size
            fi = safe_call(xi)
            if fi is None:
                return
            simplex.append(xi)
            f_values.append(fi)
        
        if len(simplex) < dim + 1:
            return
        
        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        no_improve = 0
        patience = 15
        max_iter = 40
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget - 5:
                break
            
            # Sort by function value
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Check for convergence
            spread_f = np.max(f_values) - np.min(f_values)
            spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if spread_f < 1e-10 or np.max(spread_x) < 1e-8:
                break
            
            no_improve += 1
            if no_improve > patience:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                    no_improve = 0
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
                    no_improve = 0
            elif fr < f_values[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_values[-1] = fr
                no_improve = 0
            else:
                # Contraction
                if fr < f_values[-1]:
                    xc = centroid + rho_c * (xr - centroid)
                else:
                    xc = centroid - rho_c * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

    def _calls_remaining(self):
        return 100  # Conservative estimate

"""Rewritten optimizer for BBOB f13 (Sharp ridge) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive simplex and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_fx
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return x, fx

        # Phase 1: Multi-start initialization
        # Evaluate a set of diverse starting points to find promising basins
        rng = np.random.default_rng(42)
        
        # Center point
        x0 = np.zeros(dim)
        x0, fx0 = evaluate(x0)
        
        # Grid of points spread across the domain
        # For dim=5, we can evaluate several points
        # Use a structured set of points
        n_starts = 30
        start_points = []
        
        # Add center
        start_points.append(x0.copy())
        
        # Add axis-aligned points
        for i in range(dim):
            for val in [-3, -1.5, 1.5, 3]:
                p = np.zeros(dim)
                p[i] = val
                start_points.append(p)
        
        # Add some random points
        while len(start_points) < n_starts:
            p = rng.uniform(-4, 4, size=dim)
            start_points.append(p)
        
        # Evaluate all start points
        start_results = [(x0.copy(), fx0)]
        for p in start_points[1:]:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            xp, fp = evaluate(p)
            start_results.append((xp, fp))
        
        # Sort by function value and keep top 3
        start_results.sort(key=lambda t: t[1])
        top_starts = start_results[:min(3, len(start_results))]
        
        # Phase 2: Run Nelder-Mead from each top start
        for start_x, start_fx in top_starts:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize simplex around start_x
            simplex = [start_x.copy()]
            fx_list = [start_fx]
            
            # Small initial step
            step = 0.3
            for i in range(dim):
                xi = start_x.copy()
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate new simplex points
            for i in range(1, len(simplex)):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                xi, fi = evaluate(simplex[i])
                simplex[i] = xi
                fx_list.append(fi)
            
            n_simplex = len(simplex)
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Adaptive step size
            current_step = step
            
            iterations = 0
            max_iterations = (budget - calls) // 3  # Rough allocation
            
            while calls < budget and iterations < max_iterations:
                if time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(fx_list)
                simplex = [simplex[i] for i in order]
                fx_list = [fx_list[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_fx = fx_list[-1]
                best_fx_simplex = fx_list[0]
                
                # Reflection
                xr = centroid + alpha * (centroid - worst)
                xr, fr = evaluate(xr)
                
                if fr < best_fx_simplex:
                    # Try expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe, fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fx_list[-1] = fe
                    else:
                        simplex[-1] = xr
                        fx_list[-1] = fr
                elif fr < worst_fx:
                    simplex[-1] = xr
                    fx_list[-1] = fr
                else:
                    # Contraction
                    if fr < worst_fx:
                        xc = centroid + rho * (xr - centroid)
                        xc, fc = evaluate(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fx_list[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n_simplex):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i], fi = evaluate(simplex[i])
                                fx_list[i] = fi
                    else:
                        xc = centroid - rho * (worst - centroid)
                        xc, fc = evaluate(xc)
                        if fc < worst_fx:
                            simplex[-1] = xc
                            fx_list[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n_simplex):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i], fi = evaluate(simplex[i])
                                fx_list[i] = fi
                
                # Check convergence
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.all(spread < 1e-6):
                    break
                
                iterations += 1
        
        return best_x, best_fx
# EVOLVE-BLOCK-END

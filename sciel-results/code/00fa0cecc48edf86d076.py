"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f9 (rotated Rosenbrock) in 5D.
    
    Strategy:
    1. Sparse global lattice search to identify the basin of attraction.
    2. Nelder-Mead simplex method exploiting smoothness for rapid convergence.
    3. L-BFGS-B for final polishing.
    4. Strict budget management with early termination on convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 45s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)
        call_count = 0
        best_x = np.zeros(self.dim)
        best_fx = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x_arr = np.asarray(x, dtype=float)
            fx = float(func(x_arr))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_arr.copy()
            return fx

        # Phase 1: Sparse global lattice search
        # Use a stratified random sampling approach to cover the space efficiently
        # Sample from a grid with some randomization to avoid missing the basin
        lattice_budget = int(self.budget * 0.15)
        
        # Create a stratified sample: divide each dimension into bins
        # For 5D, use 2-3 points per dimension for a reasonable grid
        points_per_dim = 3
        grid = []
        for i in range(points_per_dim ** self.dim):
            coords = []
            for d in range(self.dim):
                # Map index to grid position
                idx = (i // (points_per_dim ** d)) % points_per_dim
                # Center of bin with some randomization
                center = -5.0 + (idx + 0.5) * (10.0 / points_per_dim)
                # Add small random perturbation
                jitter = rng.uniform(-1.0, 1.0) * (10.0 / points_per_dim) * 0.3
                coords.append(np.clip(center + jitter, -5.0, 5.0))
            grid.append(np.array(coords))
        
        # Shuffle and evaluate
        rng.shuffle(grid)
        for x in grid:
            if not time_ok() or call_count >= lattice_budget:
                break
            tracked_func(x)
        
        # Additional random samples to fill budget
        while call_count < lattice_budget and time_ok():
            x = rng.uniform(-5, 5, self.dim)
            tracked_func(x)

        # Phase 2: Nelder-Mead from the best point
        # Nelder-Mead is derivative-free and robust for smooth functions
        if time_ok() and call_count < self.budget - 50:
            x_start = best_x.copy()
            
            # Implement Nelder-Mead manually for better budget control
            dim = self.dim
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrinkage
            
            # Initialize simplex
            simplex = [x_start.copy()]
            f_simplex = [best_fx]
            
            # Add perturbed points
            for d in range(dim):
                x_new = x_start.copy()
                step = 0.5 if x_start[d] > -4.5 else 0.5
                x_new[d] += step
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = tracked_func(x_new)
                simplex.append(x_new)
                f_simplex.append(f_new)
            
            nm_iterations = 0
            max_nm_iter = 200
            
            while nm_iterations < max_nm_iter and call_count < self.budget - 30 and time_ok():
                nm_iterations += 1
                
                # Sort simplex by function value
                idx = np.argsort(f_simplex)
                simplex = [simplex[i].copy() for i in idx]
                f_simplex = [f_simplex[i] for i in idx]
                
                # Check convergence
                if abs(f_simplex[-1] - f_simplex[0]) < 1e-12:
                    break
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = tracked_func(x_r)
                
                if f_simplex[0] <= f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                elif f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = tracked_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = tracked_func(x_c)
                    
                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_simplex[i] = tracked_func(simplex[i])
        
        # Phase 3: L-BFGS-B for polishing
        if time_ok() and call_count < self.budget - 10:
            x_start = best_x.copy()
            
            # Implement a simple gradient approximation for L-BFGS
            def approx_grad(x):
                grad = np.zeros(self.dim)
                eps = 1e-7
                for d in range(self.dim):
                    x_plus = x.copy()
                    x_plus[d] += eps
                    x_minus = x.copy()
                    x_minus[d] -= eps
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    grad[d] = (tracked_func(x_plus) - tracked_func(x_minus)) / (2 * eps)
                return grad
            
            # Use scipy's L-BFGS-B with gradient
            from scipy.optimize import minimize
            
            lbfgs_budget = self.budget - call_count - 5
            try:
                res = minimize(
                    tracked_func,
                    x_start,
                    jac=approx_grad,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfun': lbfgs_budget,
                        'ftol': 1e-14,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass

        # Phase 4: (1+1)-ES polish if budget remains
        if time_ok() and call_count < self.budget - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.05
            stuck = 0
            remaining = self.budget - call_count - 5

            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    stuck = 0
                else:
                    sigma *= 0.85
                    stuck += 1

                if stuck > 30:
                    x = np.clip(best_x + 0.1 * rng.standard_normal(self.dim), -5, 5)
                    fx = tracked_func(x)
                    sigma = 0.02
                    stuck = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END

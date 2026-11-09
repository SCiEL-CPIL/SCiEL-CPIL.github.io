import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with Halton quasi-random sequence
        # Halton sequence provides low-discrepancy coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_point(i, dim):
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point = []
            for d in range(dim):
                point.append(halton(i, primes[d]))
            return np.array(point) * 10.0 - 5.0  # Map to [-5, 5]

        # Use 40 Halton points for global coverage
        n_halton = 40
        for i in range(1, n_halton + 1):
            if time.time() > deadline:
                break
            x = halton_point(i, n)
            try_func(x)

        # Also add a few random points to complement
        for _ in range(10):
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, n)
            try_func(x)

        # Phase 2: Local refinement with Nelder-Mead on best point
        # Initialize simplex with geometrically stretched base to align with ridge
        if calls < budget - 20 and time.time() < deadline:
            # Create an elongated simplex to match sharp ridge anisotropy
            # Start from best_x and stretch in one direction
            base_size = 1.0
            stretch = 5.0  # Stretch factor for ridge alignment
            
            # Find a good stretching direction from recent evaluations
            # Use the direction from origin to best_x if it's not at origin
            if np.linalg.norm(best_x) > 0.1:
                stretch_dir = best_x / np.linalg.norm(best_x)
            else:
                stretch_dir = rng.standard_normal(n)
                stretch_dir /= np.linalg.norm(stretch_dir)
            
            # Create simplex: center at best_x, one vertex stretched
            simplex = []
            f_simplex = []
            
            # Center point
            simplex.append(best_x.copy())
            f_simplex.append(best_f)
            
            # Create n other vertices, with one stretched along stretch_dir
            for i in range(n):
                # Create a direction perpendicular to stretch_dir (approximately)
                d = rng.standard_normal(n)
                d -= np.dot(d, stretch_dir) * stretch_dir  # Project out stretch component
                if np.linalg.norm(d) < 1e-6:
                    d = rng.standard_normal(n)
                d /= np.linalg.norm(d)
                
                # For the first vertex, use stretched direction
                if i == 0:
                    x_new = best_x + stretch * base_size * stretch_dir
                else:
                    x_new = best_x + base_size * d
                
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = try_func(x_new)
                simplex.append(x_new)
                f_simplex.append(f_new)
            
            if len(simplex) == n + 1:
                # Run Nelder-Mead manually with the initialized simplex
                # This avoids scipy's overhead and lets us control the simplex
                max_iter = 150
                alpha = 1.0  # Reflection
                gamma = 2.0  # Expansion
                rho = 0.5    # Contraction
                sigma = 0.5  # Shrink
                
                for _ in range(max_iter):
                    if time.time() > deadline or calls >= budget:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order]
                    f_simplex = [f_simplex[i] for i in order]
                    
                    # Check convergence
                    if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                        break
                    
                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    worst_x = simplex[-1]
                    worst_f = f_simplex[-1]
                    
                    # Reflection
                    x_ref = centroid + alpha * (centroid - worst_x)
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = try_func(x_ref)
                    
                    if f_ref < f_simplex[0]:
                        # Expansion
                        x_exp = centroid + gamma * (centroid - worst_x)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = try_func(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_simplex[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            f_simplex[-1] = f_ref
                    elif f_ref < f_simplex[-2]:
                        # Just accept reflection
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho * (worst_x - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = try_func(x_con)
                        if f_con < worst_f:
                            simplex[-1] = x_con
                            f_simplex[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_simplex[i] = try_func(simplex[i])
                    
                    # Update best
                    if f_simplex[0] < best_f:
                        best_f = f_simplex[0]
                        best_x = simplex[0].copy()

        # Phase 3: Final polish with L-BFGS-B
        if calls < budget - 5 and time.time() < deadline:
            def obj(x):
                return try_func(x)
            
            try:
                res = minimize(obj, best_x, method='L-BFGS-B',
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 30, 'gtol': 1e-8, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f

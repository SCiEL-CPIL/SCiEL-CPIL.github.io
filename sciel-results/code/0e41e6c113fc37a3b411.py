# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer combining Nelder-Mead simplex with (1+1)-ES.
    Nelder-Mead exploits the smooth local structure of Griewank-Rosenbrock
    via geometric reflections, while (1+1)-ES provides robust global exploration.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Stage 1: Coarse random sampling to locate a good basin quickly
        n_coarse = min(10, budget)
        for _ in range(n_coarse):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            safe_call(x)

        if calls_made >= budget or best_x is None:
            return best_x, best_f

        # Stage 2: Nelder-Mead simplex optimization from the best point
        # Initialize simplex around the best point
        x0 = best_x.copy()
        f0 = best_f
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        # Create additional simplex vertices with small perturbations
        delta = 0.5
        for i in range(dim):
            v = x0.copy()
            v[i] += delta
            v = np.clip(v, -5.0, 5.0)
            fv = safe_call(v)
            simplex.append(v)
            simplex_f.append(fv)
            if calls_made >= budget or time.time() > deadline:
                break

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrinkage

        nm_iterations = 0
        max_nm_iterations = 200
        no_improve_nm = 0
        max_no_improve_nm = 30

        while calls_made < budget and time.time() < deadline and nm_iterations < max_nm_iterations:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check for convergence
            f_range = simplex_f[-1] - simplex_f[0]
            if f_range < 1e-10:
                break

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                no_improve_nm = 0
            elif fr < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
                no_improve_nm = 0
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                    no_improve_nm = 0
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = safe_call(simplex[i])
                    no_improve_nm += 1

            nm_iterations += 1
            
            # If Nelder-Mead stagnates, fall back to (1+1)-ES
            if no_improve_nm >= max_no_improve_nm:
                break

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Stage 3: (1+1)-ES refinement from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 0.3  # Start with smaller sigma for fine-tuning

        while calls_made < budget and time.time() < deadline:
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = safe_call(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Success: increase step size
                sigma *= 1.3
            else:
                # Failure: decrease step size
                sigma *= 0.85

            # Safety bounds for sigma
            if sigma < 1e-8:
                sigma = 1e-8
            if sigma > 5.0:
                sigma = 5.0

        return best_x, best_f
# EVOLVE-BLOCK-END

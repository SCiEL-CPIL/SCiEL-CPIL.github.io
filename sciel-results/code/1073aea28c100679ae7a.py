import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f # Should not happen, but safe
            fx = func(np.clip(x, -5, 5))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5).copy()
            return fx

        # Phase 1: Initialization
        # Start at 0
        x0 = np.zeros(dim)
        f0 = call_func(x0)
        
        # Simple random sampling to find a better start if 0 is bad
        # We use a small number of random points to ensure we don't waste budget
        # on a bad local start (though Sphere is unimodal, shift/rotation might move min)
        # Since it's a shifted sphere, the minimum is somewhere in [-4,4]^5.
        # 0 is a reasonable guess, but let's check a few random points.
        for _ in range(10):
            if calls_made >= budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-4.5, 4.5, dim)
            f_rand = call_func(x_rand)
            if f_rand < f0:
                x0 = x_rand
                f0 = f_rand
        
        # Phase 2: Nelder-Mead
        # Initialize simplex
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        # Step size for initial simplex
        # If we are far from optimum, larger step. If close, smaller.
        # For Sphere, 1.0 is a good default step for [-5,5] box.
        step = 1.0
        
        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            v = x0.copy()
            v[i] += step
            fv = call_func(v)
            simplex.append(v.copy())
            simplex_f.append(fv)

        # Nelder-Mead Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = 400 # Will be limited by budget anyway
        
        for _ in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Convergence check
            if abs(simplex_f[-1] - simplex_f[0]) < 1e-10:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = call_func(x_ref)
            
            if f_ref < simplex_f[0]:
                # Expansion
                if calls_made < budget and time.time() < deadline:
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = call_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Reflection accepted
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (centroid - simplex[-1])
                
                if calls_made < budget and time.time() < deadline:
                    f_con = call_func(x_con)
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = call_func(simplex[i])
                else:
                    # Out of budget, stop
                    break

        return best_x, best_f

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Random exploration + custom BFGS with FD gradients + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng()

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        def make_func_wrapper(fallback_val):
            def wrapper(x):
                nonlocal calls_made
                if calls_made >= budget or not time_ok():
                    return fallback_val
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                try_update(x_clipped, fx)
                return fx
            return wrapper

        # Phase 1: Coarse random exploration with more samples
        n_explore = 30
        seeds = []
        
        for i in range(n_explore):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            seeds.append((x.copy(), fx))

        if not seeds:
            return best_x, best_f

        # Sort seeds by function value
        seeds.sort(key=lambda s: s[1])
        
        # Keep top diverse seeds (at most 4)
        max_seeds = 4
        if len(seeds) > max_seeds:
            top = [seeds[0]]
            for s in seeds[1:]:
                if len(top) >= max_seeds:
                    break
                diverse = True
                for ts in top:
                    if np.linalg.norm(s[0] - ts[0]) < 2.0:
                        diverse = False
                        break
                if diverse:
                    top.append(s)
            seeds = top

        # Custom BFGS with finite-difference gradient estimation
        def fd_gradient(x, wrapper, eps=1e-5):
            """Forward-difference gradient estimate."""
            g = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = wrapper(x_plus)
                g[i] = (f_plus - wrapper(x)) / eps
            return g

        def custom_bfgs(x0, wrapper, max_iter=50):
            """Custom BFGS quasi-Newton with FD gradients and box constraints."""
            x = x0.copy()
            H_inv = np.eye(dim)  # Inverse Hessian approximation
            f_cur = wrapper(x)
            g_cur = fd_gradient(x, wrapper)
            
            for it in range(max_iter):
                if not time_ok() or calls_made >= budget:
                    break
                
                # Check gradient norm
                g_norm = np.linalg.norm(g_cur)
                if g_norm < 1e-8:
                    break
                
                # Compute search direction
                d = -H_inv @ g_cur
                
                # Line search (backtracking)
                alpha = 1.0
                c1 = 1e-4
                for _ in range(30):
                    x_new = np.clip(x + alpha * d, -5, 5)
                    f_new = wrapper(x_new)
                    if f_new <= f_cur + c1 * alpha * (d @ g_cur):
                        break
                    alpha *= 0.5
                else:
                    break
                
                # Compute new gradient
                g_new = fd_gradient(x_new, wrapper)
                
                # BFGS update
                s = x_new - x
                y = g_new - g_cur
                sy = s @ y
                if sy > 1e-12:
                    H_inv = H_inv + ((1 + sy / (y @ H_inv @ y)) * np.outer(s, s) / (s @ y) 
                                     - np.outer(s, H_inv @ y) / (s @ y) 
                                     - np.outer(H_inv @ y, s) / (s @ y))
                
                x = x_new
                f_cur = f_new
                g_cur = g_new
            
            return x, f_cur

        # Phase 2: Custom BFGS local optimization from each seed
        for sx, sf in seeds:
            if not time_ok() or calls_made >= budget:
                break
            
            wrapper = make_func_wrapper(sf)
            try:
                x_res, f_res = custom_bfgs(sx, wrapper, max_iter=40)
                if f_res < best_f:
                    best_x, best_f = x_res.copy(), f_res
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish from best point
        if best_x is not None and calls_made < budget:
            wrapper = make_func_wrapper(best_f)
            try:
                from scipy.optimize import minimize
                result = minimize(
                    wrapper,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

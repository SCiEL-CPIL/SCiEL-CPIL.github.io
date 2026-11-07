"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead + coordinate descent for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 285  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = float(func(x))
            calls_made += 1
            return x, f

        # Phase 1: Nelder-Mead simplex method
        # Initialize simplex with dim+1 points
        rng = np.random.default_rng(42)
        
        # Start from center of box
        center = np.zeros(dim)
        f_center, _ = call_func(center)
        
        # Build simplex: center plus dim perturbed points
        simplex = [center.copy()]
        simplex_f = [f_center]
        
        step_size = 1.0
        for i in range(dim):
            point = center.copy()
            point[i] += step_size
            x_pt, f_pt = call_func(point)
            simplex.append(x_pt)
            simplex_f.append(f_pt)
        
        # Update global best
        best_idx = np.argmin(simplex_f)
        best_x = simplex[best_idx].copy()
        best_f = simplex_f[best_idx]
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5 # shrinkage
        
        max_nm_iterations = 200
        nm_iter = 0
        
        while calls_made < total_budget and time.time() < deadline and nm_iter < max_nm_iterations:
            nm_iter += 1
            
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Update global best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
            
            # Check for convergence
            if simplex_f[-1] - simplex_f[0] < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            worst = simplex[-1]
            x_ref = centroid + alpha * (centroid - worst)
            x_ref, f_ref = call_func(x_ref)
            
            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp, f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                    x_con, f_con = call_func(x_con)
                    if f_con <= f_ref:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            x_sh, f_sh = call_func(simplex[i])
                            simplex[i] = x_sh
                            simplex_f[i] = f_sh
                else:
                    # Inside contraction
                    x_con = centroid + 0.5 * (worst - centroid)
                    x_con, f_con = call_func(x_con)
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            x_sh, f_sh = call_func(simplex[i])
                            simplex[i] = x_sh
                            simplex_f[i] = f_sh
        
        # Phase 2: Coordinate descent / line search refinement
        # Use remaining budget for local refinement
        if best_x is not None and calls_made < total_budget:
            x_cur = best_x.copy()
            f_cur = best_f
            step = 0.5
            
            for cd_iter in range(50):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        x_try = x_cur.copy()
                        x_try[d] += direction * step
                        x_try, f_try = call_func(x_try)
                        if f_try < f_cur:
                            x_cur = x_try
                            f_cur = f_try
                            improved = True
                            # Try to continue in same direction
                            for _ in range(3):
                                if calls_made >= total_budget or time.time() > deadline:
                                    break
                                x_next = x_cur.copy()
                                x_next[d] += direction * step
                                x_next, f_next = call_func(x_next)
                                if f_next < f_cur:
                                    x_cur = x_next
                                    f_cur = f_next
                                else:
                                    break
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
                # Update global best
                if f_cur < best_f:
                    best_f = f_cur
                    best_x = x_cur.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END

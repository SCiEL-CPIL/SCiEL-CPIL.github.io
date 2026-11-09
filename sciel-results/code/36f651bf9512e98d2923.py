"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive Multi-Start Nelder-Mead with LHS exploration and high-precision polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def lhs_sample(n, dim):
            """Latin Hypercube Sampling for stratified exploration."""
            samples = np.zeros((n, dim))
            for i in range(dim):
                perm = np.random.permutation(n)
                samples[:, i] = (perm + np.random.random(n)) / n
            return samples * 10.0 - 5.0  # Map to [-5, 5]

        def nelder_mead(x0, f0, max_iter=80):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline or calls_made >= budget - 10:
                return x0, f0
            
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Standard NM initialization
            for i in range(dim):
                new_x = x0.copy()
                step = 0.5
                if abs(x0[i]) < 0.1:
                    new_x[i] += step
                else:
                    new_x[i] += -np.sign(x0[i]) * step * 0.5
                new_x = np.clip(new_x, -5, 5)
                if np.allclose(new_x, x0):
                    new_x[i] += 0.1 * (1 if x0[i] < 0 else -1)
                f_new = safe_call(new_x)
                simplex.append(new_x)
                simplex_f.append(f_new)
                
                if time.time() > deadline or calls_made >= budget - 10:
                    return best_x, best_f

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget - 10:
                    break
                
                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Check for convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10 * (1 + abs(simplex_f[0])):
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (x_ref - centroid)
                        
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_call(x_con)
                    
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_call(simplex[i])
                            
                # Update global best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Phase 1: Initial Exploration with LHS + key points
        # 1. Origin
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        
        # 2. Axis-aligned points
        for i in range(dim):
            for sign in [1, -1]:
                if time.time() > deadline or calls_made >= budget - 100:
                    break
                x = np.zeros(dim)
                x[i] = sign * 3.0
                f = safe_call(x)
                if f < f0:
                    x0 = x
                    f0 = f
        
        # 3. LHS sampling for better coverage
        n_lhs = 30
        lhs_points = lhs_sample(n_lhs, dim)
        for x in lhs_points:
            if time.time() > deadline or calls_made >= budget - 150:
                break
            f = safe_call(x)
            if f < f0:
                x0 = x
                f0 = f
        
        # Phase 2: Local Refinement from best point
        nelder_mead(x0, f0)
        
        # Phase 3: Adaptive Multi-Start with more diverse restarts
        n_starts = 8
        for i in range(n_starts):
            if time.time() > deadline or calls_made >= budget - 50:
                break
            
            if i < n_starts // 2:
                # Local perturbations of best
                scale = 1.5 / (i + 1)
                x_new = best_x + np.random.normal(0, scale, dim)
            else:
                # Semi-global restarts
                x_new = best_x * 0.3 + np.random.uniform(-3, 3, dim)
                
            x_new = np.clip(x_new, -5, 5)
            f_new = safe_call(x_new)
            
            if f_new < best_f:
                nelder_mead(x_new, f_new)
        
        # Phase 4: High-precision polish with L-BFGS-B if budget allows
        if time.time() < deadline - 5 and calls_made < budget - 20 and best_x is not None:
            try:
                # Use remaining budget for polish
                polish_budget = min(budget - calls_made - 5, 100)
                if polish_budget > 10:
                    result = minimize(
                        func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': polish_budget, 'maxls': 20}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END

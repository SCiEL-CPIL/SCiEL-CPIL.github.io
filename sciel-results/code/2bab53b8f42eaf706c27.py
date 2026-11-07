import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        def _check_time():
            return time.time() - start_time < timeout
        
        # 1. Sobol Scan
        # Use scipy.stats.qmc.Sobol if available, else generate manually or use random
        # Since scipy is available, we use it.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True)
            # 100 points
            n_sobol = 100
            if budget > n_sobol:
                sobol_samples = sampler.random(n_sobol)
                # Map to [-5, 5]
                sobol_samples = sobol_samples * 10 - 5
            else:
                sobol_samples = np.random.rand(budget, d) * 10 - 5
                n_sobol = budget
        except ImportError:
            # Fallback to random if scipy fails
            n_sobol = min(100, budget)
            sobol_samples = np.random.rand(n_sobol, d) * 10 - 5

        best_x = None
        best_f = np.inf
        n_evals = 0
        
        # Evaluate Sobol points
        for x in sobol_samples:
            if not _check_time():
                break
            f_val = func(x)
            n_evals += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            n_evals += 1

        # 2. Nelder-Mead Refinement
        if n_evals < budget and _check_time():
            # Initialize simplex
            # Center at best_x
            # Size: 0.5 is a good start for [-5,5] box after Sobol
            alpha = 0.5
            simplex = np.zeros((d + 1, d))
            simplex[0] = best_x
            for i in range(d):
                simplex[i + 1] = best_x
                simplex[i + 1, i] += alpha
                
            f_simplex = np.array([func(x) for x in simplex])
            n_evals += d + 1
            
            # Nelder-Mead parameters
            rho = 1.0  # Reflection
            chi = 2.0  # Expansion
            gamma = 0.5 # Contraction
            sigma = 0.5 # Shrink
            
            max_iter = 200
            it = 0
            
            while n_evals + d + 1 <= budget and it < max_iter and _check_time():
                # Sort simplex
                idx = np.argsort(f_simplex)
                simplex = simplex[idx]
                f_simplex = f_simplex[idx]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = x_centroid + rho * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = func(x_reflect)
                n_evals += 1
                
                if f_reflect < f_simplex[0]:
                    # Expansion
                    x_expand = x_centroid + chi * (x_reflect - x_centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    n_evals += 1
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_simplex[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_simplex[-1] = f_reflect
                elif f_reflect < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    f_simplex[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = x_centroid + gamma * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    n_evals += 1
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        f_simplex[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            n_evals += 1
                            if not _check_time():
                                break
                        if not _check_time():
                            break
                
                # Update best
                min_idx = np.argmin(f_simplex)
                if f_simplex[min_idx] < best_f:
                    best_f = f_simplex[min_idx]
                    best_x = simplex[min_idx].copy()
                    
                it += 1

        return best_x, best_f

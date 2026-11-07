"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multistart Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Parameters
        num_starts = 10
        
        # 1. Generate initial candidates
        starts = rng.uniform(-4.5, 4.5, size=(num_starts, dim))
        fitnesses = np.array([func(x) for x in starts])
        call_count = num_starts
        
        # Sort starts by fitness
        order = np.argsort(fitnesses)
        starts = starts[order]
        fitnesses = fitnesses[order]
        
        global_best_x = starts[0].copy()
        global_best_f = fitnesses[0]
        
        # 2. Run Nelder-Mead on each start, sequentially.
        # Allocate budget to ensure all starts get a chance.
        max_evals_per_nm = max(10, (budget - call_count) // num_starts)
        
        for i in range(num_starts):
            if call_count >= budget:
                break
            
            x0 = starts[i].copy()
            f0 = fitnesses[i]
            
            # Initialize Simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            
            # Set initial step size
            step = 0.1
            for j in range(1, dim + 1):
                simplex[j] = x0.copy()
                simplex[j][j-1] += step
            
            # Evaluate initial simplex
            simplex_f = np.zeros(dim + 1)
            simplex_f[0] = f0
            
            # We need to evaluate the other dim vertices.
            if call_count + dim > budget:
                continue
                
            for j in range(1, dim + 1):
                simplex_f[j] = func(simplex[j])
                call_count += 1
                if simplex_f[j] < global_best_f:
                    global_best_f = simplex_f[j]
                    global_best_x = simplex[j].copy()
            
            # Nelder-Mead Loop
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            nm_evals = 0
            while call_count < budget and nm_evals < max_evals_per_nm:
                # Sort simplex by fitness
                order_simplex = np.argsort(simplex_f)
                simplex = simplex[order_simplex]
                simplex_f = simplex_f[order_simplex]
                
                # Check convergence
                if simplex_f[-1] - simplex_f[0] < 1e-8:
                    break
                
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                call_count += 1
                nm_evals += 1
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    if call_count >= budget or nm_evals >= max_evals_per_nm:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                        break
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    call_count += 1
                    nm_evals += 1
                    
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
                        if call_count >= budget or nm_evals >= max_evals_per_nm:
                            break
                        x_con = centroid + rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        call_count += 1
                        nm_evals += 1
                        
                        if f_con <= f_ref:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                if call_count >= budget or nm_evals >= max_evals_per_nm:
                                    break
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex_f[j] = func(simplex[j])
                                call_count += 1
                                nm_evals += 1
                    else:
                        # Inside contraction
                        if call_count >= budget or nm_evals >= max_evals_per_nm:
                            break
                        x_con = centroid - rho * (centroid - simplex[-1])
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        call_count += 1
                        nm_evals += 1
                        
                        if f_con < simplex_f[-1]:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                if call_count >= budget or nm_evals >= max_evals_per_nm:
                                    break
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex_f[j] = func(simplex[j])
                                call_count += 1
                                nm_evals += 1
                
                # Update global best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < global_best_f:
                    global_best_f = simplex_f[min_idx]
                    global_best_x = simplex[min_idx].copy()
            
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END

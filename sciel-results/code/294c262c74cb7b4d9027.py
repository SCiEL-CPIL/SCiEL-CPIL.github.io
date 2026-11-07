"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        n = self.dim
        budget = self.budget
        
        # Parameters
        n_restarts = 10
        sub_budget = budget // n_restarts
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        epsilon = 0.5  # Increased to help escape local minima
        
        best_f = np.inf
        best_x = np.zeros(n)
        
        for _ in range(n_restarts):
            if self._call_count >= budget:
                break
                
            # Random start
            x0 = np.random.uniform(-5, 5, n)
            
            # Initialize simplex
            simplex = []
            fitness = []
            
            # Point 0: x0
            x_curr = np.clip(x0, -5, 5)
            simplex.append(x_curr)
            f_curr = func(x_curr)
            self._call_count += 1
            fitness.append(f_curr)
            
            # Points 1..n: x0 + epsilon * e_i
            for i in range(n):
                x_i = x_curr.copy()
                x_i[i] += epsilon
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
                f_i = func(x_i)
                self._call_count += 1
                fitness.append(f_i)
                
            # If we ran out of budget during init
            if self._call_count >= budget:
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < best_f:
                    best_f = fitness[min_idx]
                    best_x = simplex[min_idx]
                continue

            # Nelder-Mead Loop
            while self._call_count < budget:
                # Sort simplex
                order = np.argsort(fitness)
                simplex = [simplex[i] for i in order]
                fitness = [fitness[i] for i in order]
                
                # Check convergence: if spread is small
                spread = np.max(fitness) - np.min(fitness)
                if spread < 1e-8:
                    break
                    
                # Centroid of best n points
                centroid = np.mean(simplex[:n], axis=0)
                
                # Worst point
                x_w = simplex[n]
                f_w = fitness[n]
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_w)
                x_r = np.clip(x_r, -5, 5)
                
                if self._call_count >= budget: break
                f_r = func(x_r)
                self._call_count += 1
                
                if f_r < fitness[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - x_w)
                    x_e = np.clip(x_e, -5, 5)
                    if self._call_count >= budget: break
                    f_e = func(x_e)
                    self._call_count += 1
                    
                    if f_e < f_r:
                        simplex[n] = x_e
                        fitness[n] = f_e
                    else:
                        simplex[n] = x_r
                        fitness[n] = f_r
                        
                elif f_r < fitness[n-1]:
                    # Accept reflection
                    simplex[n] = x_r
                    fitness[n] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_w - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if self._call_count >= budget: break
                    f_c = func(x_c)
                    self._call_count += 1
                    
                    if f_c < f_w:
                        simplex[n] = x_c
                        fitness[n] = f_c
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if self._call_count >= budget: break
                            f_s = func(simplex[i])
                            self._call_count += 1
                            fitness[i] = f_s
                        if self._call_count >= budget: break
                        
                # Update global best
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < best_f:
                    best_f = fitness[min_idx]
                    best_x = simplex[min_idx]
                    
        return best_x, best_f

# EVOLVE-BLOCK-END

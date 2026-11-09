import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = 320.0  # 10s margin before 330s kill

    def __call__(self, func):
        self.start_time = time.time()
        self.func = func
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Multi-start exploration
        n_starts = 10
        calls_per_start = 20
        
        rng = np.random.default_rng(42)
        
        best_x_all = None
        best_f_all = float('inf')
        
        for i in range(n_starts):
            # Check time
            if time.time() > self.deadline - 30: # Leave 30s for phase 2
                break
            if self.calls >= budget - 50:
                break

            # Random start in [-4, 4]^dim
            x0 = rng.uniform(-4, 4, dim)
            
            # Run a short Nelder-Mead
            x_best, f_best = self._nelder_mead(func, x0, calls_per_start, dim, step_init=1.0)
            
            if f_best < best_f_all:
                best_f_all = f_best
                best_x_all = x_best
                self.best_x = x_best
                self.best_f = f_best

        # Phase 2: Refinement
        # Use remaining budget for a long NM run on the best point
        if best_x_all is not None:
            remaining_budget = budget - self.calls
            if remaining_budget > 10:
                # Check time
                if time.time() < self.deadline:
                    # Use a smaller step for refinement
                    x_best, f_best = self._nelder_mead(func, best_x_all, remaining_budget, dim, step_init=0.5)
                    if f_best < self.best_f:
                        self.best_x = x_best
                        self.best_f = f_best

        # If no best found, return a random point
        if self.best_x is None:
             self.best_x = np.zeros(dim)
             self.best_f = func(self.best_x)

        return self.best_x, self.best_f

    def _nelder_mead(self, func, x0, max_iters, dim, step_init=1.0):
        """
        Custom Nelder-Mead implementation with stall detection.
        """
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        f0 = func(x0)
        self.calls += 1
        simplex_f = [f0]
        
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += step_init
            simplex_f.append(func(simplex[i]))
            self.calls += 1
            
        # Sort simplex
        order = np.argsort(simplex_f)
        simplex = simplex[order]
        simplex_f = [simplex_f[i] for i in order]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        best_f_local = simplex_f[0]
        best_x_local = simplex[0].copy()
        
        stall_count = 0
        no_improve_threshold = 15
        
        for it in range(max_iters):
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break
                
            # Convergence check
            if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                break
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            self.calls += 1
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                self.calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                self.calls += 1
                
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex_f[i] = func(simplex[i])
                        self.calls += 1
                        if self.calls >= self.budget:
                            break
                    if self.calls >= self.budget:
                        break
                    
            # Re-sort
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = [simplex_f[i] for i in order]
            
            if simplex_f[0] < best_f_local:
                best_f_local = simplex_f[0]
                best_x_local = simplex[0].copy()
                stall_count = 0
            else:
                stall_count += 1
                
            # Update global best
            if best_f_local < self.best_f:
                self.best_f = best_f_local
                self.best_x = best_x_local

            # Stall detection: Random kick
            if stall_count >= no_improve_threshold:
                # Perturb the best point
                rng = np.random.default_rng(hash(tuple(simplex[0])) % 2**32)
                kick = rng.normal(0, 0.5, dim)
                new_x = np.clip(simplex[0] + kick, -5, 5)
                new_f = func(new_x)
                self.calls += 1
                if new_f < best_f_local:
                    simplex[0] = new_x
                    simplex_f[0] = new_f
                    # Re-expand simplex around new best
                    for i in range(1, dim + 1):
                        simplex[i] = new_x.copy()
                        simplex[i][i-1] += 0.5
                        simplex_f[i] = func(simplex[i])
                        self.calls += 1
                        if self.calls >= self.budget:
                            break
                    if self.calls >= self.budget:
                        break
                    order = np.argsort(simplex_f)
                    simplex = simplex[order]
                    simplex_f = [simplex_f[i] for i in order]
                stall_count = 0

        return best_x_local, best_f_local

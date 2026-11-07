import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

        # Helper to evaluate and track
        def evaluate(x):
            if self.calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            self.calls += 1
            try:
                f = func(x)
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')
            
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Nelder-Mead implementation
        def run_nelder_mead(x_start, max_calls):
            local_best_x = x_start.copy()
            local_best_f = evaluate(x_start)
            local_calls = 1
            
            if local_calls >= max_calls or time.time() > deadline:
                return local_best_x, local_best_f, local_calls

            # Initial simplex
            simplex = [x_start.copy()]
            for i in range(dim):
                x_i = x_start.copy()
                x_i[i] += 0.5 if x_start[i] < 0 else -0.5
                simplex.append(x_i)
            
            simplex_f = [local_best_f]
            for i in range(1, dim + 1):
                simplex_f.append(evaluate(simplex[i]))
                local_calls += 1
                if simplex_f[i] < local_best_f:
                    local_best_f = simplex_f[i]
                    local_best_x = simplex[i].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-8

            while local_calls < max_calls and time.time() < deadline:
                # Sort simplex
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                if simplex_f[0] < local_best_f:
                    local_best_f = simplex_f[0]
                    local_best_x = simplex[0].copy()

                # Check convergence
                if local_calls >= max_calls - 1:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)
                local_calls += 1
                
                if f_r < local_best_f:
                    local_best_f = f_r
                    local_best_x = x_r.copy()

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    local_calls += 1
                    if f_e < local_best_f:
                        local_best_f = f_e
                        local_best_x = x_e.copy()
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
                    f_c = evaluate(x_c)
                    local_calls += 1
                    if f_c < local_best_f:
                        local_best_f = f_c
                        local_best_x = x_c.copy()
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])
                            local_calls += 1
                            if simplex_f[i] < local_best_f:
                                local_best_f = simplex_f[i]
                                local_best_x = simplex[i].copy()
                        
                        # Check if shrinkage was effective
                        if np.max(np.abs(simplex[1] - simplex[0])) < min_step:
                            break

            return local_best_x, local_best_f, local_calls

        # --- Multi-Start Strategy ---
        
        # Phase 1: Explore with 5 diverse starts using Nelder-Mead
        # This is a structural change: using a derivative-free local search method 
        # that is often faster per iteration and can handle the discus function well.
        
        starts = [
            np.zeros(dim),
            np.random.uniform(-3, 3, dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim)
        ]
        
        short_budget = 80
        best_x_phase1 = np.zeros(dim)
        best_f_phase1 = float('inf')
        
        for s in starts:
            if self.calls >= budget - 50: # Keep at least 50 for final phase
                break
            if time.time() > deadline:
                break
                
            # Run short Nelder-Mead
            x_res, f_res, calls_used = run_nelder_mead(s, short_budget)
            
            if f_res < best_f_phase1:
                best_f_phase1 = f_res
                best_x_phase1 = x_res
        
        # Phase 2: Exploit the best point found
        if self.calls < budget - 10 and time.time() < deadline:
            remaining_budget = budget - self.calls
            
            # Run Nelder-Mead from the best point found in Phase 1
            x_final, f_final, _ = run_nelder_mead(best_x_phase1, remaining_budget)
            
            # The global best_x and best_f are already updated by 'evaluate' inside run_nelder_mead
            # So self.best_x and self.best_f are correct.

        return self.best_x, self.best_f

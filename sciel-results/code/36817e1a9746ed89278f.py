import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def time_ok():
            return time.time() < deadline
            
        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Multi-start Nelder-Mead Ensemble
        # This provides rapid local convergence from diverse basins,
        # maximizing early AOCC progress.
        
        n_starts = 5
        alpha = 1.0  # Initial simplex spread
        
        # Latin Hypercube sampling for diverse starting points
        lhs_points = np.zeros((n_starts, dim))
        for d in range(dim):
            perm = np.random.permutation(n_starts)
            for s in range(n_starts):
                frac = (perm[s] + np.random.random()) / n_starts
                lhs_points[s, d] = -5 + 10 * frac
        
        simplexes = []
        f_simplexes = []
        
        # Evaluate initial simplexes
        for s in range(n_starts):
            if calls >= budget or not time_ok():
                break
                
            simplex = []
            f_vals = []
            
            # First vertex: the LHS point
            x0 = lhs_points[s].copy()
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls += 1
            record(x0, f0)
            simplex.append(x0)
            f_vals.append(f0)
            
            # Additional vertices: perturb each dimension
            for d in range(dim):
                if calls >= budget or not time_ok():
                    break
                x_i = x0.copy()
                x_i[d] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                record(x_i, f_i)
                simplex.append(x_i)
                f_vals.append(f_i)
            
            if len(simplex) >= 2:
                simplexes.append(simplex)
                f_simplexes.append(f_vals)
        
        if len(simplexes) == 0:
            if best_x is not None:
                return best_x, best_f
            else:
                # Fallback if no valid simplex could be created
                x0 = np.zeros(dim)
                f0 = func(x0)
                calls += 1
                record(x0, f0)
                return best_x, best_f
        
        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5
        
        converged = [False] * len(simplexes)
        active_indices = list(range(len(simplexes)))
        
        # Interleave evaluations across all active simplexes
        while calls < budget and time_ok() and active_indices:
            idx = active_indices.pop(0)
            active_indices.append(idx)
            
            simplex = simplexes[idx]
            f_vals = f_simplexes[idx]
            
            # Sort by fitness
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Check convergence
            if len(f_vals) >= 2 and (f_vals[-1] - f_vals[0]) < 1e-8:
                converged[idx] = True
                continue
            
            x_best = simplex[0]
            f_best_s = f_vals[0]
            x_worst = simplex[-1]
            f_worst = f_vals[-1]
            
            if len(simplex) < 3:
                # Expand simplex if too small
                if calls < budget and time_ok():
                    d_rand = np.random.randint(dim)
                    x_new = x_best.copy()
                    x_new[d_rand] += alpha * 0.1
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    record(x_new, f_new)
                    simplex.append(x_new)
                    f_vals.append(f_new)
                    simplexes[idx] = simplex
                    f_simplexes[idx] = f_vals
                continue
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + gamma * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            if not time_ok() or calls >= budget:
                break
            f_r = func(x_r)
            calls += 1
            record(x_r, f_r)
            
            if f_r < f_best_s:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_e = func(x_e)
                calls += 1
                record(x_e, f_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + sigma_nm * (x_r - centroid)
                else:
                    x_c = centroid - sigma_nm * (x_worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_c = func(x_c)
                calls += 1
                record(x_c, f_c)
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if not time_ok() or calls >= budget:
                            break
                        simplex[i] = x_best + shrink * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex_i = func(simplex[i])
                        calls += 1
                        record(simplex_i, f_simplex_i)
                        f_vals[i] = f_simplex_i
                    if not time_ok():
                        break
            
            simplexes[idx] = simplex
            f_simplexes[idx] = f_vals
            
            # Remove converged simplexes
            if converged[idx]:
                if idx in active_indices:
                    active_indices.remove(idx)
        
        # Phase 2: Polishing with (1+1)-ES
        # Use the best point found by NM as the starting point for fine-grained exploitation.
        # This leverages the robustness of ES for the final steps where NM might oscillate or converge slowly.
        
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f_current = best_f
            sigma = 0.1 # Start with a smaller step size since we are likely close
            
            # Success rate control
            target_success = 0.2
            success_count = 0
            total_generations = 0
            window_size = 10
            
            while calls < budget and time_ok():
                z = np.random.randn(dim)
                norm_z = np.linalg.norm(z)
                if norm_z < 1e-10:
                    continue
                z /= norm_z
                
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                if not time_ok() or calls >= budget:
                    break
                f_new = func(x_new)
                calls += 1
                record(x_new, f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                success = (f_new < f_current)
                success_count += success
                total_generations += 1
                
                if total_generations >= window_size:
                    current_success_rate = success_count / window_size
                    if current_success_rate > target_success:
                        sigma *= 1.5
                    elif current_success_rate < target_success:
                        sigma /= 1.5
                    success_count = 0
                    total_generations = 0
                
                if f_new < f_current:
                    x = x_new
                    f_current = f_new
                
                if sigma < 1e-10:
                    break

        return best_x, best_f

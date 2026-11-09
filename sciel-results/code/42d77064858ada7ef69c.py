"""
Strategy: Adaptive Multi-Start with Two-Phase Search.
Rationale: The Discus function (f11) has a flat plateau surrounding a narrow basin.
We use a fast exploration phase with random sampling and local perturbations to
quickly locate the basin, then switch to Nelder-Mead for efficient convergence.
Multi-starting with diverse initial points helps escape the plateau.
"""

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
        deadline = start_time + 280  # Safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            total_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Fast Exploration Phase ---
        def explore(x0, max_calls, base_step=2.0):
            """
            Fast exploration: random perturbations around x0 with adaptive step.
            Returns best found.
            """
            nonlocal total_calls, best_f, best_x
            
            if total_calls >= budget:
                return best_x, best_f
            
            rng = np.random.default_rng(hash(x0.tobytes()) % (2**32))
            current_x = x0.copy()
            current_f = safe_call(current_x)
            if current_f is None:
                return best_x, best_f
            
            start_calls = total_calls
            no_improve = 0
            step = base_step
            prev_best = best_f
            
            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break
                
                # Generate random perturbation
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-15)
                new_x = current_x + step * direction
                new_x = np.clip(new_x, -5.0, 5.0)
                
                f_val = safe_call(new_x)
                if f_val is None:
                    break
                
                if f_val < current_f - 1e-10:
                    current_x = new_x.copy()
                    current_f = f_val
                    no_improve = 0
                    step = max(0.3 * step, 0.1)  # Shrink step on improvement
                else:
                    no_improve += 1
                    step = min(step * 1.5, 4.0)  # Grow step on stagnation
                
                if best_f < prev_best - 1e-8:
                    prev_best = best_f
                    no_improve = 0
                elif no_improve > 15:
                    break
            
            return best_x, best_f

        # --- Nelder-Mead Refinement ---
        def nelder_mead(x0, f0, max_calls, initial_step=1.0):
            """
            Nelder-Mead with adaptive parameters.
            """
            nonlocal total_calls, best_f, best_x
            
            if total_calls >= budget:
                return best_x, best_f
            
            simplex = [(x0.copy(), f0)]
            start_calls = total_calls
            
            for i in range(dim):
                if total_calls >= budget or total_calls - start_calls >= max_calls:
                    break
                xi = x0.copy()
                xi[i] += initial_step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append((xi, fi))
            
            if len(simplex) < dim + 1:
                return best_x, best_f
                
            simplex.sort(key=lambda t: t[1])
            
            alpha = 0.5
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            no_improve = 0
            prev_best = best_f
            
            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break
                
                x_center = np.mean([s[0] for s in simplex], axis=0)
                spread = np.max(np.abs(np.array([s[0] for s in simplex]) - x_center))
                if spread < 1e-6:
                    break
                
                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_second_worst = simplex[-2][0]
                f_second_worst = simplex[-2][1]
                
                x_centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)
                
                x_reflect = x_centroid + gamma * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = safe_call(x_reflect)
                if f_reflect is None:
                    break
                
                if f_reflect < simplex[0][1]:
                    x_expand = x_centroid + alpha * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = safe_call(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = (x_expand, f_expand)
                    else:
                        simplex[-1] = (x_reflect, f_reflect)
                elif f_reflect < f_second_worst:
                    simplex[-1] = (x_reflect, f_reflect)
                else:
                    if f_reflect < f_worst:
                        x_contract = x_centroid + rho * (x_reflect - x_worst)
                    else:
                        x_contract = x_centroid - rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = safe_call(x_contract)
                    if f_contract is None:
                        break
                    if f_contract <= min(f_reflect, f_worst):
                        simplex[-1] = (x_contract, f_contract)
                    else:
                        for i in range(1, dim + 1):
                            if total_calls >= budget:
                                break
                            new_x = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                            new_x = np.clip(new_x, -5.0, 5.0)
                            new_f = safe_call(new_x)
                            if new_f is None:
                                break
                            simplex[i] = (new_x, new_f)
                        if total_calls >= budget:
                            break
                
                simplex.sort(key=lambda t: t[1])
                
                if best_f < prev_best - 1e-8:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1
                    if no_improve > 20:
                        break
                        
            return best_x, best_f

        # --- Main Strategy ---
        
        # Phase 1: Exploration with multiple starts
        # Use 40% of budget for exploration
        explore_budget = int(budget * 0.4)
        explore_per_start = explore_budget // 4
        
        rng_main = np.random.default_rng(42)
        
        # Start 1: Center
        x0_1 = np.zeros(dim)
        explore(x0_1, explore_per_start, base_step=2.0)
        
        # Start 2: Random in [-3, 3]
        if total_calls < budget:
            x0_2 = rng_main.uniform(-3.0, 3.0, dim)
            explore(x0_2, explore_per_start, base_step=2.0)
        
        # Start 3: Random in [-4, 4]
        if total_calls < budget:
            x0_3 = rng_main.uniform(-4.0, 4.0, dim)
            explore(x0_3, explore_per_start, base_step=2.0)
        
        # Start 4: Around best found so far
        if total_calls < budget:
            explore(best_x.copy(), explore_per_start, base_step=1.0)
        
        # Phase 2: Refinement with Nelder-Mead
        remaining = budget - total_calls
        if remaining > 10:
            # Use most of remaining budget for refinement
            nm_budget = remaining
            nelder_mead(best_x.copy(), best_f, nm_budget, initial_step=0.5)
        
        return best_x, best_f

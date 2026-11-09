"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead with Halton initialization for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget or check_deadline():
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            return fx

        # Phase 1: Generate well-distributed initial points using Halton sequence
        # Halton sequence provides low-discrepancy coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    # Use prime bases for each dimension
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    points[i, j] = halton(i + 1, primes[j])
            return points

        # Generate 25 well-distributed points in [0,1]^dim
        n_initial = 25
        halton_points = halton_sequence(n_initial, dim)
        # Map to [-5, 5]^dim
        halton_points = halton_points * 10.0 - 5.0

        # Add the origin and a few random points for diversity
        extra_points = [np.zeros(dim)]
        extra_points.extend(rng.uniform(-5, 5, size=(5, dim)))
        
        all_initial_points = np.vstack([halton_points, np.array(extra_points)])
        rng.shuffle(all_initial_points)

        # Evaluate initial points
        for x in all_initial_points:
            fx = evaluate(x)
            if fx is None:
                break

        if best_x is None:
            return None, np.inf

        # Phase 2: Multi-start Nelder-Mead optimization
        # Use remaining budget for local refinement around top candidates
        
        # Identify top candidates from initial evaluation
        # We'll track the best few points and run Nelder-Mead from each
        
        # Simple approach: run Nelder-Mead from the best point first,
        # then from a few other promising points if budget allows
        
        def nelder_mead(x0, max_iter):
            nonlocal calls_made, best_f, best_x
            
            if check_deadline() or calls_made >= budget:
                return
            
            # Initialize simplex
            x = x0.copy()
            fx = evaluate(x)
            if fx is None:
                return
            
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            # Build simplex
            simplex = [x.copy()]
            simplex_f = [fx]
            
            # Add perturbed points
            for i in range(dim):
                perturbation = x.copy()
                perturbation[i] += 0.5 * (perturbation[i] != 0) + 0.5
                perturbation = np.clip(perturbation, -5, 5)
                
                if check_deadline() or calls_made >= budget:
                    return
                
                fp = evaluate(perturbation)
                if fp is None:
                    return
                simplex.append(perturbation)
                simplex_f.append(fp)
                
                if fp < best_f:
                    best_f, best_x = fp, perturbation.copy()
            
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            for _ in range(max_iter):
                if check_deadline() or calls_made >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                if check_deadline() or calls_made >= budget:
                    break
                
                f_reflect = evaluate(x_reflect)
                if f_reflect is None:
                    break
                
                if f_reflect < best_f:
                    best_f, best_x = f_reflect, x_reflect.copy()
                
                # If reflection is good, accept it
                if simplex_f[0] <= f_reflect < simplex_f[-2]:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                # If reflection is very good, try expansion
                elif f_reflect < simplex_f[0]:
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    
                    f_expand = evaluate(x_expand)
                    if f_expand is None:
                        break
                    
                    if f_expand < best_f:
                        best_f, best_x = f_expand, x_expand.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                # If reflection is not good enough, try contraction
                else:
                    # Outside contraction
                    x_contract = centroid + 0.5 * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    
                    f_contract = evaluate(x_contract)
                    if f_contract is None:
                        break
                    
                    if f_contract < best_f:
                        best_f, best_x = f_contract, x_contract.copy()
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if check_deadline() or calls_made >= budget:
                                break
                            
                            f_shrink = evaluate(simplex[i])
                            if f_shrink is None:
                                break
                            
                            simplex_f[i] = f_shrink
                            if f_shrink < best_f:
                                best_f, best_x = f_shrink, simplex[i].copy()
            
            # After Nelder-Mead, do fine-tuning with small steps
            if not check_deadline() and calls_made < budget:
                x = simplex[0].copy()
                fx = simplex_f[0]
                sigma = 0.1
                stall_count = 0
                max_stall = 10
                
                while calls_made < budget and not check_deadline():
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy is None:
                        break
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.05
                        stall_count = 0
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        sigma *= 0.9
                        stall_count += 1
                    
                    if stall_count > max_stall:
                        break

        # Run Nelder-Mead from the best initial point
        remaining_budget = budget - calls_made
        if remaining_budget > 10:
            nelder_mead(best_x.copy(), max_iter=min(30, remaining_budget // 2))
        
        # If budget remains, try from second-best initial point
        # We need to track the second-best point, but for simplicity,
        # let's just do a few more local refinements
        
        if calls_made < budget and not check_deadline():
            # Fine-tuning phase
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            stall_count = 0
            max_stall = 15
            
            while calls_made < budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.03
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.95
                    stall_count += 1
                
                if stall_count > max_stall:
                    # Try a larger perturbation
                    sigma = 0.3
                    stall_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END

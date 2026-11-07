"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Global-Local Optimizer for BBOB f22.
    Uses structured initialization for early progress, 
    a population-based stochastic tunneling method for global exploration, 
    and local (1+1)-ES for final refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def bounded_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        def local_refine(x0, fx0, max_calls, rng):
            """
            Fast (1+1)-ES refinement with adaptive sigma.
            Returns (best_x, best_f, calls_used)
            """
            x = x0.copy()
            fx = fx0
            sigma = 0.5
            calls_used = 0
            success_count = 0
            window = 5
            
            for _ in range(max_calls):
                if calls_made + 1 > budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = bounded_func(y)
                calls_used += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    pass
                
                # Adapt sigma every 'window' steps
                if (calls_used) % window == 0:
                    rate = success_count / window
                    if rate > 0.6:
                        sigma *= 1.2
                    elif rate < 0.4:
                        sigma *= 0.8
                    success_count = 0
                    sigma = np.clip(sigma, 1e-4, 2.0)
                    
            return x, fx, calls_used

        # Phase 1: Structured Initialization
        # Use a Halton sequence or simple grid for diverse, deterministic coverage.
        # 5D space. Let's use a 3-point grid per dimension: {-4, 0, 4}. Total 3^5 = 243 points.
        # This is too many if we want to leave budget for tunneling.
        # Let's use a smaller structured set: 2^5 = 32 points at corners of a box, plus center.
        # Or better: A quasi-random sequence (Sobol/Halton) for better space-filling.
        
        # Simple deterministic Halton-like initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_init = 40
        init_points = []
        for i in range(n_init):
            pt = np.array([halton(i + 1, b) for b in [2, 3, 5, 7, 11]])
            # Map [0,1) to [-5, 5)
            pt = pt * 10.0 - 5.0
            init_points.append(pt)
        
        # Also add the origin and a few random points to ensure coverage
        init_points.append(np.zeros(dim))
        init_points.append(rng.uniform(-5, 5, dim))
        
        population = []
        for pt in init_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            fx = bounded_func(pt)
            population.append((fx, pt.copy()))
            
            if fx < best_f:
                best_f = fx
                best_x = pt.copy()

        # Sort population by fitness
        population.sort(key=lambda x: x[0])
        
        # Phase 2: Stochastic Tunneling Search
        # Maintain a population of top K distinct points
        K = 10
        # Ensure population size is K
        while len(population) < K:
            # If we don't have enough, add random
            pt = rng.uniform(-5, 5, dim)
            fx = bounded_func(pt)
            population.append((fx, pt.copy()))
            if fx < best_f:
                best_f = fx
                best_x = pt.copy()
        
        population = population[:K]
        
        # Tunneling parameters
        # We will iterate until budget or time is up.
        # Each iteration generates one new point.
        
        while calls_made < budget:
            if time.time() > deadline - 5.0:
                break
                
            # Select parent: Roulette wheel or Tournament? 
            # Let's use a simple weighted selection based on rank (better = higher chance)
            # Or just pick the best one with high prob, others with lower.
            # To encourage exploration, let's pick a random index from 0 to K-1, 
            # but bias towards better ones.
            # Simple: Pick index i with prob proportional to 1/(i+1)
            weights = np.array([1.0 / (i + 1.0) for i in range(len(population))])
            weights /= np.sum(weights)
            idx = rng.choice(len(population), p=weights)
            parent_f, parent_x = population[idx]
            
            # Generate offspring
            r = rng.random()
            if r < 0.3:
                # Local perturbation (Gaussian)
                # Sigma adapts roughly based on distance to best? Let's keep fixed small sigma
                sigma = 0.5 
                offspring = parent_x + sigma * rng.standard_normal(dim)
            else:
                # Tunneling: Jump towards/away from another point
                # Pick a target from population (can be same or different)
                target_idx = rng.choice(len(population))
                target_f, target_x = population[target_idx]
                
                # Extrapolation factor
                # If target is better, jump towards it (and beyond)
                # If target is worse, jump away from it
                if target_f < parent_f:
                    # Move towards target
                    # alpha > 0
                    alpha = rng.uniform(0.5, 2.0) # 0.5 to 2.0 times the distance
                else:
                    # Move away from target
                    alpha = rng.uniform(-2.0, -0.5)
                
                # New point = parent + alpha * (target - parent)
                # This is equivalent to (1-alpha)*parent + alpha*target
                # If alpha=1, it's the target.
                # If alpha=2, it's target + (target-parent) -> beyond target
                # If alpha=0.5, it's midpoint
                # If alpha=-1, it's reflection
                direction = target_x - parent_x
                offspring = parent_x + alpha * direction
                
            offspring = np.clip(offspring, -5.0, 5.0)
            
            f_off = bounded_func(offspring)
            
            # Update global best
            if f_off < best_f:
                best_f = f_off
                best_x = offspring.copy()
            
            # Update population
            # Replace the worst point if the new one is better and distinct
            worst_idx = np.argmax([p[0] for p in population])
            if f_off < population[worst_idx][0]:
                # Check distinctness: distance > 0.1
                dist = np.linalg.norm(offspring - population[worst_idx][1])
                if dist > 0.1:
                    population[worst_idx] = (f_off, offspring.copy())
                    population.sort(key=lambda x: x[0])
            else:
                # Even if not better than worst, if it's distinct and better than some others, 
                # we might want to keep diversity. 
                # For simplicity, only replace if it beats the worst.
                pass

        # Phase 3: Final Refinement
        # Use remaining budget to polish the best point found
        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline - 1.0:
            # Run local refinement on best_x
            _, _, _ = local_refine(best_x, best_f, remaining, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END

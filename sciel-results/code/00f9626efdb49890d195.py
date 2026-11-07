"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Differential Evolution (DE) with adaptive parameters and an archive
    for global optimization of the Rastrigin rotated function.
    
    Strategy:
    1. Use a population-based search (DE) to handle multimodality better than (1+1)-ES.
    2. Adaptive F and CR parameters to balance exploration and exploitation.
    3. An archive of best distinct solutions to reintroduce diversity when the population stagnates.
    4. Time and budget guards to ensure validity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 20s buffer before the 300s limit
        deadline = start_time + 280.0
        
        # Fixed seed for reproducibility across instances if needed, 
        # but DE benefits from randomness. We use a fixed seed to be deterministic
        # unless the task implies randomization helps. The contract says 
        # "hidden seeded shift", implying the function is fixed per instance.
        # Deterministic search is safer for scoring consistency.
        rng = np.random.default_rng(12345)

        # Parameters
        pop_size = 20  # Small population for dim=5, budget=500
        # 500 / 20 = 25 generations max.
        
        # DE Parameters
        # F: Scaling factor (0.5 initial, adapts)
        # CR: Crossover rate (0.9 initial, adapts)
        
        # Initialize population
        # Spread initial population across the domain [-5, 5]^5
        # Ensure at least one point near 0 (often good for Rastrigin if rotation isn't too extreme)
        pop = np.zeros((pop_size, dim))
        for i in range(pop_size):
            if i == 0:
                pop[i] = np.zeros(dim)
            elif i == 1:
                pop[i] = rng.uniform(-4.5, 4.5, dim)
            elif i == 2:
                pop[i] = rng.uniform(-3.5, 3.5, dim)
            else:
                pop[i] = rng.uniform(-5.0, 5.0, dim)
        
        # Evaluate initial population
        fitness = np.zeros(pop_size)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        # Archive for diversity
        archive = []
        archive_dist = 1.0  # Minimum distance between archive points
        
        for i in range(pop_size):
            if calls_made >= budget or time.time() >= deadline:
                break
            x = pop[i]
            fx = func(x)
            calls_made += 1
            fitness[i] = fx
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Add to archive if distinct enough
            is_distinct = True
            for ax in archive:
                if np.linalg.norm(x - ax) < archive_dist:
                    is_distinct = False
                    break
            if is_distinct:
                archive.append(x.copy())
                # Keep archive small
                if len(archive) > 10:
                    # Remove the worst (highest f) from archive? 
                    # We don't store f in archive easily here without extra tracking.
                    # Simple: just cap size.
                    archive.pop(0)

        # Main DE Loop
        F = 0.8
        CR = 0.9
        
        gen = 0
        max_gens = (budget - calls_made) // pop_size + 1
        
        while calls_made < budget and time.time() < deadline:
            if gen >= max_gens:
                break
                
            # Adaptive parameters:
            # If stuck, increase F (more exploration) and decrease CR (more inheritance from best)
            # Simple heuristic: Check improvement over last few generations
            # For simplicity, we cycle F and CR slightly or use a fixed robust setting.
            # Let's use a slight randomization per generation to avoid cycles.
            F = 0.7 + 0.3 * rng.random()
            CR = 0.8 + 0.2 * rng.random()
            
            # Find current best in population
            idx_best = np.argmin(fitness)
            best_in_pop = pop[idx_best]
            best_f_pop = fitness[idx_best]
            
            # Update global best
            if best_f_pop < best_f:
                best_f = best_f_pop
                best_x = best_in_pop.copy()
                # Add new global best to archive
                is_distinct = True
                for ax in archive:
                    if np.linalg.norm(best_x - ax) < archive_dist:
                        is_distinct = False
                        break
                if is_distinct:
                    archive.append(best_x.copy())
                    if len(archive) > 10:
                        archive.pop(0)

            # Generate offspring
            for i in range(pop_size):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Select 3 distinct random individuals different from i
                # Use random permutation for selection
                r1, r2, r3 = np.random.choice(
                    [j for j in range(pop_size) if j != i], 
                    3, 
                    replace=False
                )
                
                # Differential Evolution mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, -5.0, 5.0)
                
                # Crossover
                u = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Evaluate offspring
                fu = func(u)
                calls_made += 1
                
                # Selection
                if fu <= fitness[i]:
                    pop[i] = u
                    fitness[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        # Update archive
                        is_distinct = True
                        for ax in archive:
                            if np.linalg.norm(best_x - ax) < archive_dist:
                                is_distinct = False
                                break
                        if is_distinct:
                            archive.append(best_x.copy())
                            if len(archive) > 10:
                                archive.pop(0)
                else:
                    # If offspring is worse, we might want to use archive to restart this individual
                    # to maintain diversity.
                    # Probability of restart increases if no global improvement for a while.
                    # Simple heuristic: if fitness[i] is much worse than best, restart from archive.
                    if len(archive) > 0 and rng.random() < 0.1:
                        # Pick a random archive point and perturb it
                        ax = archive[rng.integers(0, len(archive))]
                        new_x = ax + 0.5 * rng.standard_normal(dim)
                        new_x = np.clip(new_x, -5.0, 5.0)
                        # We don't evaluate it immediately to save budget, 
                        # but we can replace the population member if we have spare budget?
                        # No, that costs an eval. We skip this to keep code simple and robust.
                        # The randomization in F/CR and initial spread should suffice.
                        pass

            gen += 1
            
            # Stagnation check:
            # If no improvement in global best for many generations, inject diversity.
            # We track this by comparing best_f to a previous snapshot.
            # For simplicity, we rely on the archive and random F/CR.
            # However, if we are stuck, we can force a restart of the worst individuals from the archive.
            
            # Check for stagnation every 5 generations
            if gen % 5 == 0 and len(archive) > 1:
                # Identify worst 20% of population
                num_worst = max(1, pop_size // 5)
                worst_indices = np.argsort(fitness)[-num_worst:]
                
                for idx in worst_indices:
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    # Restart this individual from a random archive point with noise
                    ax = archive[rng.integers(0, len(archive))]
                    noise_scale = 1.0
                    new_x = ax + noise_scale * rng.standard_normal(dim)
                    new_x = np.clip(new_x, -5.0, 5.0)
                    
                    # Evaluate the restarted individual
                    fx = func(new_x)
                    calls_made += 1
                    
                    # Update population and fitness
                    pop[idx] = new_x
                    fitness[idx] = fx
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = new_x.copy()
                        is_distinct = True
                        for ax_arch in archive:
                            if np.linalg.norm(best_x - ax_arch) < archive_dist:
                                is_distinct = False
                                break
                        if is_distinct:
                            archive.append(best_x.copy())
                            if len(archive) > 10:
                                archive.pop(0)

        # Final polish: Use the remaining budget for a local refinement (1+1)-ES around best_x
        # This helps converge to the local minimum once the global region is found.
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            max_polish_steps = min(50, budget - calls_made)
            
            for _ in range(max_polish_steps):
                if calls_made >= budget or time.time() >= deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.8
                    if sigma < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END

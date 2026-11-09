"""Multi-objective optimizer on WFG4 using Multi-Objective Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MO-DE optimizer for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)

        # MO-DE Parameters
        # Population size needs to be large enough to cover the front but small enough
        # to allow many generations within the budget.
        # Budget 200. If pop_size=60, we have ~3 generations.
        # If pop_size=40, we have ~5 generations.
        # If pop_size=20, we have ~10 generations.
        # WFG4 requires good spread. Let's try pop_size=50.
        # Actually, standard DE works well with pop_size ~ 5-10 * dim.
        # dim=12 -> 60-120. But budget is tight.
        # Let's use pop_size = 40.
        pop_size = 40
        
        # Mutation factor F (0.5-1.0) and Crossover rate CR (0.1-0.9)
        F = 0.8
        CR = 0.5

        # Initialize population uniformly in bounds
        # We want to ensure the initial population covers the space well.
        pop = np.random.uniform(lo, hi, size=(pop_size, dim))
        
        # Evaluate initial population
        F_pop = np.zeros((pop_size, n_obj))
        used = 0
        for i in range(pop_size):
            if used >= budget or time.time() > deadline:
                break
            F_pop[i] = np.asarray(problem(pop[i]), dtype=float)
            used += 1
        
        if used == 0:
            return None

        # Ensure we have a full population or handle partial
        actual_pop_size = used
        pop = pop[:actual_pop_size]
        F_pop = F_pop[:actual_pop_size]
        
        # Archive for non-dominated points
        archive_x = []
        archive_f = []
        
        # Helper to update archive
        def update_archive(x_new, f_new):
            nonlocal archive_x, archive_f
            if len(archive_x) == 0:
                archive_x.append(x_new)
                archive_f.append(f_new)
                return
            
            # Check if dominated by any archive point
            dominated = False
            for f_a in archive_f:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    dominated = True
                    break
            
            if dominated:
                return
            
            # Remove points from archive dominated by f_new
            new_archive_x = []
            new_archive_f = []
            for j in range(len(archive_x)):
                f_a = archive_f[j]
                if np.all(f_new <= f_a) and np.any(f_new < f_a):
                    continue
                new_archive_x.append(archive_x[j])
                new_archive_f.append(f_a)
            
            new_archive_x.append(x_new)
            new_archive_f.append(f_new)
            
            archive_x = new_archive_x
            archive_f = new_archive_f

        # Initialize archive with current population
        for i in range(actual_pop_size):
            update_archive(pop[i], F_pop[i])

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            # DE/rand/1/bin
            # v_i = x_r1 + F * (x_r2 - x_r3)
            # u_i = v_i if random < CR else x_i
            # Evaluate u_i
            # Selection: If u_i dominates x_i, replace. If x_i dominates u_i, keep. Else use hypervolume.
            
            # We will generate one offspring per parent in the current population
            # To save evaluations, we might not update every single one if budget is low,
            # but standard DE updates the whole population.
            
            # We need to be careful with the budget.
            # Each generation costs pop_size evaluations.
            # If budget - used < pop_size, we can only do a partial generation or stop.
            # Let's do a full generation if we have enough budget, otherwise stop or do partial.
            
            if budget - used < pop_size:
                # Not enough for a full generation. 
                # We could do a partial generation, but it's messy.
                # Let's just break to avoid partial updates which might mess up the DE logic.
                # Or, we can just evaluate the remaining budget worth of random points?
                # No, DE relies on the population structure.
                # Let's just break. The archive holds the best found so far.
                break

            # Generate new population
            new_pop = np.zeros_like(pop)
            new_F_pop = np.zeros_like(F_pop)
            
            for i in range(actual_pop_size):
                # Select three distinct random indices
                r1, r2, r3 = np.random.choice(actual_pop_size, size=3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                # u_j = v_j if random < CR else x_i_j
                mask = np.random.rand(dim) < CR
                # Ensure at least one gene is from v (standard DE practice)
                if not np.any(mask):
                    mask[np.random.randint(dim)] = True
                
                u = np.where(mask, v, pop[i])
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = np.asarray(problem(u), dtype=float)
                used += 1
                
                # Check time
                if time.time() > deadline:
                    # Fill remaining with current pop to keep structures aligned?
                    # Or just break out of the inner loop and outer loop.
                    # If we break, we need to handle the state.
                    # Let's just break out of the inner loop.
                    # We will copy the rest of the current population to new_pop
                    new_pop[i] = u
                    new_F_pop[i] = f_u
                    # Copy remaining
                    for k in range(i+1, actual_pop_size):
                        new_pop[k] = pop[k]
                        new_F_pop[k] = F_pop[k]
                    # Break out of outer loop logic
                    # We need to flag that we broke early
                    break_outer = True
                    break
                else:
                    break_outer = False
                    new_pop[i] = u
                    new_F_pop[i] = f_u

            if break_outer:
                # We broke early due to time or budget
                # Update archive with the evaluated points so far
                for i in range(actual_pop_size):
                    # Only update if we actually evaluated it?
                    # In the loop above, if we broke early, indices > i were copied from old pop.
                    # We should only update archive for the ones we actually evaluated in this gen?
                    # No, the archive update is idempotent for dominated points.
                    # But we haven't evaluated the copied ones in this "new" context, they are old.
                    # So we just need to ensure the new evaluated ones are in the archive.
                    # Actually, the simplest is to just update archive for all points in new_pop
                    # but that's redundant for the copied ones.
                    # Let's just update for the ones we know we evaluated.
                    # Since we broke, we don't know exactly how many.
                    # Let's assume we evaluated up to 'i'.
                    # Wait, the 'break' in the inner loop sets break_outer.
                    # The index 'i' is the last one processed.
                    # So we evaluated 0..i.
                    for k in range(i + 1):
                        update_archive(new_pop[k], new_F_pop[k])
                    
                    # Update population to the new one (partially updated)
                    pop = new_pop
                    F_pop = new_F_pop
                    break # Exit while loop

            # If we completed the generation:
            # Perform selection
            for i in range(actual_pop_size):
                f_old = F_pop[i]
                f_new = new_F_pop[i]
                
                # Dominance check
                # 1. If new dominates old, replace
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    pop[i] = new_pop[i]
                    F_pop[i] = f_new
                    update_archive(pop[i], F_pop[i])
                # 2. If old dominates new, keep old
                elif np.all(f_old <= f_new) and np.any(f_old < f_new):
                    pass
                # 3. Incomparable: Use Hypervolume contribution
                else:
                    # Simple heuristic: Keep the one with better "spread" or lower sum?
                    # For MO-DE, a common strategy is to keep the one that contributes more to the hypervolume
                    # of the current front + candidate.
                    # However, calculating exact HV contribution is expensive.
                    # Alternative: Keep the one with lower sum of objectives (simple scalarization)
                    # or random selection to maintain diversity.
                    # Let's use a simple hypervolume proxy: 
                    # Calculate the box volume added by each point to the current archive?
                    # That's also expensive.
                    
                    # Let's use a simpler diversity metric: Distance to the centroid of the archive?
                    # Or just keep the one with lower sum.
                    # Lower sum tends to favor convergence.
                    # To maintain diversity, we might want to keep the one further from the center.
                    
                    # Let's try: If sums are close, keep the one that is more "different" from the current population?
                    # For simplicity and robustness, let's use the sum of objectives.
                    # If sum(new) < sum(old), replace.
                    if np.sum(f_new) < np.sum(f_old):
                        pop[i] = new_pop[i]
                        F_pop[i] = f_new
                        update_archive(pop[i], F_pop[i])
                    else:
                        # Keep old, but ensure old is in archive
                        update_archive(pop[i], F_pop[i])

        # Final return
        if archive_x:
            return archive_x[0]
        return None

# EVOLVE-BLOCK-END

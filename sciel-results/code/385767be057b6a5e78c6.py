"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Evolutionary Multi-Objective Optimizer with Error Repair (NSGA-II variant)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        used = 0
        archive = []  # List of (x, f)

        def add_to_archive(x, f):
            nonlocal archive
            f = np.array(f)
            # Check domination
            dominated = False
            for af in [p[1] for p in archive]:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            
            new_archive.append((x, f))
            archive = new_archive
            return True

        # Phase 1: Initial Population (Latin Hypercube Sampling)
        # Use a significant portion of the budget to establish a diverse front
        n_init = 40
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * span
        for x in X_init:
            if time.time() > deadline or used >= self.budget:
                break
            f = problem(x)
            used += 1
            add_to_archive(x, f)
        
        # Phase 2: Evolutionary Loop with Error Repair
        # We maintain a population. Each generation, we create offspring via crossover/mutation.
        # "Error Repair" here means: if an offspring is dominated by an archive point, 
        # we try to "repair" it by moving it towards a non-dominated archive point 
        # that is far away in objective space, effectively using the archive as a guide for escape.
        
        pop_size = 20
        # Initialize population from archive if possible, otherwise random
        if len(archive) >= pop_size:
            # Sample from archive
            idxs = np.random.choice(len(archive), size=pop_size, replace=False)
            population = [archive[i][0].copy() for i in idxs]
        else:
            population = [np.random.uniform(lo, hi) for _ in range(pop_size)]
            # Evaluate initial population
            for x in population:
                if used >= self.budget or time.time() > deadline:
                    break
                f = problem(x)
                used += 1
                add_to_archive(x, f)

        max_generations = 50
        sigma = 0.2  # Mutation strength

        for gen in range(max_generations):
            if time.time() > deadline or used >= self.budget:
                break
            
            offspring = []
            
            # Generate offspring
            for i in range(pop_size):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Crossover: Blend two parents
                p1_idx = np.random.randint(0, len(population))
                p2_idx = np.random.randint(0, len(population))
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                
                # Simple arithmetic crossover
                alpha = np.random.rand(self.dim)
                child = alpha * p1 + (1 - alpha) * p2
                
                # Mutation
                child += np.random.normal(0, sigma, self.dim)
                
                # Bounds handling (Clip)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                if used < self.budget:
                    f = problem(child)
                    used += 1
                    
                    # Error Repair Logic:
                    # Check if child is dominated by archive
                    is_dominated = False
                    for af in [p[1] for p in archive]:
                        if np.all(af <= f) and np.any(af < f):
                            is_dominated = True
                            break
                    
                    if is_dominated:
                        # Repair: Try to move towards a "gap" in the front.
                        # Pick a random non-dominated archive point that is "far" in objective space
                        # and blend the child towards it.
                        if len(archive) > 0:
                            # Calculate distance to all archive points in objective space
                            # We want to find a direction that might escape domination.
                            # A simple repair: take a random archive point and blend.
                            repair_idx = np.random.randint(0, len(archive))
                            repair_x = archive[repair_idx][0]
                            repair_f = archive[repair_idx][1]
                            
                            # Blend child towards repair point
                            repair_alpha = np.random.rand(self.dim)
                            repaired_child = repair_alpha * child + (1 - repair_alpha) * repair_x
                            repaired_child = np.clip(repaired_child, lo, hi)
                            
                            # Evaluate repaired child
                            if used < self.budget:
                                f_rep = problem(repaired_child)
                                used += 1
                                # Use the repaired child if it's better or non-dominated
                                # Simple heuristic: if it's not dominated by the original child, use it
                                if not (np.all(f <= f_rep) and np.any(f < f_rep)):
                                    child = repaired_child
                                    f = f_rep
                                else:
                                    # If original was better, keep original
                                    pass
                                add_to_archive(child, f)
                    else:
                        add_to_archive(child, f)
                
                offspring.append(child)
            
            # Update population: Keep the best individuals
            # Sort offspring by a simple scalarization (sum of objectives) for selection
            # This is a crude selection, but helps maintain diversity if combined with archive
            if len(offspring) == pop_size:
                # Evaluate all offspring (already done above, but we need their f values for sorting)
                # We didn't store f in offspring list. Let's re-evaluate or store them.
                # To save budget, let's just select randomly or based on archive membership?
                # Better: Store (x, f) in offspring generation step.
                
                # Let's refactor slightly to store f values during offspring creation.
                # For now, let's just replace population with offspring.
                # This is "generational replacement".
                population = offspring
            else:
                # If budget ran out during offspring generation, stop
                break

        return archive

# EVOLVE-BLOCK-END

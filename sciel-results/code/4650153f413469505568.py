"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution based Multi-Objective Optimizer."""

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
        archive_X = []
        archive_F = []
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.array(f)
            # Check if f is strictly better than reference point
            # Reference point for RE34 is [1705.0, 12.0, 0.3]
            # We don't strictly need to filter by ref point for the archive itself 
            # as long as we record all, but for DE selection we might want 
            # to focus on the "interesting" region. However, standard NSGA-II 
            # style archive is safer.
            
            # Check domination
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # DE Parameters
        # Budget is 200. 
        # Population size N. 
        # Total evaluations = N (init) + N * generations.
        # Let's try N=10, Generations=19. Total = 10 + 190 = 200.
        N = 10
        generations = (self.budget - N) // N
        
        # Initialize population
        # Use Latin Hypercube or Random? Random is standard for DE.
        # Let's use Random Uniform within bounds.
        pop_X = np.random.rand(N, self.dim) * span + lo
        pop_F = np.array([problem(x) for x in pop_X])
        used += N
        
        # Initialize archive
        for i in range(N):
            add_to_archive(pop_X[i], pop_F[i])
        
        # DE Loop
        # Mutation: v = x_r1 + F * (x_r2 - x_r3)
        # Crossover: binomial with CR
        # Selection: 1-to-1 tournament or direct comparison?
        # For multi-objective, we usually use a crowded distance or hypervolume contribution.
        # Given the small budget, a simple "archive-based" selection is risky.
        # Let's use a standard DE update rule but with a twist:
        # We maintain a population. For each individual i, we create a trial vector.
        # We evaluate the trial vector.
        # We decide whether to replace pop[i] with trial based on:
        # 1. If trial dominates pop[i] -> replace.
        # 2. If pop[i] dominates trial -> keep.
        # 3. If non-dominated -> use a diversity metric (e.g. distance to nearest neighbor in population or hypervolume contribution).
        # To keep it simple and fast within the tight budget, let's use a 
        # "Hypervolume Island" approach or just simple domination + random tie-breaking?
        # Actually, a common effective simple strategy for small budgets is 
        # to use the archive for selection pressure.
        
        # Let's implement a simple DE with archive-based selection:
        # If trial is non-dominated with pop[i], we keep the one with better 
        # "spread" contribution. Approximating spread: 
        # Distance to the nearest point in the *current population* (not archive) 
        # is a good proxy for diversity.
        
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        for gen in range(generations):
            if time.time() > deadline or used >= self.budget:
                break
            
            for i in range(N):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Select 3 distinct random indices
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                
                # Mutation
                v = pop_X[r1] + F * (pop_X[r2] - pop_X[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover
                u = pop_X[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Evaluate trial
                f_u = problem(u)
                used += 1
                
                # Add to archive
                add_to_archive(u, f_u)
                
                # Selection: Decide if u replaces pop_X[i]
                f_i = pop_F[i]
                
                # Check domination
                u_dom_i = np.all(f_u <= f_i) and np.any(f_u < f_i)
                i_dom_u = np.all(f_i <= f_u) and np.any(f_i < f_u)
                
                if u_dom_i:
                    # Trial is better, replace
                    pop_X[i] = u
                    pop_F[i] = f_u
                elif i_dom_u:
                    # Current is better, keep
                    pass
                else:
                    # Non-dominated: Use diversity in population
                    # Calculate distance from u to other points in pop
                    # and from i to other points in pop.
                    # Keep the one that is "more different" (further away) to maintain spread.
                    
                    # Simple diversity metric: sum of distances to other population members
                    # Exclude self
                    def div_score(x, pop_x):
                        dists = []
                        for k in range(N):
                            if k == i:
                                continue
                            dists.append(np.linalg.norm(x - pop_x[k]))
                        return np.mean(dists) if dists else 0.0
                    
                    # Note: This is O(N^2) per generation, which is 10*10*19 = 1900 ops, very fast.
                    div_u = div_score(u, pop_X)
                    div_i = div_score(pop_X[i], pop_X)
                    
                    if div_u > div_i:
                        pop_X[i] = u
                        pop_F[i] = f_u
                    else:
                        # Keep current, but maybe add u to archive if it wasn't dominated there?
                        # It was already added to archive above.
                        pass

        return archive_X

# EVOLVE-BLOCK-END

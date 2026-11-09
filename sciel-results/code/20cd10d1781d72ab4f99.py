"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 using NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    
    Strategy:
    1. Treat the problem as a black-box multi-objective optimization in 22D.
    2. Use NSGA-II with a population size of 200 (using the full budget for one generation 
       or a few small generations, but given the 330s limit and 200 budget, we can do 
       exactly 1 large generation or a few small ones. Actually, with 200 evaluations, 
       we can do a single generation of size 200, or multiple generations if we reuse 
       points? No, "Every returned objective vector is recorded". We have 200 evals.
       Standard NSGA-II needs population N, then N children. That's 2N evals.
       With budget 200, we can have N=100, do 1 generation (100 parents + 100 children = 200 evals).
       Or we can just do 200 random samples? No, NSGA-II is better.
       
       Let's do:
       Population size N = 100.
       Gen 1: Evaluate 100 random points (or Sobol for better start).
       Gen 2: Create 100 children via tournament selection, crossover, mutation. Evaluate them.
       Total evals = 200.
       Return all 200 points.
       
       This allows the algorithm to "learn" the structure (e.g., tail variables) over the 
       second generation if the first generation has some variance in tails.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        n_pop = n_evals // 2  # 100 for 200 budget
        
        # Helper to check deadline
        def is_deadline():
            return time.time() > deadline

        # 1. Initialize Population
        # Use random initialization to explore the space broadly
        pop = np.random.rand(n_pop, self.dim) * (hi - lo) + lo
        evals = 0
        all_points = []
        all_fvals = []
        
        # Evaluate initial population
        for i in range(n_pop):
            if is_deadline():
                break
            x = pop[i]
            f = problem(x)
            all_points.append(x)
            all_fvals.append(f)
            evals += 1
        
        # If we ran out of budget or time, return what we have
        if evals >= n_evals or is_deadline():
            return all_fvals

        # 2. NSGA-II Step: Selection, Crossover, Mutation
        # We need to perform non-dominated sorting on the current population
        fvals = np.array(all_fvals)
        
        # Simple Non-Dominated Sorting
        def non_dominated_sort(points):
            n = len(points)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            remaining = list(range(n))
            
            while remaining:
                front = []
                for i in remaining:
                    is_dom = False
                    for j in remaining:
                        if i == j:
                            continue
                        # Check if j dominates i
                        if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                            is_dom = True
                            break
                    if not is_dom:
                        front.append(i)
                fronts.append(front)
                for i in front:
                    ranks[i] = len(fronts) - 1
                    remaining.remove(i)
            return ranks, fronts

        ranks, fronts = non_dominated_sort(fvals)
        
        # Calculate Crowding Distance
        def crowding_distance(points, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            m = points.shape[1]
            
            for k in range(m):
                sorted_idx = np.argsort(points[front_indices, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    denom = points[front_indices[idx_next], k] - points[front_indices[idx_prev], k]
                    if denom != 0:
                        cd[idx] += (points[front_indices[idx_next], k] - points[front_indices[idx_prev], k]) / denom
                    else:
                        cd[idx] = np.inf
            return cd

        # Assign crowding distances
        cd = np.zeros(n_pop)
        for front in fronts:
            if len(front) > 0:
                cd[front] = crowding_distance(fvals, front)

        # Selection: Tournament Selection
        def tournament_select(pop_indices, k=2):
            selected = []
            for _ in range(len(pop_indices) // 2):
                # Pick k random individuals
                candidates = np.random.choice(pop_indices, k, replace=False)
                # Find the one with best rank, then best crowding distance
                best_idx = -1
                best_rank = np.inf
                best_cd = -np.inf
                for idx in candidates:
                    if ranks[idx] < best_rank:
                        best_rank = ranks[idx]
                        best_cd = cd[idx]
                        best_idx = idx
                    elif ranks[idx] == best_rank and cd[idx] > best_cd:
                        best_cd = cd[idx]
                        best_idx = idx
                selected.append(best_idx)
            return selected

        # Generate Children
        parent_indices = list(range(n_pop))
        # Select n_pop parents (with replacement effectively, but we need n_pop children)
        # Standard NSGA-II selects N parents to produce N children
        selected_parents = []
        while len(selected_parents) < n_pop:
            batch = tournament_select(parent_indices, k=2)
            selected_parents.extend(batch)
            # Stop if we have enough
            if len(selected_parents) >= n_pop:
                break
        
        selected_parents = selected_parents[:n_pop]
        
        children = np.zeros((n_pop, self.dim))
        
        # Crossover: SBX (Simulated Binary Crossover) or Simple Blend
        # Let's use a simple blend crossover with mutation
        eta_c = 20.0
        eta_m = 20.0
        
        for i in range(0, n_pop, 2):
            if i + 1 >= n_pop:
                break
            p1_idx = selected_parents[i]
            p2_idx = selected_parents[i+1]
            
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]
            
            # Crossover
            # Random probability of crossover per gene
            cross_mask = np.random.rand(self.dim) < 0.5
            
            c1 = p1.copy()
            c2 = p2.copy()
            
            for j in range(self.dim):
                if cross_mask[j]:
                    # SBX
                    if p1[j] == p2[j]:
                        c1[j] = p1[j]
                        c2[j] = p2[j]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            alpha = 2.0 * u
                            beta = ((alpha + 1.0) ** (1.0 / (eta_c + 1.0)))
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta = ((alpha + 1.0) ** (1.0 / (eta_c + 1.0)))
                        
                        c1[j] = 0.5 * ((1.0 + beta) * p1[j] + (1.0 - beta) * p2[j])
                        c2[j] = 0.5 * ((1.0 + 1.0/beta) * p2[j] + (1.0 - 1.0/beta) * p1[j])
                        
                        # Clip to bounds
                        c1[j] = np.clip(c1[j], lo[j], hi[j])
                        c2[j] = np.clip(c2[j], lo[j], hi[j])
            
            # Mutation
            # Polynomial Mutation
            for j in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    y = c1[j]
                    yl = lo[j]
                    yu = hi[j]
                    delta1 = (y - yl) / (yu - yl) if yu != yl else 0
                    delta2 = (yu - y) / (yu - yl) if yu != yl else 0
                    u = np.random.rand()
                    if u < 0.5:
                        xy = 2.0 * u
                        cmut = (xy + 1.0) ** (1.0 / (eta_m + 1.0))
                    else:
                        xy = 2.0 * (1.0 - u)
                        cmut = (xy + 1.0) ** (1.0 / (eta_m + 1.0))
                    deltaq = (yu - yl) * (cmut - 1.0)
                    y = y + deltaq * (2.0 * np.random.rand() - 1.0)
                    c1[j] = np.clip(y, lo[j], hi[j])
                
                if np.random.rand() < 1.0 / self.dim:
                    y = c2[j]
                    yl = lo[j]
                    yu = hi[j]
                    delta1 = (y - yl) / (yu - yl) if yu != yl else 0
                    delta2 = (yu - y) / (yu - yl) if yu != yl else 0
                    u = np.random.rand()
                    if u < 0.5:
                        xy = 2.0 * u
                        cmut = (xy + 1.0) ** (1.0 / (eta_m + 1.0))
                    else:
                        xy = 2.0 * (1.0 - u)
                        cmut = (xy + 1.0) ** (1.0 / (eta_m + 1.0))
                    deltaq = (yu - yl) * (cmut - 1.0)
                    y = y + deltaq * (2.0 * np.random.rand() - 1.0)
                    c2[j] = np.clip(y, lo[j], hi[j])
            
            children[i] = c1
            children[i+1] = c2
            
            # Fill remaining if n_pop is odd (shouldn't be with 100)
            if i + 1 == n_pop - 1:
                 children[i+1] = c2 # Just duplicate or keep as is

        # Evaluate Children
        for i in range(n_pop):
            if is_deadline():
                break
            x = children[i]
            f = problem(x)
            all_points.append(x)
            all_fvals.append(f)
            evals += 1
            
            if evals >= n_evals:
                break
        
        return all_fvals

# EVOLVE-BLOCK-END

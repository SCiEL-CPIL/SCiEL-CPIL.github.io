"""Multi-objective optimizer: Adaptive Grid Refinement + Differential Evolution hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # 1. Initialization: Latin Hypercube Sampling for good initial spread
        # We want a decent initial population to define the initial "grid" or search space.
        pop_size = 30
        # Ensure we don't exceed budget in init
        if pop_size > budget:
            pop_size = budget

        def lhs_sample(n, dim):
            samples = np.zeros((n, dim))
            for j in range(dim):
                perm = np.arange(n)
                np.random.shuffle(perm)
                samples[:, j] = (perm + np.random.random(n)) / n
            return samples

        X = lo + (hi - lo) * lhs_sample(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: Check dominance
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Non-dominated sort (returns list of fronts, each list of indices)
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                front0 = next_front
            # Assign remaining
            assigned = set()
            for f in fronts:
                assigned.update(f)
            if len(assigned) < n:
                remaining = [i for i in range(n) if i not in assigned]
                if remaining:
                    fronts.append(remaining)
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            idx = np.array(indices)
            for m in range(n_obj):
                vals = F[idx, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                vrange = vals[order[-1]] - vals[order[0]]
                if vrange > 1e-15:
                    for k in range(1, n - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / vrange
            return cd

        # Main Loop: Hybrid Adaptive Grid + DE
        # We maintain a population. In each iteration, we identify the "best" cells
        # (non-dominated points with high crowding distance) and generate offspring
        # specifically around them using DE-like perturbations, effectively refining
        # the grid around promising regions.

        # To avoid excessive overhead of complex grid structures, we simulate "grid refinement"
        # by focusing mutation intensity on the Pareto front.
        
        # Parameters for DE-like mutation
        F_mut = 0.5  # Differential weight
        CR = 0.9     # Crossover rate

        while used < budget:
            if time.time() > deadline:
                break

            # Determine number of offspring to generate in this step
            # We do smaller batches to allow frequent re-evaluation of fronts
            n_off = min(pop_size, budget - used)
            if n_off <= 0:
                break

            # 1. Identify "Promising Regions" (Grid Cells to Refine)
            # We use the current population's Pareto front.
            fronts = non_dominated_sort(F)
            if not fronts:
                break
            
            # The first front is the Pareto front
            front0 = fronts[0]
            if not front0:
                break
                
            # Calculate crowding distance for the front to identify "gaps" or "edges"
            # Points with high crowding distance are in sparse regions, which is where
            # "grid refinement" should happen to fill gaps in the Pareto front coverage.
            cd = crowding_distance(F, front0)
            
            # We want to pick parents from the front, biased towards high crowding distance
            # This effectively subdivides the space where the front is currently under-sampled.
            
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_off):
                # Select parent from Pareto front (or first few fronts if front is small)
                # We bias selection towards points with high crowding distance.
                # Simple approach: Randomly pick from the front, but if we want to be smarter:
                # Let's just pick a random parent from the front for now, as the mutation
                # will handle the local exploration.
                
                # To implement "adaptive grid refinement", we can vary the mutation scale
                # based on how "crowded" the neighborhood is.
                # If a point is isolated (high CD), we can use a larger mutation step to explore.
                # If it's crowded (low CD), use a smaller step to refine.
                
                p_idx = np.random.choice(front0)
                p = X[p_idx]
                
                # Adaptive Mutation Scale
                # Normalize crowding distance to [0, 1] roughly
                max_cd = np.max(cd)
                min_cd = np.min(cd)
                if max_cd > min_cd:
                    norm_cd = (cd[front0.index(p_idx)] - min_cd) / (max_cd - min_cd)
                else:
                    norm_cd = 0.5
                    
                # Base scale is 0.1 of the domain range
                # If norm_cd is high (isolated point), scale is larger (explore)
                # If norm_cd is low (crowded point), scale is smaller (exploit/refine)
                # Let's say scale = base * (1 + norm_cd)
                base_scale = 0.1
                scale = base_scale * (1.0 + norm_cd)
                
                # Differential Mutation: v = X[r1] + F * (X[r2] - X[r3])
                # Pick 3 distinct random indices from the whole population
                # Ensure we have at least 3 points
                if len(X) >= 3:
                    r1, r2, r3 = np.random.choice(len(X), 3, replace=False)
                    v = X[r1] + F_mut * (X[r2] - X[r3])
                else:
                    # Fallback to simple random mutation if population is tiny
                    v = p + scale * (hi - lo) * np.random.standard_normal(dim)
                    v = np.clip(v, lo, hi)
                    fc = problem(v)
                    used += 1
                    offspring_X.append(v)
                    offspring_F.append(fc)
                    continue

                # Clip the vector to bounds before crossover
                v = np.clip(v, lo, hi)
                
                # Binomial Crossover
                child = np.copy(p)
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() <= CR or j == j_rand:
                        child[j] = v[j]
                    else:
                        child[j] = p[j]
                
                # Additional Gaussian mutation to ensure exploration, scaled by adaptive factor
                # This helps escape local optima and "refine" the grid in unexplored directions
                child = child + scale * 0.5 * (hi - lo) * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                
                if used >= budget or time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # 2. Selection: Combine and Select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined
            all_fronts = non_dominated_sort(combined_F)
            
            # Select top pop_size
            new_X = []
            new_F = []
            for front in all_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(cd)[::-1]  # descending crowding
                    selected = [front[k] for k in order[:remaining]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break

            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)
            else:
                break

        return X
# EVOLVE-BLOCK-END

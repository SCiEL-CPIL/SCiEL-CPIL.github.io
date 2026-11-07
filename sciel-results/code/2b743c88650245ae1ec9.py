"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II Multi-Objective Optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 10
        if self.budget < N:
            N = max(1, self.budget)
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II Parameters
        eta_c = 20.0
        eta_m = 20.0
        pc = 0.9
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            front = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    rank[i] = 0
                    front.append(i)
            
            fronts = []
            k = 0
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, rank, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort([F[i, m] for i in indices])
                sorted_indices = [indices[i] for i in sorted_idx]
                
                min_val = F[sorted_indices[0], m]
                max_val = F[sorted_indices[-1], m]
                range_val = max_val - min_val
                if range_val == 0:
                    continue
                    
                dist[0] = np.inf
                dist[-1] = np.inf
                
                for i in range(1, n - 1):
                    dist[i] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / range_val
            return dist

        def tournament_select(pop_X, pop_F, pop_rank, pop_dist):
            idx1 = np.random.randint(len(pop_X))
            idx2 = np.random.randint(len(pop_X))
            
            if pop_rank[idx1] != pop_rank[idx2]:
                return idx1 if pop_rank[idx1] < pop_rank[idx2] else idx2
            else:
                return idx1 if pop_dist[idx1] > pop_dist[idx2] else idx2

        def sbx_crossover(x1, x2):
            if np.random.rand() > pc:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if x1[i] == x2[i]:
                        continue
                    if x1[i] > x2[i]:
                        x1[i], x2[i] = x2[i], x1[i]
                    
                    beta1 = 1.0 + (2.0 * (x1[i] - lo[i]) / (x2[i] - x1[i]))
                    beta2 = 1.0 + (2.0 * (hi[i] - x2[i]) / (x2[i] - x1[i]))
                    
                    alpha1 = 2.0 - beta1 ** (-(eta_c + 1.0))
                    alpha2 = 2.0 - beta2 ** (-(eta_c + 1.0))
                    
                    c1[i] = 0.5 * ((x1[i] + x2[i]) - alpha1 * (x2[i] - x1[i]))
                    c2[i] = 0.5 * ((x1[i] + x2[i]) + alpha2 * (x2[i] - x1[i]))
                    
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        def polynomial_mutation(x):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                    u = np.random.rand()
                    if u < 0.5:
                        val = (2.0 * u + (1.0 - 2.0 * u) * delta1 ** (eta_m + 1.0)) ** (1.0 / (eta_m + 1.0))
                        deltaq = (hi[i] - lo[i]) * (val - 1.0)
                    else:
                        val = (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * delta2 ** (eta_m + 1.0)) ** (1.0 / (eta_m + 1.0))
                        deltaq = (hi[i] - lo[i]) * (1.0 - val)
                    x_mut[i] = np.clip(x_mut[i] + deltaq, lo[i], hi[i])
            return x_mut

        # Main Loop
        while used < self.budget:
            # Generate Offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(N):
                if used >= self.budget:
                    break
                
                p1_idx = tournament_select(X, F, rank, dist)
                p2_idx = tournament_select(X, F, rank, dist)
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                
                f_c1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f_c1)
                
                # We only have budget for N offspring per gen effectively if we pair them up?
                # Standard NSGA-II generates N offspring from N parents.
                # Here we generate one child per parent selection pair? 
                # Usually, we generate N children. 
                # To save budget, we can just generate N children in total per generation.
                # The loop above generates 1 child per iteration. We need N children.
                # But wait, the loop runs N times. So we generate N children.
                # However, we also need to evaluate them.
                # The code above appends to offspring lists.
                
            if not offspring_X:
                break
                
            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)
            
            # Combine
            pop_X = np.vstack([X, offspring_X])
            pop_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts, rank = non_dominated_sort(pop_F)
            
            # Crowding distance
            dist = np.zeros(len(pop_X))
            for front in fronts:
                if len(front) == 0:
                    continue
                # Calculate crowding distance for this front
                # We need to map indices back to local indices in the front list
                # My crowding_distance function takes indices into the global F array.
                # Let's adapt it or call it carefully.
                # The current crowding_distance(F, rank, indices) expects indices into F.
                # It returns distances for those indices.
                # I will create a helper that returns a full dist vector.
                
                pass # Placeholder, need to implement full dist vector calculation
            
            # Let's rewrite the selection logic to be cleaner.
            # I'll create a helper to get the next population.
            
            # Actually, let's just inline the selection.
            # We need to select N best individuals from pop_X (size 2N).
            
            next_X = []
            next_F = []
            
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    # Take all
                    for i in front:
                        next_X.append(pop_X[i])
                        next_F.append(pop_F[i])
                else:
                    # Need to select some from this front based on crowding distance
                    # Calculate crowding distance for this front
                    front_indices = front
                    n_front = len(front_indices)
                    if n_front <= 2:
                        # Take all if possible, else random?
                        # If we can't take all, we take all if they fit, but they don't.
                        # We need to trim.
                        pass
                    else:
                        # Calculate distances
                        local_dist = np.zeros(n_front)
                        for m in range(self.n_obj):
                            sorted_idx = np.argsort([pop_F[i, m] for i in front_indices])
                            sorted_indices = [front_indices[i] for i in sorted_idx]
                            
                            min_val = pop_F[sorted_indices[0], m]
                            max_val = pop_F[sorted_indices[-1], m]
                            range_val = max_val - min_val
                            if range_val == 0:
                                continue
                                
                            local_dist[0] = np.inf
                            local_dist[-1] = np.inf
                            
                            for k in range(1, n_front - 1):
                                local_dist[k] += (pop_F[sorted_indices[k+1], m] - pop_F[sorted_indices[k-1], m]) / range_val
                        
                        # Sort by distance descending
                        sorted_by_dist = np.argsort(local_dist)[::-1]
                        
                        # Select needed
                        needed = N - len(next_X)
                        for k in range(needed):
                            idx = front_indices[sorted_by_dist[k]]
                            next_X.append(pop_X[idx])
                            next_F.append(pop_F[idx])
                    break # We only process one partial front
                
            if not next_X:
                break

            X = np.array(next_X)
            F = np.array(next_F)
            
            # Recalculate rank and dist for next iteration's tournament
            fronts, rank = non_dominated_sort(F)
            dist = np.zeros(len(X))
            for front in fronts:
                n_front = len(front)
                if n_front <= 2:
                    for i in front:
                        dist[i] = np.inf
                    continue
                local_dist = np.zeros(n_front)
                for m in range(self.n_obj):
                    sorted_idx = np.argsort([F[i, m] for i in front])
                    sorted_indices = [front[i] for i in sorted_idx]
                    
                    min_val = F[sorted_indices[0], m]
                    max_val = F[sorted_indices[-1], m]
                    range_val = max_val - min_val
                    if range_val == 0:
                        continue
                        
                    local_dist[0] = np.inf
                    local_dist[-1] = np.inf
                    
                    for k in range(1, n_front - 1):
                        local_dist[k] += (F[sorted_indices[k+1], m] - F[sorted_indices[k-1], m]) / range_val
                for k, i in enumerate(front):
                    dist[i] = local_dist[k]

        return X
# EVOLVE-BLOCK-END

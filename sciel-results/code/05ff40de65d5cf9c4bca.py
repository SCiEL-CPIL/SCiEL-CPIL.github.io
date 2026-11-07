import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters for NSGA-II
        # Budget is 200. Dim is 4.
        # We want to maximize hypervolume, so diversity is key.
        # Let's use a population size that divides the budget nicely.
        # Pop size 20, 10 generations (1 init + 9 offspring) = 200 evals.
        pop_size = 20
        n_generations = (self.budget - pop_size) // pop_size
        
        # 1. Initialization: Sobol sequence for good spread
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_sobol = 2 ** int(np.ceil(np.log2(pop_size)))
        points = sampler.random(n_sobol)
        points = points[:pop_size]
        
        X = lo + points * (hi - lo)
        F = np.array([problem(x) for x in X])
        
        # Helper for NSGA-II
        def non_dominated_sort(F):
            """
            Returns ranks (0-based) and fronts (list of lists of indices).
            """
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            S = [[] for _ in range(n)] # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int) # n_dom[i] = number of points dominating i
            
            # Compute domination relationships
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # If i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1 # Wait, this is wrong. 
                        # If i dominates j, then j is dominated by i.
                        # So n_dom[j] += 1.
                        # Let's restart this logic carefully.
                        
            # Correct domination count
            n_dom = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                        
            # First front
            current_front = [i for i in range(n) if n_dom[i] == 0]
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = rank
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                fronts.append(current_front)
                current_front = next_front
                rank += 1
                
            return ranks, fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for points in front_indices.
            Returns a dictionary or array of distances for those indices.
            """
            n_front = len(front_indices)
            if n_front <= 2:
                return {i: float('inf') for i in front_indices}
                
            dist = np.zeros(n_front)
            # Map front indices to local indices
            local_to_global = {local: global_idx for local, global_idx in enumerate(front_indices)}
            
            for m in range(self.n_obj):
                # Sort front by m-th objective
                sorted_locals = np.argsort([F[front_indices[i], m] for i in range(n_front)])
                
                # Assign infinity to extremes
                dist[sorted_locals[0]] = float('inf')
                dist[sorted_locals[-1]] = float('inf')
                
                # Calculate range
                f_min = F[front_indices[sorted_locals[0]], m]
                f_max = F[front_indices[sorted_locals[-1]], m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for k in range(1, n_front - 1):
                    prev_local = sorted_locals[k-1]
                    next_local = sorted_locals[k+1]
                    prev_f = F[front_indices[prev_local], m]
                    next_f = F[front_indices[next_local], m]
                    dist[sorted_locals[k]] += (next_f - prev_f) / f_range
                    
            # Return distances for global indices
            return {front_indices[i]: dist[i] for i in range(n_front)}

        def tournament_select(ranks, crowding, pop_size, n_select):
            """
            Select n_select indices from pop_size candidates using tournament selection.
            candidates are implicitly the current population indices 0..pop_size-1.
            """
            selected = []
            for _ in range(n_select):
                # Pick 2 random individuals
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
                
                # Compare
                if ranks[i1] < ranks[i2]:
                    winner = i1
                elif ranks[i2] < ranks[i1]:
                    winner = i2
                else:
                    # Same rank, compare crowding distance
                    if crowding[i1] > crowding[i2]:
                        winner = i1
                    else:
                        winner = i2
                selected.append(winner)
            return selected

        def sbx_crossover(x1, x2, lo, hi, eta=15.0):
            """
            Simulated Binary Crossover.
            Returns two offspring.
            """
            x1 = x1.copy()
            x2 = x2.copy()
            
            # Crossover probability
            if np.random.rand() < 0.9:
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        # Calculate beta
                        if x1[j] == x2[j]:
                            continue
                            
                        if np.random.rand() <= 0.5:
                            beta = (x1[j] - x2[j]) / (hi[j] - lo[j] + 1e-9)
                        else:
                            beta = (x2[j] - x1[j]) / (hi[j] - lo[j] + 1e-9)
                            
                        if beta <= 0: beta = 1e-9
                        if beta >= 1: beta = 1 - 1e-9
                            
                        if beta <= 0.5:
                            alpha = 2 * beta
                        else:
                            alpha = 2 * (1 - beta)
                            
                        beta_q = ((1 + (1 + 2 * (1 - 1/eta)) * alpha) ** (-1.0 / (1 + 2/eta)))
                        # Wait, standard formula:
                        # beta_q = (0.5 * (1 + (1 + (2/(eta+1)) * alpha)^(-1/(eta+1))))
                        # Let's use a simpler standard implementation:
                        
                        # Recalculate properly
                        if np.random.rand() < 0.5:
                             if x1[j] < x2[j]:
                                 beta = (x2[j] - x1[j]) / (hi[j] - lo[j] + 1e-9)
                             else:
                                 beta = (x1[j] - x2[j]) / (hi[j] - lo[j] + 1e-9)
                        else:
                             if x2[j] < x1[j]:
                                 beta = (x1[j] - x2[j]) / (hi[j] - lo[j] + 1e-9)
                             else:
                                 beta = (x2[j] - x1[j]) / (hi[j] - lo[j] + 1e-9)
                        
                        if beta <= 0: beta = 1e-9
                        if beta >= 1: beta = 1 - 1e-9
                        
                        if beta <= 0.5:
                            alpha = 2 * beta
                        else:
                            alpha = 2 * (1 - beta)
                            
                        beta_q = (0.5 * (1 + (1 + (2/(eta+1)) * alpha) ** (-1.0 / (eta+1))))
                        
                        x1[j] = x1[j] + beta_q * (x2[j] - x1[j])
                        x2[j] = x2[j] - beta_q * (x1[j] - x2[j])
                        
            x1 = np.clip(x1, lo, hi)
            x2 = np.clip(x2, lo, hi)
            return x1, x2

        def polynomial_mutation(x, lo, hi, eta=20.0):
            """
            Polynomial Mutation.
            """
            x = x.copy()
            for j in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    y = x[j]
                    delta1 = (y - lo[j]) / (hi[j] - lo[j] + 1e-9)
                    delta2 = (hi[j] - y) / (hi[j] - lo[j] + 1e-9)
                    uk = np.random.rand()
                    
                    if uk < 0.5:
                        xy = 1 - delta1
                        val = 2 * uk + (1 - 2 * uk) * (xy ** (eta + 1))
                        deltaq = val ** (1.0 / (eta + 1)) - 1
                    else:
                        xy = 1 - delta2
                        val = 2 * (1 - uk) + 2 * (uk - 0.5) * (xy ** (eta + 1))
                        deltaq = 1 - val ** (1.0 / (eta + 1))
                        
                    x[j] = y + deltaq * (hi[j] - lo[j])
                    x[j] = np.clip(x[j], lo, hi)
            return x

        # Main NSGA-II Loop
        for g in range(n_generations):
            # 1. Calculate ranks and crowding distance for current population P
            ranks, fronts = non_dominated_sort(F)
            
            # Calculate crowding distance for all points
            # We need crowding distance for selection.
            # It's usually only calculated for the front where selection happens,
            # but for simplicity in small pop, we can calculate for all or just the first few.
            # Standard NSGA-II calculates it for the fronts involved in selection.
            # Since we select from the whole pop, we need it for all.
            # Optimization: Only calculate for fronts that might be selected?
            # With pop_size 20, it's cheap enough to calculate for all.
            
            crowding = np.zeros(pop_size)
            for front in fronts:
                if not front: continue
                dists = crowding_distance(F, front)
                for idx, d in dists.items():
                    crowding[idx] = d

            # 2. Generate offspring Q
            Q = np.zeros((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                # Select parents
                p1_idx = tournament_select(ranks, crowding, pop_size, 1)[0]
                p2_idx = tournament_select(ranks, crowding, pop_size, 1)[0]
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                Q[i] = c1
                Q[i+1] = c2
            
            # 3. Evaluate Q
            F_Q = np.array([problem(x) for x in Q])
            
            # 4. Combine P and Q into R
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, F_Q])
            
            # 5. Non-dominated sort R
            ranks_R, fronts_R = non_dominated_sort(R_F)
            
            # 6. Select next P from R
            next_P_indices = []
            for front in fronts_R:
                if len(next_P_indices) + len(front) <= pop_size:
                    next_P_indices.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    # Calculate crowding distance for this front in R
                    dists = crowding_distance(R_F, front)
                    # Sort front by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda x: dists[x], reverse=True)
                    needed = pop_size - len(next_P_indices)
                    next_P_indices.extend(sorted_front[:needed])
                    break
            
            # Update P
            X = R_X[next_P_indices]
            F = R_F[next_P_indices]
            
        return None

    def _pareto_front_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

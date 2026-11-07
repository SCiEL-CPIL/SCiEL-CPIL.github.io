import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        # Budget 200. Let's use pop_size 20. 
        # 1 init + 19 generations of 20 offspring = 1 + 19*20 = 381 > 200.
        # So we need to be careful with the loop.
        # Let's use a generation-based loop that stops when budget is exhausted.
        
        pop_size = 20
        if pop_size > budget:
            pop_size = budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        eta_c = 15.0  # Crossover distribution index
        eta_m = 20.0  # Mutation distribution index
        p_c = 0.9     # Crossover probability
        p_m = 1.0 / n # Mutation probability per gene
        
        def fast_non_dominated_sort(F):
            """
            Returns a list of lists, where each list contains indices of non-dominated points
            in that front.
            """
            m = len(F)
            S = [set() for _ in range(m)]  # Dominated set
            n_dom = np.zeros(m, dtype=int)  # Number of dominators
            fronts = [[]]
            
            for i in range(m):
                for j in range(m):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        
            for i in range(m):
                if n_dom[i] == 0:
                    fronts[0].append(i)
                    
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
                
            return fronts[:-1] # Remove the last empty list if any

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of points.
            Returns a dictionary of index -> distance.
            """
            m = len(indices)
            if m == 0:
                return {}
            if m == 1:
                return {indices[0]: np.inf}
                
            dist = np.zeros(m)
            F_sub = F[indices]
            n_obj = F_sub.shape[1]
            
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                if F_sub[order[0], k] == F_sub[order[-1], k]:
                    continue
                for i in range(1, m - 1):
                    dist[order[i]] += (F_sub[order[i+1], k] - F_sub[order[i-1], k]) / (F_sub[order[-1], k] - F_sub[order[0], k])
                    
            return {idx: dist[i] for i, idx in enumerate(indices)}

        def sbx_crossover(x1, x2, lo, hi, eta):
            if np.random.random() < 0.5:
                c1, c2 = x1, x2
            else:
                c1, c2 = x2, x1
                
            for i in range(n):
                if np.random.random() < 0.5:
                    continue
                if abs(c1[i] - c2[i]) < 1e-14:
                    c1[i] = c2[i]
                    continue
                if c1[i] > c2[i]:
                    c1[i], c2[i] = c2[i], c1[i]
                    
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                c1[i] = 0.5 * ((c1[i] + c2[i]) - beta * (c2[i] - c1[i]))
                c2[i] = 0.5 * ((c1[i] + c2[i]) + beta * (c2[i] - c1[i]))
                
                c1[i] = np.clip(c1[i], lo[i], hi[i])
                c2[i] = np.clip(c2[i], lo[i], hi[i])
                
            return c1, c2

        def polynomial_mutation(x, lo, hi, eta):
            for i in range(n):
                if np.random.random() < p_m:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                    u = np.random.random()
                    if u < 0.5:
                        xy = 1 - (1 - delta1) ** (eta + 1)
                        mut_pow = 2.0 - xy
                    else:
                        xy = 1 - (1 - delta2) ** (eta + 1)
                        mut_pow = 2.0 - xy
                    mut_pow = 1.0 / (mut_pow * (eta + 1))
                    mut_pow = (2 * mut_pow + 1) ** (1.0 / (eta + 1))
                    delta = (hi[i] - lo[i]) * mut_pow
                    x[i] += np.random.choice([-1, 1]) * delta
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main loop
        while used < budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # We need to generate pop_size offspring
            # But we might not have enough budget for a full generation
            # So we generate one by one or in small batches?
            # Standard NSGA-II generates a full population of offspring, then combines.
            # If budget is tight, we can just generate as many as we can.
            
            # Let's try to generate a full generation if possible.
            gen_size = min(pop_size, budget - used)
            if gen_size <= 0:
                break
                
            # Selection: Binary tournament on combined current population?
            # No, we select parents from current population.
            
            for _ in range(gen_size):
                # Select parents via tournament
                # Tournament size 2
                i = np.random.randint(0, pop_size)
                j = np.random.randint(0, pop_size)
                
                # We need ranks and crowding for current population
                # This is expensive to compute every time. 
                # Better to compute once per generation.
                pass
                
            # To optimize, let's restructure:
            # 1. Compute ranks and crowding for current population.
            # 2. Generate offspring using tournament selection based on these.
            # 3. Evaluate offspring.
            # 4. Combine, sort, select next population.
            
            # But we might not have budget for a full generation.
            # Let's just generate offspring one by one and add to a pool, 
            # then when we have enough (or budget runs out), do the selection.
            
            # Actually, simpler approach:
            # Keep a list of all evaluated points.
            # At the end, we don't need to maintain a population strictly, 
            # but NSGA-II relies on maintaining a diverse population.
            
            # Let's stick to the standard loop but handle budget carefully.
            
            # Pre-compute ranks and crowding for current X, F
            fronts = fast_non_dominated_sort(F)
            crowding = {}
            for front in fronts:
                crowding.update(crowding_distance(F, front))
            ranks = np.zeros(pop_size, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            def tournament():
                i = np.random.randint(0, pop_size)
                j = np.random.randint(0, pop_size)
                if ranks[i] < ranks[j]:
                    return i
                elif ranks[j] < ranks[i]:
                    return j
                else:
                    if crowding[i] > crowding[j]:
                        return i
                    else:
                        return j

            # Generate offspring
            off_X = []
            off_F = []
            
            for _ in range(gen_size):
                p1 = tournament()
                p2 = tournament()
                
                if np.random.random() < p_c:
                    c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi, eta_c)
                else:
                    c1, c2 = X[p1].copy(), X[p2].copy()
                    
                c1 = polynomial_mutation(c1, lo, hi, eta_m)
                c2 = polynomial_mutation(c2, lo, hi, eta_m)
                
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                off_X.extend([c1, c2])
                off_F.extend([f1, f2])
                
                if used >= budget:
                    break
            
            if not off_X:
                break
                
            # Combine
            X_all = np.vstack([X, np.array(off_X)])
            F_all = np.vstack([F, np.array(off_F)])
            
            # Sort combined population
            fronts_all = fast_non_dominated_sort(F_all)
            
            # Select next population
            X_new = []
            F_new = []
            
            for front in fronts_all:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(X_new)
                    if remaining <= 0:
                        break
                    crowding_front = crowding_distance(F_all, front)
                    # Sort front by crowding distance descending
                    sorted_indices = sorted(front, key=lambda i: crowding_front[i], reverse=True)
                    for idx in sorted_indices[:remaining]:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                    break
                    
            if len(X_new) < pop_size:
                # If we didn't fill the population (shouldn't happen if budget allows), 
                # or if we stopped early due to budget, we just keep what we have.
                # But the loop condition `while used < budget` implies we continue.
                # However, if we break out of the offspring loop due to budget,
                # we should still update the population to the best ones found.
                pass
            
            X = np.array(X_new)
            F = np.array(F_new)
            pop_size = len(X) # Update pop_size if we had to shrink
            
            if used >= budget:
                break
                
        return X

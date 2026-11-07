import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # With 200 evals, N=20 allows 10 generations. N=10 allows 20.
        # Let's use N=20 for a balance of exploration and exploitation.
        N = 20
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Helper functions for NSGA-II (Vectorized where possible)
        
        def non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            F is (N, M) array.
            """
            N = F.shape[0]
            M = F.shape[1]
            domination_count = np.zeros(N, dtype=int)
            dominated_set = [[] for _ in range(N)]
            fronts = []
            
            # Vectorized domination check
            # F[i] <= F[j] and F[i] < F[j]
            # Create a matrix of comparisons
            # This is O(N^2 * M). For N=20, 400*3 = 1200 ops. Very fast.
            
            for i in range(N):
                for j in range(i+1, N):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        domination_count[j] += 1
                        dominated_set[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        domination_count[i] += 1
                        dominated_set[j].append(i)
                    
            current_front = [i for i in range(N) if domination_count[i] == 0]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for indices in F.
            """
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dists = np.zeros(n)
            M = F.shape[1]
            
            for m in range(M):
                # Sort indices by F[:, m]
                sorted_idx = np.argsort(F[indices, m])
                dists[sorted_idx[0]] = np.inf
                dists[sorted_idx[-1]] = np.inf
                
                range_val = F[indices[sorted_idx[-1]], m] - F[indices[sorted_idx[0]], m]
                if range_val == 0:
                    continue
                    
                # Vectorized calculation for middle points
                # dists[sorted_idx[i]] += (F[indices[sorted_idx[i+1]], m] - F[indices[sorted_idx[i-1]], m]) / range_val
                idxs = sorted_idx
                # We need to compute for i from 1 to n-2
                if n > 2:
                    next_vals = F[indices[idxs[2:]], m]
                    prev_vals = F[indices[idxs[:-2]], m]
                    mid_idx = idxs[1:-1]
                    dists[mid_idx] += (next_vals - prev_vals) / range_val
            
            return dists

        def select_population(F):
            """
            Select N best points from F (size 2N) using NSGA-II selection.
            """
            fronts = non_dominated_sort(F)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= N:
                    selected.extend(front)
                else:
                    remaining = N - len(selected)
                    cd = crowding_distance(F, front)
                    # Sort front by crowding distance descending
                    sorted_front = [front[i] for i in np.argsort(cd)[::-1]]
                    selected.extend(sorted_front[:remaining])
                    break
            return selected

        # Evolution Loop
        while evals < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Determine how many offspring to create
            # We create N offspring to maintain population size N
            max_offspring = min(N, budget - evals)
            
            if max_offspring == 0:
                break

            # We need to generate max_offspring children.
            # Standard approach: Binary tournament selection for parents.
            # To vectorize, we can select indices in batch.
            
            # 1. Select parents
            # For each offspring, pick 2 parents via tournament.
            # Tournament size 2.
            # Generate 2 * max_offspring random indices
            p1_idx = np.random.randint(N, size=max_offspring)
            p2_idx = np.random.randint(N, size=max_offspring)
            
            # Evaluate fitness of these candidates to determine winners?
            # We already have F for the current population.
            # So we can just compare F[p1_idx] and F[p2_idx]
            
            # Determine winner for each pair
            # If p1 dominates p2, p1 is winner. Else if p2 dominates p1, p2 is winner.
            # Else random.
            
            f_p1 = F[p1_idx]
            f_p2 = F[p2_idx]
            
            # Vectorized domination check for batch
            # p1 dominates p2: all(f_p1 <= f_p2) and any(f_p1 < f_p2)
            p1_dom_p2 = np.all(f_p1 <= f_p2, axis=1) & np.any(f_p1 < f_p2, axis=1)
            p2_dom_p1 = np.all(f_p2 <= f_p1, axis=1) & np.any(f_p2 < f_p1, axis=1)
            
            # If p1_dom_p2, winner is p1. If p2_dom_p1, winner is p2.
            # If neither (non-dominated), pick random.
            
            # Initialize winners
            winner_idx = np.where(p1_dom_p2, p1_idx, p2_idx)
            # Where p2_dom_p1, ensure p2 is selected (it already is if p1_dom_p2 is false)
            # But if neither, we need random.
            neither = ~(p1_dom_p2 | p2_dom_p1)
            if np.any(neither):
                # Pick random from p1_idx and p2_idx for those
                rand_choice = np.random.random(np.sum(neither)) < 0.5
                # This is tricky to vectorize cleanly without masks.
                # For small N, loop is fine for parent selection.
                for k in range(max_offspring):
                    if neither[k]:
                        if np.random.random() < 0.5:
                            winner_idx[k] = p1_idx[k]
                        else:
                            winner_idx[k] = p2_idx[k]
            
            # Now we have N winners. But SBX needs 2 parents.
            # We can use the winner and a random second parent?
            # Or we can just use the two candidates p1, p2 and let SBX mix them?
            # Standard SBX uses 2 parents.
            # Let's use p1 and p2 as the parents for the children.
            # We will generate 2 children per pair, but we only have budget for N children.
            # So we generate N children, each from a unique pair (p1, p2).
            # We can alternate which child (c1 or c2) we keep, or just take c1.
            # Let's take c1.
            
            x1 = X[p1_idx]
            x2 = X[p2_idx]
            
            # Vectorized SBX
            # u ~ U(0, 1)
            u = np.random.random((max_offspring, dim))
            
            # Calculate beta1
            # beta1 = (2*u)^(1/(eta+1)) if u <= 1/(eta+1)
            # beta1 = (2*(1-u))^(1/(eta+1)) if u >= 1 - 1/(eta+1)
            # beta1 = 1 otherwise
            
            eta = 20.0
            inv_eta = 1.0 / (eta + 1)
            
            beta1 = np.ones_like(u)
            
            mask1 = u <= inv_eta
            mask2 = u >= 1 - inv_eta
            
            beta1[mask1] = (2 * u[mask1]) ** inv_eta
            beta1[mask2] = (2 * (1 - u[mask2])) ** inv_eta
            
            # c1 = 0.5 * ((1 + beta1) * x1 + (1 - beta1) * x2)
            c1 = 0.5 * ((1 + beta1) * x1 + (1 - beta1) * x2)
            
            # Clip to bounds
            c1 = np.clip(c1, lo, hi)
            
            # Mutation
            # Polynomial mutation
            # prob = 1/dim
            # u ~ U(0, 1)
            # delta1 = (2*u)^(1/(eta+1)) - 1 if u <= 0.5
            # delta1 = 1 - (2*(1-u))^(1/(eta+1)) if u > 0.5
            # x_mut = x + delta1 * (hi - lo)
            
            # Apply mutation to c1
            # We decide which dimensions to mutate
            mut_mask = np.random.random((max_offspring, dim)) < (1.0 / dim)
            
            if np.any(mut_mask):
                u_mut = np.random.random((max_offspring, dim))
                
                delta1 = np.zeros_like(u_mut)
                mask_m1 = u_mut <= 0.5
                mask_m2 = u_mut > 0.5
                
                delta1[mask_m1] = (2 * u_mut[mask_m1]) ** inv_eta - 1
                delta1[mask_m2] = 1 - (2 * (1 - u_mut[mask_m2])) ** inv_eta
                
                # Apply only where mut_mask is True
                # c1[mut_mask] += delta1[mut_mask] * (hi - lo)[mut_mask]
                
                # Broadcasting for bounds
                bounds_diff = hi - lo
                # We need to apply delta1 * bounds_diff only where mut_mask is True
                # Since mut_mask is boolean array, we can do:
                # c1 += delta1 * bounds_diff * mut_mask  (This applies 0 where False)
                
                c1 += delta1 * bounds_diff * mut_mask
                c1 = np.clip(c1, lo, hi)
            
            offspring_X = c1
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += max_offspring
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Select next population
            selected_indices = select_population(F_comb)
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]
            
            if evals >= budget:
                break
                
        return X

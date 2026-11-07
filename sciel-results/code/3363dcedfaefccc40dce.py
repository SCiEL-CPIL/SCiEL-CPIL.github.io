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
        
        # NSGA-II Parameters
        # Population size N
        N = 50
        if N > budget:
            N = budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, n))
        F = np.array([problem(x) for x in X])
        used = N
        
        # SBX Crossover and Polynomial Mutation parameters
        eta_c = 20.0
        eta_m = 20.0
        p_m = 1.0 / n
        pc = 0.9 # Crossover probability

        def sbx_crossover(x1, x2, lo, hi, eta):
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(n):
                if np.random.random() < 0.5:
                    continue
                if abs(c1[i] - c2[i]) < 1e-14:
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

        def non_dominated_sort(F):
            """
            Returns list of fronts, each front is a list of indices.
            """
            m = len(F)
            S = [[] for _ in range(m)]  # Dominates set
            n_dom = [0] * m             # Number of dominators
            rank = [0] * m              # Rank (front index)
            
            for i in range(m):
                for j in range(i + 1, m):
                    if all(F[i][k] <= F[j][k] for k in range(n_obj)) and any(F[i][k] < F[j][k] for k in range(n_obj)):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif all(F[j][k] <= F[i][k] for k in range(n_obj)) and any(F[j][k] < F[i][k] for k in range(n_obj)):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(m) if n_dom[i] == 0]
            if not current_front:
                return [[]] # Should not happen usually, but safety
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """
            Returns array of crowding distances for the given indices.
            """
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = len(indices)
            dist = [0.0] * m
            # Sort by each objective
            for k in range(n_obj):
                sorted_idx = sorted(range(m), key=lambda i: F[indices[i]][k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_range = F[indices[sorted_idx[-1]]][k] - F[indices[sorted_idx[0]]][k]
                if f_range == 0:
                    continue
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (F[indices[sorted_idx[i+1]]][k] - F[indices[sorted_idx[i-1]]][k]) / f_range
            return dist

        # Main Loop
        while used < budget:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We can generate up to 2 * (budget - used) / 2 offspring? 
            # Standard NSGA-II generates N offspring to replace N parents.
            # If budget allows, we do a full generation. If not, we do as many as possible.
            # To be safe and efficient, let's generate pairs until budget is low.
            
            # We need to select parents from current X
            # Since we are in the loop, X is the current population of size N
            
            # Determine how many pairs we can evaluate
            max_pairs = (budget - used) // 2
            if max_pairs <= 0:
                break
            
            # We will generate max_pairs offspring pairs
            # We need to select parents. Simple random selection from X for simplicity, 
            # or tournament selection. Tournament is better.
            
            def tournament_select(F, size=2):
                """
                Selects a better individual from a random tournament of 'size' individuals.
                Returns index.
                """
                candidates = np.random.choice(len(F), size=size, replace=False)
                best = candidates[0]
                for c in candidates[1:]:
                    # Check if best dominates c
                    if all(F[best][k] <= F[c][k] for k in range(n_obj)) and any(F[best][k] < F[c][k] for k in range(n_obj)):
                        continue
                    # Check if c dominates best
                    elif all(F[c][k] <= F[best][k] for k in range(n_obj)) and any(F[c][k] < F[best][k] for k in range(n_obj)):
                        best = c
                    else:
                        # Non-dominated, pick random
                        best = np.random.choice([best, c])
                return best

            for _ in range(max_pairs):
                if used + 2 > budget:
                    break
                    
                # Select parents
                p1_idx = tournament_select(F)
                p2_idx = tournament_select(F)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover
                c1, c2 = sbx_crossover(x1, x2, lo, hi, eta_c)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi, eta_m)
                c2 = polynomial_mutation(c2, lo, hi, eta_m)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
            if not offspring_X:
                break

            # Combine parent and offspring
            # X is (N, n), F is (N, 2)
            # Offspring is (2*max_pairs, n)
            
            # If we generated more than N offspring, we only need N to form the next generation.
            # If we generated less, we fill the rest with parents? 
            # Standard NSGA-II: R = P + Q. Select N from R.
            # Here, if max_pairs * 2 < N, we have fewer offspring than parents.
            # We should just combine and select.
            
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.array([np.array(F)] + [np.array(offspring_F)])
            # Fix: F is already (N, 2). np.array(F) makes it (N, 2).
            # We need to stack correctly.
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            total_size = len(F_combined)
            
            # Non-dominated sorting
            fronts = non_dominated_sort(F_combined)
            
            # Select next generation
            next_X = []
            next_F = []
            
            for front in fronts:
                if len(next_F) + len(front) <= N:
                    # Add all
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Need to select some from this front based on crowding distance
                    needed = N - len(next_F)
                    if needed <= 0:
                        break
                    
                    # Calculate crowding distance for this front
                    cd = crowding_distance(F_combined, front)
                    
                    # Sort indices by crowding distance (descending)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    
                    for i in range(needed):
                        idx = front[sorted_front[i]]
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break
            
            # Update population
            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)
            else:
                # Fallback if something went wrong (should not happen)
                break

        return X

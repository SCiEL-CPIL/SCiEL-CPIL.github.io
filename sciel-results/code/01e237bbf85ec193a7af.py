import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0  # safety margin before 330s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x), dtype=float)
            used += 1
            return f

        # Vectorized dominance check
        # Returns array of booleans where True means i dominates j
        # A dominates B if A <= B in all objectives and A < B in at least one
        def dominance_matrix(F1, F2):
            # F1: (N, M), F2: (M, K)
            # less_equal: (N, K)
            less_equal = np.all(F1[:, :, None] <= F2[None, :, :], axis=1)
            less = np.any(F1[:, :, None] < F2[None, :, :], axis=1)
            return less_equal & less

        def fast_nondominated_sort(F):
            """
            F: (N, M) array of objective vectors
            Returns: list of fronts, each front is a list of indices
            """
            N = len(F)
            if N == 0:
                return []
            
            S = [[] for _ in range(N)]  # S[i] = set of indices dominated by i
            n_dom = np.zeros(N, dtype=int) # n_dom[i] = number of solutions dominating i
            fronts = []
            
            # O(N^2) dominance checks
            # For large N this is slow, but N=200 is fine.
            # We can vectorize the inner loop.
            # F[i] vs F[j]
            # F1 = F[i], F2 = F[j]
            
            # Vectorized: 
            # F1: (N, M, 1), F2: (1, M, N) -> (N, N, M)
            # This uses N^2 * M memory. 200^2 * 2 = 80,000 floats. Very safe.
            
            # less_equal[i, j] is True if F[i] <= F[j] in all objectives
            # less[i, j] is True if F[i] < F[j] in at least one objective
            
            le = np.all(F[:, :, None] <= F[None, :, :], axis=1) # (N, N)
            lt = np.any(F[:, :, None] < F[None, :, :], axis=1)  # (N, N)
            
            # i dominates j if le[i,j] and lt[i,j]
            dom = le & lt
            
            for i in range(N):
                # Find j such that i dominates j
                dominated_by_i = np.where(dom[i, :])[0]
                S[i] = dominated_by_i.tolist()
                
                # Find k such that k dominates i
                # This is column i of dom
                n_dom[i] = np.sum(dom[:, i])
                
                if n_dom[i] == 0:
                    fronts.append([i])
            
            # Now build fronts
            if not fronts:
                return []
                
            current_front = fronts[0]
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            """
            F: (N, M)
            front_indices: list of indices in this front
            Returns: dict mapping index to distance
            """
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            distances = {i: 0.0 for i in front_indices}
            M = len(front_indices[0]) # Number of objectives
            
            for m in range(M):
                # Sort front by objective m
                sorted_front = sorted(front_indices, key=lambda idx: F[idx, m])
                distances[sorted_front[0]] = float('inf')
                distances[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for k in range(1, len(sorted_front) - 1):
                    idx = sorted_front[k]
                    distances[idx] += (F[sorted_front[k+1], m] - F[sorted_front[k-1], m]) / f_range
            
            return distances

        # Initialization: Random sampling
        N_pop = 50  # Population size
        # Use LHS or just random? Random is fine for NSGA-II.
        # Let's use a slightly structured init if possible, but random is standard.
        
        # We have 200 budget.
        # If we run NSGA-II for T generations with N_pop=50, that's 50*T evaluations.
        # 200 / 50 = 4 generations. That's very few.
        # Maybe smaller population? N_pop=20? 200/20 = 10 generations.
        # Or N_pop=10? 200/10 = 20 generations.
        
        # Let's try N_pop=20.
        N_pop = 20
        
        # Initial population
        X = lo + (hi - lo) * np.random.random((N_pop, dim))
        F = np.array([evaluate(x) for x in X])
        
        all_F.extend(F.tolist())
        
        # NSGA-II Loop
        while used < budget and time.time() < deadline:
            # Create offspring
            # Tournament selection
            def tournament_select(F, fronts, crowd_dists):
                # fronts: list of lists of indices
                # crowd_dists: dict of dicts? Or list of dicts?
                # Let's maintain a single crowd_dist array for current population
                pass
            
            # Standard NSGA-II implementation structure:
            # 1. Sort population into fronts
            # 2. Assign crowding distances
            # 3. Select parents via tournament
            # 4. Create offspring via crossover/mutation
            # 5. Evaluate offspring
            # 6. Combine P and Q
            # 7. Sort combined into fronts
            # 8. Select next P from combined
            
            # We need to store X and F for current population
            # Let's restructure to keep X and F as arrays.
            
            # Current state: X (N_pop, dim), F (N_pop, M)
            
            # 1. Fast non-dominated sort
            fronts = fast_nondominated_sort(F)
            
            # 2. Crowding distances
            # We need to store distances for all points
            # Initialize with 0
            crowd_dist = np.zeros(N_pop)
            for front in fronts:
                dists = crowding_distance(F, front)
                for idx in front:
                    crowd_dist[idx] = dists[idx]
            
            # 3. Selection
            # We need to select 2 parents per offspring
            # Total offspring = N_pop
            # So we need 2 * N_pop parents
            
            def select_parent():
                # Tournament of size 2
                i1 = np.random.randint(N_pop)
                i2 = np.random.randint(N_pop)
                
                # Compare i1 and i2
                # Find front number
                front_num = np.full(N_pop, -1)
                for k, front in enumerate(fronts):
                    for idx in front:
                        front_num[idx] = k
                
                # Lower front number is better
                if front_num[i1] < front_num[i2]:
                    return i1
                elif front_num[i2] < front_num[i1]:
                    return i2
                else:
                    # Same front, higher crowding distance is better
                    if crowd_dist[i1] > crowd_dist[i2]:
                        return i1
                    else:
                        return i2

            offspring_X = []
            offspring_F = []
            
            for _ in range(N_pop):
                if used >= budget or time.time() > deadline:
                    break
                    
                p1 = select_parent()
                p2 = select_parent()
                
                # Crossover: Simulated Binary Crossover (SBX) or just Uniform?
                # For 4D, simple uniform crossover or blend might be enough.
                # Let's use a simple mix.
                # Actually, for 4 variables, a simple local search or random restart might be better.
                # But let's stick to NSGA-II structure.
                
                # SBX is complex. Let's use a simple crossover:
                # For each dimension, pick from p1 or p2
                # Or average + noise?
                # Let's use: child = alpha * p1 + (1-alpha) * p2 + noise
                alpha = np.random.random()
                child = alpha * X[p1] + (1 - alpha) * X[p2]
                
                # Mutation: Gaussian
                # Scale mutation based on bounds
                # sigma = 0.1 * (hi - lo)
                child = child + np.random.randn(dim) * 0.1 * (hi - lo)
                child = np.clip(child, lo, hi)
                
                offspring_X.append(child)
                
                f = evaluate(child)
                offspring_F.append(f)
                
                # If we run out of budget, break
                if used >= budget:
                    break

            if not offspring_F:
                break

            # 5. Combine
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, np.array(offspring_F)])
            
            all_F.extend(offspring_F)
            
            # 6. Sort combined
            fronts_comb = fast_nondominated_sort(F_comb)
            
            # 7. Select next population
            new_X = []
            new_F = []
            new_fronts = []
            
            # We need to pick N_pop individuals from X_comb
            # Iterate through fronts
            for front in fronts_comb:
                if len(new_X) + len(front) <= N_pop:
                    # Add all
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Need to select some from this front based on crowding distance
                    # Calculate crowding distance for this front
                    dists = crowding_distance(F_comb, front)
                    # Sort front by distance descending
                    sorted_front = sorted(front, key=lambda idx: dists[idx], reverse=True)
                    
                    # Take only as many as needed
                    needed = N_pop - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break
            
            if len(new_X) == 0:
                break
                
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Update all_F? No, we already added offspring.
            # Note: Dominated points in P are discarded, but they remain in all_F.
            # This is correct for hypervolume calculation of *all* recorded vectors.

        return all_F

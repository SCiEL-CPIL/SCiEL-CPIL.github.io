"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II Multi-Objective Optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We return early at 300s to be safe.
        deadline = start_time + 300
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size for NSGA-II
        # With budget 200, we can't have a huge population. 
        # Typical NSGA-II uses population size N.
        # Let's try N=40. This allows for about (200-40)/40 = 4 generations.
        # This is low for evolution, but better than local search for hypervolume.
        # We will use a smaller population to allow more generations? 
        # Or just run as many generations as budget allows.
        pop_size = 40
        
        # 1. Initialization
        # Use random initialization for the initial population
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        
        eval_count = pop_size
        
        # Helper function to evaluate a population
        def evaluate_population(pop):
            nonlocal eval_count
            f_vals = np.array([problem(x) for x in pop])
            eval_count += len(pop)
            return f_vals

        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            M = len(F)
            S = [[] for _ in range(M)]
            n = [0] * M
            fronts = [[]]
            
            for p in range(M):
                for q in range(M):
                    if p == q:
                        continue
                    # Check if p dominates q
                    # p dominates q if p <= q and p != q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
                if n[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1] # Remove the last empty list

        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            if not indices:
                return {}
            M = len(indices)
            if M <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_idx = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_min = F[sorted_idx[0]][m]
                f_max = F[sorted_idx[-1]][m]
                if f_max - f_min < 1e-15:
                    continue
                for k in range(1, M - 1):
                    dist[sorted_idx[k]] += (F[sorted_idx[k+1]][m] - F[sorted_idx[k-1]][m]) / (f_max - f_min)
            return dist

        # Helper: Binary Tournament Selection
        def tournament_select(pop, F, fronts):
            # Assign rank to each individual
            rank = [0] * len(pop)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Select two random individuals
            i1 = np.random.randint(0, len(pop))
            i2 = np.random.randint(0, len(pop))
            
            if rank[i1] < rank[i2]:
                return i1
            elif rank[i2] < rank[i1]:
                return i2
            else:
                # Same rank, use crowding distance
                # We need dist for the front they are in
                # For simplicity in tournament, we can just pick randomly if rank is same,
                # or precompute dist for all fronts. Precomputing is better.
                # But tournament is called many times. Let's precompute dist for all fronts in the main loop.
                # For now, simple random pick if rank is tied.
                return i1 if np.random.rand() < 0.5 else i2

        # Main NSGA-II Loop
        while eval_count < self.budget:
            if time.time() > deadline - 10:
                break
                
            # Determine how many offspring we can generate
            # We generate a population of offspring of size pop_size
            # If remaining budget < pop_size, we generate as many as possible?
            # NSGA-II usually works with fixed sizes. 
            # If budget is tight, we might just do one final generation.
            if self.budget - eval_count < pop_size:
                # Generate remaining individuals
                n_offspring = self.budget - eval_count
                offspring_X = lo + (hi - lo) * np.random.rand(n_offspring, self.dim)
                # Simple mutation/crossover not really applicable for just a few points without parents
                # Just random sampling for the tail end
                offspring_F = evaluate_population(offspring_X)
                # Combine and select best? Or just keep them?
                # The task records ALL evaluations. So we just need to make the calls.
                # The "best" set isn't returned, only the recorded points matter for hypervolume.
                # So we don't strictly need to maintain a population for the *next* step if we are done.
                # But if we have more budget, we need to continue evolving.
                
                if n_offspring > 0:
                    # We have already evaluated them.
                    pass
                break
            
            # 1. Fast Non-Dominated Sort on current population
            fronts = fast_non_dominated_sort(F)
            
            # Precompute crowding distance for each front
            front_dists = []
            for front in fronts:
                front_dists.append(crowding_distance(F, front))
            
            # 2. Create Offspring Population
            offspring_X = np.zeros((pop_size, self.dim))
            
            # Binary Tournament Selection and Crossover/Mutation
            for i in range(pop_size):
                # Select parents
                p1_idx = tournament_select(X, F, fronts)
                p2_idx = tournament_select(X, F, fronts)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Simulated Binary Crossover (SBX) or Simple Blend
                # Simple Blend Crossover
                alpha = np.random.rand(self.dim)
                child = alpha * p1 + (1 - alpha) * p2
                child = np.clip(child, lo, hi)
                
                # Polynomial Mutation or Gaussian Mutation
                # Gaussian mutation with small sigma
                sigma = 0.1 * (hi - lo)
                if np.random.rand() < 0.5: # Mutation rate 0.5 per individual? Or per gene?
                    # Per gene mutation
                    mask = np.random.rand(self.dim) < 0.1
                    child[mask] += np.random.randn(np.sum(mask)) * sigma[mask]
                    child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
            
            # 3. Evaluate Offspring
            offspring_F = evaluate_population(offspring_X)
            
            # 4. Combine Populations
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # 5. Selection
            # Sort combined population
            fronts_combined = fast_non_dominated_sort(F_combined)
            
            # Select best pop_size individuals
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            new_idx = 0
            
            for i, front in enumerate(fronts_combined):
                if new_idx + len(front) <= pop_size:
                    # Take all individuals in this front
                    for idx in front:
                        new_X[new_idx] = X_combined[idx]
                        new_F[new_idx] = F_combined[idx]
                        new_idx += 1
                else:
                    # Need to select from this front based on crowding distance
                    dists = crowding_distance(F_combined, front)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda idx: dists[idx], reverse=True)
                    needed = pop_size - new_idx
                    for k in range(needed):
                        idx = sorted_front[k]
                        new_X[new_idx] = X_combined[idx]
                        new_F[new_idx] = F_combined[idx]
                        new_idx += 1
                    break
            
            # Update population
            X = new_X
            F = new_F

        # The return value is ignored, but we must return something valid
        if len(F) > 0:
            return X[0]
        return lo

# EVOLVE-BLOCK-END

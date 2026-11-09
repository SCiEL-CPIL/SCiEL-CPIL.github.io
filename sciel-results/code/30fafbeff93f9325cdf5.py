"""Multi-objective optimizer on RE34 using NSGA-II style approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with DE-style crossover and mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Use the evaluator's seed for reproducibility
        rng = np.random.default_rng(0)

        # Population size: balance exploration and exploitation
        pop_size = min(50, max(20, budget // 4))
        # Ensure we have enough evaluations
        # Initial population + generations * offspring
        # Let's use pop_size as both parent and offspring size
        # Total evaluations = pop_size (init) + num_generations * pop_size (offspring)
        # We need to stay within budget
        # Let's compute how many generations we can afford
        # But first, let's just use a fixed population and iterate

        # Initialize population
        X = lo + range_ * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated solutions
        # We'll maintain the population and use NSGA-II selection

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_doms = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1
            
            # First front
            front = [i for i in range(n) if n_doms[i] == 0]
            if not front:
                front = [0]  # Fallback
            
            fronts.append(front)
            
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front
            
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min = f_vals.min()
                f_max = f_vals.max()
                if f_max - f_min < 1e-12:
                    continue
                # Sort by objective m
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[order[i]] += (f_vals[order[i + 1]] - f_vals[order[i - 1]]) / (f_max - f_min)
            
            return dist

        def tournament_select(F, fronts, pop_size):
            """Tournament selection based on rank and crowding distance."""
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distance per front
            crowd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        crowd[idx] = cd[j]
            
            # Tournament selection
            selected = []
            for _ in range(pop_size):
                # Pick two random candidates
                i1 = rng.integers(len(F))
                i2 = rng.integers(len(F))
                # Compare by rank, then crowding distance
                if rank[i1] < rank[i2]:
                    selected.append(i1)
                elif rank[i2] < rank[i1]:
                    selected.append(i2)
                else:
                    if crowd[i1] >= crowd[i2]:
                        selected.append(i1)
                    else:
                        selected.append(i2)
            
            return selected

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Select parents
            parents = tournament_select(F, fronts, pop_size)
            
            # Generate offspring using DE-style crossover
            offspring_X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                # Pick 3 distinct parents
                p1 = parents[i]
                # Pick two more distinct from the selected
                other_indices = [j for j in range(pop_size) if j != p1]
                p2 = other_indices[rng.integers(len(other_indices))]
                other_indices.remove(p2)
                p3 = other_indices[rng.integers(len(other_indices))]
                
                # DE crossover: child = x_p1 + F * (x_p2 - x_p3)
                F_factor = 0.5
                child = X[p1] + F_factor * (X[p2] - X[p3])
                
                # Mutation
                child += 0.1 * range_ * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine parent and offspring populations
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined population
            fronts_all = fast_non_dominated_sort(F_all)
            
            # Select next population
            next_X = []
            next_F = []
            for front in fronts_all:
                if len(next_X) + len(front) <= pop_size:
                    # Add entire front
                    for idx in front:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F_all, front)
                    # Sort by crowding distance (descending)
                    order = np.argsort(-cd)
                    for j in range(min(remaining, len(front))):
                        idx = front[order[j]]
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    break
            
            if len(next_X) == 0:
                # Fallback: keep current population
                break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X

# EVOLVE-BLOCK-END

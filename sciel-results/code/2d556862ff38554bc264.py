"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Fast EMO optimizer with adaptive mutation and efficient dominance sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominance_sort(self, F):
        """
        Fast non-dominated sorting.
        F: list of objective vectors (list of lists or numpy array).
        Returns: list of fronts, where each front is a list of indices.
        """
        n = len(F)
        # Convert to list of lists for faster indexing if it's a numpy array
        if isinstance(F, np.ndarray):
            F = F.tolist()
        
        S = [[] for _ in range(n)]  # S[i] = list of indices dominated by i
        nd = [0] * n                # nd[i] = number of solutions dominating i
        ranks = [0] * n             # rank of each solution

        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                # i dominates j if i <= j in all objectives and i < j in at least one
                i_dom_j = True
                j_dom_i = True
                for k in range(self.n_obj):
                    if F[i][k] > F[j][k]:
                        i_dom_j = False
                    if F[i][k] < F[j][k]:
                        j_dom_i = False
                
                if i_dom_j:
                    S[i].append(j)
                    nd[j] += 1
                if j_dom_i:
                    S[j].append(i)
                    nd[i] += 1

        # Initialize first front
        front = [i for i in range(n) if nd[i] == 0]
        fronts = []
        if front:
            fronts.append(front)
            for i in front:
                ranks[i] = 0

        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
            if next_front:
                fronts.append(next_front)
            front = next_front
            k += 1

        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of indices in F.
        Returns a list of distances corresponding to the indices.
        """
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n

        # Convert F to list of lists if numpy
        if isinstance(F, np.ndarray):
            F = F.tolist()

        dist = [0.0] * n
        m = self.n_obj

        for k in range(m):
            # Sort indices by objective k
            sorted_indices = sorted(indices, key=lambda i: F[i][k])
            
            dist[sorted_indices[0]] = float('inf')
            dist[sorted_indices[-1]] = float('inf')
            
            f_min = F[sorted_indices[0]][k]
            f_max = F[sorted_indices[-1]][k]
            
            if f_max == f_min:
                continue

            for i in range(1, n - 1):
                idx = sorted_indices[i]
                dist[idx] += (F[sorted_indices[i + 1]][k] - F[sorted_indices[i - 1]][k]) / (f_max - f_min)

        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, leave 20s buffer
        deadline = start_time + 280
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Larger population to better cover the front
        pop_size = 100
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Adaptive mutation sigma
        sigma = 0.2
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = self._dominance_sort(F)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size
            
            while len(offspring_X) < n_offspring:
                if time.time() > deadline:
                    break
                
                # Select 2 parents from the best fronts
                parents = []
                for front in fronts:
                    if len(parents) + len(front) >= 2:
                        # Calculate crowding distance for this front
                        cd = self._crowding_distance(F, front)
                        front_indices = np.array(front)
                        cd_arr = np.array(cd)
                        # Sort by crowding distance descending
                        order = np.argsort(-cd_arr)
                        # Pick top 2 (or fewer if front is small)
                        for idx in order:
                            if len(parents) < 2:
                                parents.append(front_indices[idx])
                            else:
                                break
                        break
                    else:
                        parents.extend(front)
                        if len(parents) >= 2:
                            break
                
                if len(parents) < 2:
                    # Fallback: random selection
                    parents = list(rng.choice(pop_size, 2, replace=False))
                
                p1, p2 = parents[0], parents[1]
                
                # SBX Crossover
                child1 = X[p1].copy()
                child2 = X[p2].copy()
                
                if rng.random() < 0.9:
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / 20)
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / 20)
                            child1[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                            child2[d] = 0.5 * ((1 - beta) * X[p1][d] + (1 + beta) * X[p2][d])
                
                # Adaptive Mutation
                # Start with larger steps, shrink over time
                # We have 200 evals, pop 100, so roughly 100 offspring generated.
                # Let's scale sigma based on used budget.
                current_sigma = 0.2 * (1.0 - used / self.budget) + 0.05
                
                for child in [child1, child2]:
                    if rng.random() < 0.9:
                        # Gaussian mutation
                        child = child + current_sigma * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)
                
                offspring_X.append(child1)
                offspring_F.append(problem(child1))
                used += 1
                if used >= self.budget or time.time() > deadline:
                    break
                
                offspring_X.append(child2)
                offspring_F.append(problem(child2))
                used += 1
            
            if not offspring_X:
                break
            
            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort combined
            fronts_comb = self._dominance_sort(F_comb)
            
            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    remaining = pop_size - len(X_new)
                    cd = self._crowding_distance(F_comb, front)
                    front_indices = np.array(front)
                    cd_arr = np.array(cd)
                    order = np.argsort(-cd_arr)
                    for idx in order[:remaining]:
                        X_new.append(X_comb[front_indices[idx]])
                        F_new.append(F_comb[front_indices[idx]])
                    break
            
            if X_new:
                X = np.array(X_new)
                F = np.array(F_new)
        
        return X
# EVOLVE-BLOCK-END

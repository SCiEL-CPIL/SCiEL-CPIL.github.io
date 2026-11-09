"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II-like multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = min(20, budget)  # Small population for small budget
        
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        X = np.random.uniform(lo, hi, (self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations = self.pop_size
        
        def fast_non_dominated_sort(F):
            """
            Returns a list of fronts. Front 0 is the non-dominated set, etc.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominating = [0] * n
            rank = [0] * n
            front = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominating[i] += 1
                    else:
                        # If neither dominates the other, they might be in the same front later
                        pass
            
            for i in range(n):
                if n_dominating[i] == 0:
                    rank[i] = 0
                    front.append(i)
                    
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                front = next_front
                
            # Group by rank
            fronts = []
            max_rank = max(rank) if rank else 0
            for r in range(max_rank + 1):
                fronts.append([i for i in range(n) if rank[i] == r])
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices.
            """
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            m = self.n_obj
            dist = np.zeros(n)
            F_subset = F[indices]
            
            for j in range(m):
                order = np.argsort(F_subset[:, j])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                if F_subset[order[-1], j] != F_subset[order[0], j]:
                    for k in range(1, n - 1):
                        dist[order[k]] += (F_subset[order[k+1], j] - F_subset[order[k-1], j]) / (F_subset[order[-1], j] - F_subset[order[0], j])
                        
            # Map back to original indices
            final_dist = np.zeros(len(indices))
            for i, idx in enumerate(indices):
                final_dist[i] = dist[i]
            return final_dist

        def combine_population(X1, F1, X2, F2):
            """
            Combine two populations and select the best non-dominated set.
            """
            Xc = np.vstack([X1, X2])
            Fc = np.vstack([F1, F2])
            fronts = fast_non_dominated_sort(Fc)
            
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= self.pop_size:
                    for idx in front:
                        X_new.append(Xc[idx])
                        F_new.append(Fc[idx])
                else:
                    # Need to truncate this front using crowding distance
                    needed = self.pop_size - len(X_new)
                    cd = crowding_distance(Fc, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for i in range(needed):
                        idx = front[order[i]]
                        X_new.append(Xc[idx])
                        F_new.append(Fc[idx])
                    break
                    
            return np.array(X_new), np.array(F_new)

        def sbx_crossover(x1, x2, lo, hi):
            """Simulated Binary Crossover."""
            if np.random.rand() < 0.5:
                return x1, x2
                
            x1_new = np.zeros_like(x1)
            x2_new = np.zeros_like(x2)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    x1_new[i] = x1[i]
                    x2_new[i] = x2[i]
                else:
                    if abs(x1[i] - x2[i]) < 1e-10:
                        x1_new[i] = x1[i]
                        x2_new[i] = x2[i]
                    else:
                        if np.random.rand() < 0.5:
                            beta = (2 * (x1[i] - x2[i]) / (hi[i] - lo[i])) ** (1 / 21)
                        else:
                            beta = (2 * (x2[i] - x1[i]) / (hi[i] - lo[i])) ** (1 / 21)
                        x1_new[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        x2_new[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        x1_new[i] = np.clip(x1_new[i], lo[i], hi[i])
                        x2_new[i] = np.clip(x2_new[i], lo[i], hi[i])
            return x1_new, x2_new

        def polynomial_mutation(x, lo, hi):
            """Polynomial Mutation."""
            x_new = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    if np.random.rand() < 0.5:
                        delta = (x_new[i] - lo[i]) / (hi[i] - lo[i])
                        x_new[i] = x_new[i] - (hi[i] - lo[i]) * (2 * delta) ** (1/21) * np.random.rand()
                    else:
                        delta = (hi[i] - x_new[i]) / (hi[i] - lo[i])
                        x_new[i] = x_new[i] + (hi[i] - lo[i]) * (2 * delta) ** (1/21) * np.random.rand()
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        # Main loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            X_off = []
            F_off = []
            
            for _ in range(self.pop_size // 2):
                # Tournament selection
                idx1 = np.random.choice(self.pop_size)
                idx2 = np.random.choice(self.pop_size)
                # Simple tournament: pick random, if one dominates the other pick it, else random
                # For speed, we can just pick two random and use the one with better rank if we had ranks, 
                # but here we just do a simple binary tournament on dominance
                # Actually, let's just pick two random parents for simplicity and speed, 
                # or implement a proper tournament. 
                # Let's do a proper tournament of size 2 based on non-domination and crowding.
                # To keep it fast, we'll just pick two random individuals.
                # A better way: pick two random, evaluate dominance.
                p1_idx = np.random.choice(self.pop_size)
                p2_idx = np.random.choice(self.pop_size)
                
                # If p1 dominates p2, p1 is better. If p2 dominates p1, p2 is better.
                # If neither, pick random.
                if np.all(F[p1_idx] <= F[p2_idx]) and np.any(F[p1_idx] < F[p2_idx]):
                    parent1 = X[p1_idx]
                elif np.all(F[p2_idx] <= F[p1_idx]) and np.any(F[p2_idx] < F[p1_idx]):
                    parent1 = X[p2_idx]
                else:
                    parent1 = X[np.random.choice([p1_idx, p2_idx])]
                    
                # Second parent
                p3_idx = np.random.choice(self.pop_size)
                p4_idx = np.random.choice(self.pop_size)
                if np.all(F[p3_idx] <= F[p4_idx]) and np.any(F[p3_idx] < F[p4_idx]):
                    parent2 = X[p3_idx]
                elif np.all(F[p4_idx] <= F[p3_idx]) and np.any(F[p4_idx] < F[p3_idx]):
                    parent2 = X[p4_idx]
                else:
                    parent2 = X[np.random.choice([p3_idx, p4_idx])]
                
                child1, child2 = sbx_crossover(parent1, parent2, lo, hi)
                child1 = polynomial_mutation(child1, lo, hi)
                child2 = polynomial_mutation(child2, lo, hi)
                
                f1 = problem(child1)
                evaluations += 1
                f2 = problem(child2)
                evaluations += 1
                
                X_off.append(child1)
                F_off.append(f1)
                X_off.append(child2)
                F_off.append(f2)
                
                if evaluations >= self.budget:
                    break
            if evaluations >= self.budget:
                break
                
            X_off = np.array(X_off)
            F_off = np.array(F_off)
            
            # Combine and select
            X, F = combine_population(X, F, X_off, F_off)
            
        return X
# EVOLVE-BLOCK-END

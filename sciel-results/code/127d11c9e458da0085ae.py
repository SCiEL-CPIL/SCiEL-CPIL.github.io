"""Optimized program for WFG9 using a two-phase strategy:
Phase 1: NSGA-II with crowding distance for diversity
Phase 2: Local search refinement on the best non-dominated solutions
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II + Local Search Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        pop_size = 40
        
        # Initialize population
        X = lo + np.random.rand(pop_size, self.dim) * range_
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # NSGA-II parameters
        n_gen_max = 40
        crossover_prob = 0.9
        mutation_prob = 0.1
        mutation_scale = 0.5
        
        def dominates(a, b):
            """Check if a dominates b (minimization)."""
            return (a <= b).all() and (a < b).any()
        
        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each front is list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = [0] * n  # n_d[i] = number of solutions that dominate i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            # First front
            front = []
            for i in range(n):
                if n_d[i] == 0:
                    front.append(i)
                    fronts.append(front)
                    front = []
            
            # Subsequent fronts
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            
            return fronts
        
        def crowding_distance(F, front_indices):
            """Calculate crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            F_front = F[front_indices]
            
            for m in range(self.n_obj):
                # Sort by objective m
                order = np.argsort(F_front[:, m])
                dist[order[0]] = float('inf')
                dist[order[-1]] = float('inf')
                
                f_range = F_front[order[-1], m] - F_front[order[0], m]
                if f_range < 1e-10:
                    continue
                
                for i in range(1, n - 1):
                    dist[order[i]] += (F_front[order[i + 1], m] - F_front[order[i - 1], m]) / f_range
            
            return dist
        
        # Main NSGA-II loop
        for gen in range(n_gen_max):
            if time.time() > deadline:
                break
            if evaluations >= self.budget:
                break
            
            # Combine parents and offspring
            # Create offspring of same size as population
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)
            
            n_offspring = min(pop_size, self.budget - evaluations)
            
            for i in range(n_offspring):
                if time.time() > deadline:
                    break
                if evaluations >= self.budget:
                    break
                
                # Binary tournament selection
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                
                # Get ranks
                fronts = fast_non_dominated_sort(F)
                rank = np.zeros(pop_size, dtype=int)
                for r, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = r
                
                if rank[idx1] < rank[idx2]:
                    parent = X[idx1]
                elif rank[idx2] < rank[idx1]:
                    parent = X[idx2]
                else:
                    # Same rank, use crowding distance
                    front_idx = rank[idx1]
                    front = fronts[front_idx]
                    cd = crowding_distance(F, front)
                    cd_map = dict(zip(front, cd))
                    if cd_map[idx1] >= cd_map[idx2]:
                        parent = X[idx1]
                    else:
                        parent = X[idx2]
                
                # SBX crossover (simplified: blend)
                if np.random.rand() < crossover_prob:
                    beta = np.random.rand(self.dim)
                    child1 = (1 - beta) * parent + beta * X[np.random.randint(pop_size)]
                    child2 = beta * parent + (1 - beta) * X[np.random.randint(pop_size)]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                else:
                    child1 = parent.copy()
                    child2 = parent.copy()
                
                # Polynomial mutation
                if np.random.rand() < mutation_prob:
                    child1 += mutation_scale * np.random.randn(self.dim) * range_
                    child1 = np.clip(child1, lo, hi)
                
                offspring_X[i] = child1
                offspring_F[i] = problem(child1)
                evaluations += 1
            
            # Combine parent and offspring
                X_combined = np.vstack([X, offspring_X[:n_offspring]])
                F_combined = np.vstack([F, offspring_F[:n_offspring]])
            
            # Non-dominated sorting on combined population
            fronts = fast_non_dominated_sort(F_combined)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(cd)[::-1]  # Sort by crowding distance descending
                    for idx in order[:remaining]:
                        new_X.append(X_combined[front[idx]])
                        new_F.append(F_combined[front[idx]])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        # Phase 2: Local search on the best non-dominated solutions
        if evaluations < self.budget and time.time() < deadline:
            # Get the first front (non-dominated solutions)
            fronts = fast_non_dominated_sort(F)
            if fronts:
                first_front = fronts[0]
                n_local = min(len(first_front), self.budget - evaluations)
                
                # Local search parameters
                ls_radius = 0.1 * range_
                n_ls_iter = 5
                
                for idx in first_front[:n_local]:
                    if evaluations >= self.budget or time.time() > deadline:
                        break
                    
                    best_x = X[idx].copy()
                    best_f = F[idx].copy()
                    
                    for _ in range(n_ls_iter):
                        if evaluations >= self.budget or time.time() > deadline:
                            break
                        
                        # Perturb
                        perturbation = np.random.randn(self.dim) * ls_radius
                        candidate = best_x + perturbation
                        candidate = np.clip(candidate, lo, hi)
                        
                        f_candidate = problem(candidate)
                        evaluations += 1
                        
                        # Accept if it doesn't dominate the current best and is non-dominated
                        if not dominates(f_candidate, best_f):
                            # Check if candidate is non-dominated by current best
                            if not dominates(best_f, f_candidate):
                                best_x = candidate
                                best_f = f_candidate
                    
                    # Update population
                    X[idx] = best_x
                    F[idx] = best_f
        
        # Return the best non-dominated solutions
        return X
# EVOLVE-BLOCK-END

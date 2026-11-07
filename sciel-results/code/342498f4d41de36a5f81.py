"""Optimized program for WFG9 using a hybrid evolutionary approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Strategy:
        # 1. Initialize with a diverse population spread across the search space.
        # 2. Use a hybrid of NSGA-II (for diversity) and local search (for convergence).
        # 3. Keep an external archive of non-dominated solutions.
        # 4. Use adaptive mutation/crossover rates.
        
        pop_size = 40
        archive = []  # List of (x, f)
        
        # Initialize population
        X = np.random.uniform(lo, hi, (pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # Initialize archive
        for i in range(pop_size):
            self._add_to_archive(archive, X[i], F[i])
        
        # Parameters
        mutation_rate = 0.3
        crossover_rate = 0.9
        f_mutation = 0.5
        f_crossover = 0.5
        
        # Main loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Binary Tournament Selection
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                winner = self._tournament_select(F[idx1], F[idx2], F)
                
                # Crossover (SBX-like or Simple Blend)
                if np.random.rand() < crossover_rate:
                    # Pick a second parent
                    idx3 = np.random.randint(pop_size)
                    p1 = X[winner]
                    p2 = X[idx3]
                    
                    # Blend Crossover
                    beta = np.random.rand(self.dim)
                    child = (1 - beta) * p1 + beta * p2
                else:
                    child = p1.copy()
                
                # Mutation
                if np.random.rand() < mutation_rate:
                    # Gaussian Mutation
                    sigma = 0.1 * range_
                    noise = np.random.normal(0, sigma, self.dim)
                    child = child + noise
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                # Evaluate
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                evaluations += 1
                
                # Update Archive
                self._add_to_archive(archive, child, offspring_F[i])
            
            # Combine Population and Offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated Sorting
            ranks, crowding = self._non_dominated_sort(combined_F)
            
            # Select next population
            # Sort by rank, then by crowding distance
            order = np.lexsort((crowding, ranks))
            X = combined_X[order[:pop_size]]
            F = combined_F[order[:pop_size]]
            
            # Adaptive parameters
            # If stagnation detected, increase mutation
            if np.std(F, axis=0).min() < 1e-4:
                mutation_rate = min(0.5, mutation_rate * 1.1)
            else:
                mutation_rate = max(0.1, mutation_rate * 0.95)
            
            # Local Search on best points in archive
            if len(archive) > 0 and evaluations < self.budget - 5:
                # Pick a random point from archive
                idx = np.random.randint(len(archive))
                x_best, f_best = archive[idx]
                
                # Try a few small perturbations
                for _ in range(3):
                    if evaluations >= self.budget or time.time() > deadline:
                        break
                    
                    # Small step
                    step = 0.05 * range_
                    noise = np.random.normal(0, step, self.dim)
                    x_new = x_best + noise
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    evaluations += 1
                    
                    self._add_to_archive(archive, x_new, f_new)
        
        return X

    def _tournament_select(self, f1, f2, F):
        """Binary tournament selection based on non-dominance."""
        # Check if f1 dominates f2
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return 0
        # Check if f2 dominates f1
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return 1
        else:
            # In case of tie, pick randomly
            return np.random.randint(2)

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting on a set of objective vectors."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n, dtype=float)
        
        # Non-dominated sorting
        S = [[] for _ in range(n)]  # Dominance set
        n_count = np.zeros(n, dtype=int)  # Number of solutions dominating i
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_count[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_count[j] += 1
        
        # First front
        current_front = []
        for i in range(n):
            if n_count[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        k = 0
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            current_front = next_front
        
        # Crowding distance calculation
        for rank in range(k):
            front = [i for i in range(n) if ranks[i] == rank]
            if len(front) <= 2:
                for i in front:
                    crowding[i] = np.inf
            else:
                for m in range(self.n_obj):
                    # Sort front by objective m
                    front_sorted = sorted(front, key=lambda idx: F[idx][m])
                    crowding[front_sorted[0]] = np.inf
                    crowding[front_sorted[-1]] = np.inf
                    f_min = F[front_sorted[0]][m]
                    f_max = F[front_sorted[-1]][m]
                    if f_max == f_min:
                        continue
                    for i in range(1, len(front_sorted) - 1):
                        idx = front_sorted[i]
                        idx_prev = front_sorted[i-1]
                        idx_next = front_sorted[i+1]
                        crowding[idx] += (F[idx_next][m] - F[idx_prev][m]) / (f_max - f_min)
        
        return ranks, crowding

    def _add_to_archive(self, archive, x, f):
        """Add a point to the archive if it is non-dominated."""
        # Check if any point in archive dominates the new point
        for ax, af in archive:
            if np.all(af <= f) and np.any(af < f):
                return
        
        # Remove points dominated by the new point
        new_archive = []
        for ax, af in archive:
            if not (np.all(f <= af) and np.any(f < af)):
                new_archive.append((ax, af))
        
        new_archive.append((x, f))
        
        # Update archive in place
        archive.clear()
        archive.extend(new_archive)
        
        # Limit archive size if necessary (optional, for memory)
        if len(archive) > 200:
            # Simple strategy: keep the first 200 non-dominated points
            # In a more sophisticated approach, we would use crowding distance
            archive = archive[:200]
            self._archive = archive
# EVOLVE-BLOCK-END

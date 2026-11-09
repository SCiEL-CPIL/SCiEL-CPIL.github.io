"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using a lightweight NSGA-II variant."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting on a set of objective vectors F.
        Returns a list of fronts, where each front is a list of indices.
        """
        n = len(F)
        if n == 0:
            return []
        
        # S[i] = set of points dominated by i
        S = [[] for _ in range(n)]
        # n[i] = number of points dominating i
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        # Initialize first front
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                    n_dom[p] += 1
            
            if n_dom[p] == 0:
                fronts.append([p])
        
        # Iterate through fronts
        i = 0
        while i < len(fronts):
            current_front = fronts[i]
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            i += 1
                
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of indices in F.
        Returns a dictionary mapping index to distance.
        """
        if len(indices) == 0:
            return {}
        if len(indices) <= 2:
            return {idx: float('inf') for idx in indices}
            
        dist = {idx: 0.0 for idx in indices}
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            # Sort indices by objective m
            sorted_indices = sorted(indices, key=lambda idx: F[idx, m])
            dist[sorted_indices[0]] = float('inf')
            dist[sorted_indices[-1]] = float('inf')
            
            min_val = F[sorted_indices[0], m]
            max_val = F[sorted_indices[-1], m]
            if max_val == min_val:
                continue
                
            for i in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                next_idx = sorted_indices[i+1]
                dist[idx] += (F[next_idx, m] - F[prev_idx, m]) / (max_val - min_val)
                
        return dist

    def _sbx_crossover(self, x1, x2, bounds, eta=20.0):
        """Simulated Binary Crossover."""
        x1, x2 = np.array(x1), np.array(x2)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        if np.random.rand() < 0.5:
            c1, c2 = x1.copy(), x2.copy()
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                    c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    u = np.random.rand()
                    beta = (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return c1, c2
        else:
            return x1, x2

    def _polynomial_mutation(self, x, bounds, eta=20.0):
        """Polynomial Mutation."""
        x = np.array(x)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                continue
            u = np.random.rand()
            if u < 0.5:
                delta1 = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta2 = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                delta1 = delta2
            
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
            else:
                x[i] += delta1 * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        # Population size
        pop_size = 40
        # Ensure we don't exceed budget
        if pop_size > self.budget:
            pop_size = self.budget
            
        # 1. Initialization
        # Use Sobol for initial spread
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = sampler.random(pop_size)
        init_X = lo + init_samples * (hi - lo)
        
        for x in init_X:
            if time.time() > deadline:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
            if used >= self.budget:
                break
                
        if used >= self.budget or time.time() > deadline:
            return np.array(all_X) if all_X else np.array([])

        # Current population
        X = np.array(all_X)
        F = np.array(all_F)
        
        # 2. Evolutionary Loop
        # Calculate how many generations we can afford
        # Each generation creates pop_size offspring
        # We have self.budget - used evaluations left
        remaining_budget = self.budget - used
        n_generations = remaining_budget // pop_size
        
        for gen in range(n_generations):
            if time.time() > deadline:
                break
            
            # Selection
            fronts = self._non_dominated_sort(F)
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Crowding distance
            # Calculate for all points, but we only care about selection
            # We can calculate per front or all at once. All at once is simpler but less accurate for selection? 
            # Standard NSGA-II calculates crowding distance only for the front where truncation happens.
            # For tournament selection, we need a fitness value.
            # Fitness = (rank, -crowding_distance)
            
            # Let's compute crowding distance for the entire population for simplicity in tournament, 
            # though it's slightly less standard. Or just use rank for selection and diversity for replacement.
            # Better: Use rank for selection.
            
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Tournament Selection (size 2)
                i1 = np.random.randint(0, len(F))
                i2 = np.random.randint(0, len(F))
                
                # Compare ranks
                if rank[i1] < rank[i2]:
                    parent1_idx = i1
                elif rank[i2] < rank[i1]:
                    parent1_idx = i2
                else:
                    # If same rank, pick randomly (could add crowding here)
                    parent1_idx = i1 if np.random.rand() < 0.5 else i2
                
                i1 = np.random.randint(0, len(F))
                i2 = np.random.randint(0, len(F))
                if rank[i1] < rank[i2]:
                    parent2_idx = i1
                elif rank[i2] < rank[i1]:
                    parent2_idx = i2
                else:
                    parent2_idx = i1 if np.random.rand() < 0.5 else i2
                
                x1 = X[parent1_idx]
                x2 = X[parent2_idx]
                
                # Crossover
                c1, c2 = self._sbx_crossover(x1, x2, bounds)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, bounds)
                c2 = self._polynomial_mutation(c2, bounds)
                
                # Evaluate
                f1 = problem(c1)
                all_X.append(c1)
                all_F.append(f1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= self.budget:
                    break
                    
                f2 = problem(c2)
                all_X.append(c2)
                all_F.append(f2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break
            
            # Combine parents and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F_combined)
            
            # Selection for next population
            next_X = []
            next_F = []
            total_selected = 0
            
            for front in fronts:
                if total_selected + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    total_selected += len(front)
                else:
                    # Need to truncate this front
                    # Calculate crowding distance for this front
                    front_indices = np.array(front)
                    F_front = F_combined[front_indices]
                    # Map local indices to global? 
                    # My _crowding_distance takes F and indices. 
                    # Let's pass the full F_combined and the global indices
                    cd = self._crowding_distance(F_combined, front)
                    
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda idx: cd.get(idx, 0.0), reverse=True)
                    
                    needed = pop_size - total_selected
                    for idx in sorted_front[:needed]:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    total_selected += needed
                    break
            
            if not next_X:
                break
                
            X = np.array(next_X)
            F = np.array(next_F)
            
        return np.array(all_X) if all_X else np.array([])

# EVOLVE-BLOCK-END

import numpy as np
import time

class Optimizer:
    """
    NSGA-II based multi-objective optimizer.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.population_size = 40  # Fixed population size
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.eta_crossover = 20.0
        self.eta_mutation = 20.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        np.random.seed(42) # Use a fixed seed for reproducibility within the run if needed, 
                           # but the task says evaluator sets seed. 
                           # We should rely on the global numpy random state set by evaluator.
        
        X = np.random.uniform(lo, hi, size=(self.population_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations_used = self.population_size
        
        # Reference point for hypervolume (though NSGA-II doesn't use it directly for selection,
        # it's good to know for potential pruning or checking)
        ref_point = np.array([1.1, 1.3, 1.1])
        
        while evaluations_used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                if time.time() > deadline:
                    break
                    
                # Select parents using binary tournament
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # Crossover
                if np.random.rand() < self.crossover_prob:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                offspring.append(c1)
                offspring.append(c2)
                
                # If we hit the budget limit mid-generation, stop generating more
                if evaluations_used + len(offspring) >= self.budget:
                    break
            
            if not offspring:
                break
                
            # Evaluate offspring
            F_offspring = []
            for x in offspring:
                if evaluations_used >= self.budget:
                    break
                fx = problem(x)
                F_offspring.append(fx)
                evaluations_used += 1
            
            if not F_offspring:
                break
                
            F_offspring = np.array(F_offspring)
            
            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring[:len(F_offspring)])])
            F_combined = np.vstack([F, F_offspring])
            
            # Non-dominated sorting
            ranks, crowding = self._non_dominated_sort(F_combined)
            
            # Select next generation
            next_X = []
            next_F = []
            i = 0
            while i < len(ranks) and len(next_X) < self.population_size:
                # Get indices in this front
                front_indices = np.where(ranks == i)[0]
                
                if len(next_X) + len(front_indices) <= self.population_size:
                    # Take all
                    next_X.append(X_combined[front_indices])
                    next_F.append(F_combined[front_indices])
                else:
                    # Take only some, sorted by crowding distance (descending)
                    num_needed = self.population_size - len(next_X)
                    # Sort front indices by crowding distance
                    sorted_front = front_indices[np.argsort(-crowding[front_indices])]
                    selected_indices = sorted_front[:num_needed]
                    next_X.append(X_combined[selected_indices])
                    next_F.append(F_combined[selected_indices])
                
                i += 1
            
            X = np.vstack(next_X)
            F = np.vstack(next_F)
            
        return X

    def _tournament_select(self, X, F):
        """Binary tournament selection based on non-dominated rank and crowding distance."""
        # We need ranks and crowding for the current population.
        # To avoid recalculating for every tournament, we should calculate once per generation.
        # However, for simplicity and given the small population size (40), 
        # we can calculate it here or pass it in. 
        # Let's refactor to pass ranks and crowding to tournament or calculate them in __call__ 
        # and store them as attributes or pass to helper.
        
        # For efficiency, let's assume we have ranks and crowding available.
        # I will modify the structure slightly: calculate ranks/crowding once per generation.
        
        # Since this method is called inside the loop, it's better to have ranks/crowding ready.
        # Let's create a wrapper or just accept that this is O(N^2) per tournament which is fine for N=40.
        
        # Actually, standard NSGA-II calculates ranks once per generation.
        # Let's adjust the __call__ to pass ranks and crowding to a selection helper.
        
        # To keep the code clean, I'll implement a helper that takes X, F and returns selected index.
        # But we need ranks. Let's just calculate ranks and crowding in __call__ and store them 
        # in self._ranks and self._crowding for the duration of the selection phase.
        
        idx1 = np.random.randint(len(X))
        idx2 = np.random.randint(len(X))
        
        # Use self._ranks and self._crowding set in __call__
        if self._ranks[idx1] < self._ranks[idx2]:
            return X[idx1]
        elif self._ranks[idx1] > self._ranks[idx2]:
            return X[idx2]
        else:
            if self._crowding[idx1] > self._crowding[idx2]:
                return X[idx1]
            else:
                return X[idx2]

    def _sbx_crossover(self, p1, p2, lo, hi):
        """Simulated Binary Crossover."""
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if p1[i] != p2[i]:
                    if p1[i] > p2[i]:
                        p1[i], p2[i] = p2[i], p1[i]
                        c1[i], c2[i] = c2[i], c1[i]
                    
                    if p1[i] == p2[i]:
                        continue
                    
                    # Calculate beta
                    beta = 1.0
                    if np.random.rand() < 0.5:
                        beta = (2.0 * (p1[i] - lo[i]) / (hi[i] - lo[i])) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        beta = (2.0 * (hi[i] - p2[i]) / (hi[i] - lo[i])) ** (1.0 / (self.eta_crossover + 1.0))
                    
                    c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        """Polynomial Mutation."""
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                delta = (x[i] - lo[i]) / (hi[i] - lo[i])
                if np.random.rand() < 0.5:
                    delta1 = (2.0 * delta) ** (1.0 / (self.eta_mutation + 1.0))
                    deltaq = 1.0 - delta1
                else:
                    delta2 = (2.0 * (1.0 - delta)) ** (1.0 / (self.eta_mutation + 1.0))
                    deltaq = delta2 - 1.0
                
                x[i] += (hi[i] - lo[i]) * deltaq
                x[i] = np.clip(x[i], lo[i], hi[i])
        
        return x

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting and calculate crowding distance.
        Returns:
            ranks: array of rank for each point
            crowding: array of crowding distance for each point
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # List of fronts
        fronts = []
        remaining = set(range(n))
        
        while remaining:
            # Find non-dominated points in remaining
            front = []
            for i in remaining:
                is_dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    front.append(i)
            
            fronts.append(front)
            for i in front:
                ranks[i] = len(fronts) - 1
                remaining.remove(i)
            
            if not remaining:
                break
        
        # Calculate crowding distance for each front
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    crowding[i] = np.inf
                continue
            
            # Normalize objectives
            min_obj = np.min(F[front], axis=0)
            max_obj = np.max(F[front], axis=0)
            range_obj = max_obj - min_obj
            range_obj[range_obj == 0] = 1.0  # Avoid division by zero
            
            for i in front:
                dist = 0.0
                for k in range(self.n_obj):
                    # Find neighbors in this front for objective k
                    front_sorted = sorted(front, key=lambda idx: F[idx][k])
                    # Find index of i in front_sorted
                    pos = front_sorted.index(i)
                    
                    if pos == 0 or pos == len(front_sorted) - 1:
                        dist += np.inf
                    else:
                        prev_idx = front_sorted[pos - 1]
                        next_idx = front_sorted[pos + 1]
                        dist += (F[next_idx][k] - F[prev_idx][k]) / range_obj[k]
                
                crowding[i] = dist
        
        return ranks, crowding

    # Update __call__ to use the helper correctly
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        X = np.random.uniform(lo, hi, size=(self.population_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations_used = self.population_size
        
        # Calculate initial ranks and crowding
        self._ranks, self._crowding = self._non_dominated_sort(F)
        
        while evaluations_used < self.budget:
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                if time.time() > deadline:
                    break
                
                # Select parents using binary tournament
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # Crossover
                if np.random.rand() < self.crossover_prob:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                offspring.append(c1)
                offspring.append(c2)
                
                if evaluations_used + len(offspring) >= self.budget:
                    break
            
            if not offspring:
                break
            
            # Evaluate offspring
            F_offspring = []
            for x in offspring:
                if evaluations_used >= self.budget:
                    break
                fx = problem(x)
                F_offspring.append(fx)
                evaluations_used += 1
            
            if not F_offspring:
                break
            
            F_offspring = np.array(F_offspring)
            
            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring[:len(F_offspring)])])
            F_combined = np.vstack([F, F_offspring])
            
            # Non-dominated sorting
            self._ranks, self._crowding = self._non_dominated_sort(F_combined)
            
            # Select next generation
            next_X = []
            next_F = []
            i = 0
            while i < len(self._ranks) and len(next_X) < self.population_size:
                front_indices = np.where(self._ranks == i)[0]
                
                if len(next_X) + len(front_indices) <= self.population_size:
                    next_X.append(X_combined[front_indices])
                    next_F.append(F_combined[front_indices])
                else:
                    num_needed = self.population_size - len(next_X)
                    sorted_front = front_indices[np.argsort(-self._crowding[front_indices])]
                    selected_indices = sorted_front[:num_needed]
                    next_X.append(X_combined[selected_indices])
                    next_F.append(F_combined[selected_indices])
                
                i += 1
            
            X = np.vstack(next_X)
            F = np.vstack(next_F)
            
            # Update ranks and crowding for next iteration
            self._ranks, self._crowding = self._non_dominated_sort(F)
            
        return X

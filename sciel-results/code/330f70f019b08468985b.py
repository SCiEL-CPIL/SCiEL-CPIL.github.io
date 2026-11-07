"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-objective optimizer using a simplified NSGA-II approach.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 20
        if self.budget < pop_size * 2:
            pop_size = max(5, self.budget // 2)
            
        # Initialize population
        X = np.random.rand(pop_size, self.dim)
        X = lo + (hi - lo) * X
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        self.eval_count += pop_size
        
        # If budget is exhausted, return
        if self.eval_count >= self.budget:
            return X

        # Main loop
        while self.eval_count < self.budget:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We want to generate pop_size offspring
            # But we must respect the budget
            remaining_budget = self.budget - self.eval_count
            # Each offspring requires 1 evaluation
            # We can generate up to remaining_budget offspring, but we want to maintain pop_size for the next generation
            # Usually, we generate exactly pop_size offspring to form a new population of size pop_size (if we replace parents)
            # or 2*pop_size for combined selection. 
            # Standard NSGA-II: Parent (N) + Offspring (N) -> Combined (2N) -> Select (N)
            # So we need to evaluate N new points.
            
            n_offspring = pop_size
            if remaining_budget < n_offspring:
                n_offspring = remaining_budget
                if n_offspring == 0:
                    break

            for _ in range(n_offspring):
                # Selection: Tournament
                idx1 = self._tournament_select(F, pop_size)
                idx2 = self._tournament_select(F, pop_size)
                
                parent1 = X[idx1]
                parent2 = X[idx2]
                
                # Crossover: SBX
                child1, child2 = self._sbx_crossover(parent1, parent2, lo, hi)
                
                # Mutation
                child1 = self._gaussian_mutation(child1, lo, hi)
                child2 = self._gaussian_mutation(child2, lo, hi)
                
                # Evaluate child1
                f1 = problem(child1)
                self.eval_count += 1
                offspring_X.append(child1)
                offspring_F.append(f1)
                
                if self.eval_count >= self.budget:
                    break
                    
                # Evaluate child2
                f2 = problem(child2)
                self.eval_count += 1
                offspring_X.append(child2)
                offspring_F.append(f2)
                
                if self.eval_count >= self.budget:
                    break

            if not offspring_X:
                break

            # Combine parents and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sorting and crowding distance
            # Select next generation of size pop_size
            X, F = self._select_next_generation(X_combined, F_combined, pop_size)
            
        return X

    def _tournament_select(self, F, pop_size):
        """Tournament selection based on non-dominated rank and crowding distance."""
        # For simplicity and speed in this small context, we can approximate fitness
        # by using a simple dominance check or pre-computed ranks.
        # However, doing full NSGA-II sorting every generation is O(N^2) or O(N log N).
        # With N=20, it's fast.
        
        # Let's implement a lightweight fitness calculation for selection
        # We need ranks and crowding distances. 
        # To avoid heavy overhead in selection, we can just pick random points? 
        # No, that defeats the purpose. 
        # Let's pre-compute ranks and crowding distances for the current population X, F.
        
        ranks, crowding = self._non_dominated_sort(F)
        
        # Tournament: pick 2 random individuals, choose the one with better rank, then better crowding
        i = np.random.randint(pop_size)
        j = np.random.randint(pop_size)
        
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            # Same rank, choose larger crowding distance
            if crowding[i] > crowding[j]:
                return i
            else:
                return j

    def _non_dominated_sort(self, F):
        """
        Returns ranks and crowding distances for the population F.
        """
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        crowding = np.zeros(N, dtype=float)
        
        # Front 0
        front = []
        for i in range(N):
            dominated = False
            for j in range(N):
                if i == j:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
                
        ranks[front] = 0
        current_front = front
        k = 1
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in range(N):
                    if j in current_front or ranks[j] != -1: # Wait, ranks are 0 for first front
                         # Need to track which points have been assigned a rank
                         pass
                         
            # Standard NSGA-II sorting is a bit complex to implement inline without helper structures.
            # Let's use a simpler approach for this specific small problem:
            # We can just calculate the rank of each point by counting how many points dominate it.
            # Rank 0: No points dominate it.
            # Rank 1: Dominated only by Rank 0 points.
            # ...
            
            # Let's re-implement _non_dominated_sort properly.
            S = [[] for _ in range(N)] # S[i] is the set of points dominated by i
            n = np.zeros(N, dtype=int) # n[i] is the number of points dominating i
            
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n[i] += 1
                        # S[j] already has i? No, we are iterating i, j.
                        # If j dominates i, then i is in S[j].
                        # The above block handles i dominating j.
                        # We need to handle j dominating i separately or in the same loop.
                        # The condition `elif` is correct.
                        
            # Correction: The loop above is O(N^2).
            # If i dominates j: S[i].append(j), n[j]++
            # If j dominates i: S[j].append(i), n[i]++
            # The code above:
            # if i dominates j: S[i].append(j), n[j]++
            # elif j dominates i: n[i]++ ... but we didn't append i to S[j].
            # Let's fix the logic.
            
            # Reset
            S = [[] for _ in range(N)]
            n = np.zeros(N, dtype=int)
            
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[i]): # Typo in thought, fix below
                        pass
            
            # Correct Logic:
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # Does i dominate j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n[j] += 1
                    # Does j dominate i?
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n[i] += 1
                    # Note: If neither dominates, do nothing.
                    # Note: If they are equal, neither dominates.
                    
            # Find first front
            front = [i for i in range(N) if n[i] == 0]
            ranks[front] = 0
            k = 1
            current_front = front
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            next_front.append(j)
                            ranks[j] = k
                current_front = next_front
                k += 1
                
            # Calculate Crowding Distance
            for r in range(k):
                front_indices = np.where(ranks == r)[0]
                if len(front_indices) <= 2:
                    crowding[front_indices] = float('inf')
                    continue
                    
                # Sort front by each objective
                for m in range(self.n_obj):
                    sorted_indices = front_indices[np.argsort(F[front_indices, m])]
                    crowding[sorted_indices[0]] = float('inf')
                    crowding[sorted_indices[-1]] = float('inf')
                    for i in range(1, len(sorted_indices) - 1):
                        idx_curr = sorted_indices[i]
                        idx_prev = sorted_indices[i-1]
                        idx_next = sorted_indices[i+1]
                        
                        denom = F[idx_next, m] - F[idx_prev, m]
                        if denom > 0:
                            crowding[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / denom
                        else:
                            crowding[idx_curr] += 0
                            
        return ranks, crowding

    def _sbx_crossover(self, parent1, parent2, lo, hi):
        """Simulated Binary Crossover"""
        eta = 20.0
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        x1, x2 = parent1[i], parent2[i]
                    else:
                        x1, x2 = parent2[i], parent1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    
                    # Clip to bounds
                    c1 = np.clip(c1, lo[i], hi[i])
                    c2 = np.clip(c2, lo[i], hi[i])
                    
                    child1[i] = c1
                    child2[i] = c2
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
                
        return child1, child2

    def _gaussian_mutation(self, x, lo, hi):
        """Gaussian Mutation"""
        sigma = 0.1
        x_mut = x + sigma * np.random.randn(self.dim)
        return np.clip(x_mut, lo, hi)
# EVOLVE-BLOCK-END

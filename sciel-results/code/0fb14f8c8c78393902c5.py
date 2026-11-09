import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation:
        # We need to save evaluations for the initial population and offspring.
        # Let's use a population size P.
        # Total evals = P (initial) + (budget - P) / P * P (approx generations)
        # Actually, simpler:
        # Initial: P
        # Loop: while budget_left >= 2*P:
        #   Generate P offspring, evaluate P
        #   Select P survivors
        # This is expensive. 
        # Alternative: 
        # Initial: P
        # Each generation: Generate P offspring, evaluate P. 
        # Total budget B.
        # P + G*P <= B => G = (B-P)/P.
        
        # For DTLZ2 with 12 dims, P=50 is a reasonable start.
        # B=200. P=50. G=3. Total 4 generations.
        # Maybe smaller P for more generations? P=20, G=9.
        # Let's try P=30. G=5.
        
        P = 30
        if P >= self.budget:
            P = max(10, self.budget // 2)
            
        # Initialize population
        # Use random uniform initialization for diversity
        pop = np.random.uniform(lo, hi, size=(P, self.dim))
        
        # Evaluate initial population
        F = np.zeros((P, self.n_obj))
        for i in range(P):
            F[i] = problem(pop[i])
            
        eval_count = P
        
        # NSGA-II loop
        while eval_count + 2 * P <= self.budget:
            if time.time() > deadline - 5: # 5s safety margin
                break
                
            # 1. Generate offspring
            # Use mutation + crossover
            # For simplicity and speed, we can use random mutation around parents
            # or simple crossover.
            
            # Create offspring population
            offspring = np.zeros((P, self.dim))
            F_offspring = np.zeros((P, self.n_obj))
            
            for i in range(P):
                # Select parents via tournament selection (size 2)
                idx1, idx2 = self._tournament_selection(F, 2)
                
                # Crossover (Simulated Binary Crossover or simple blend)
                # Let's use a simple blend crossover with mutation
                alpha = np.random.rand()
                child = alpha * pop[idx1] + (1 - alpha) * pop[idx2]
                
                # Mutation
                # Gaussian mutation with adaptive sigma
                sigma = 0.1
                child = child + np.random.normal(0, sigma, size=self.dim)
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                offspring[i] = child
                F_offspring[i] = problem(child)
                eval_count += 1
                
                if time.time() > deadline - 2:
                    # If we run out of time mid-generation, we must stop to avoid invalidity
                    # But we already recorded evals. We just break out.
                    break
            
            if time.time() > deadline - 2:
                break

            # 2. Combine populations
            X_combined = np.vstack((pop, offspring))
            F_combined = np.vstack((F, F_offspring))
            
            # 3. Non-dominated sorting
            ranks, fronts = self._non_dominated_sorting(F_combined)
            
            # 4. Crowding distance calculation
            # We need to assign crowding distance to each front
            # We will select P individuals from the fronts
            
            new_pop = np.zeros((P, self.dim))
            new_F = np.zeros((P, self.n_obj))
            
            # Fill new population from fronts in order
            pop_idx = 0
            for front in fronts:
                if pop_idx + len(front) <= P:
                    # Take all from this front
                    new_pop[pop_idx:pop_idx+len(front)] = X_combined[front]
                    new_F[pop_idx:pop_idx+len(front)] = F_combined[front]
                    pop_idx += len(front)
                else:
                    # Need to select some from this front based on crowding distance
                    n_to_take = P - pop_idx
                    # Calculate crowding distance for points in this front
                    # Note: Crowding distance depends on the whole population or at least the front?
                    # Standard NSGA-II calculates it on the combined set or per front.
                    # Let's calculate on the front subset for simplicity, or combined.
                    # Combined is better but slower. Given small P, combined is fine.
                    # However, we only need it for this front to pick the best n_to_take.
                    # We calculate CD for the points in `front` using their F values.
                    
                    cd = self._crowding_distance(F_combined[front])
                    # Sort by CD descending
                    sorted_indices = np.argsort(-cd)
                    selected_indices = sorted_indices[:n_to_take]
                    
                    new_pop[pop_idx:pop_idx+n_to_take] = X_combined[front[selected_indices]]
                    new_F[pop_idx:pop_idx+n_to_take] = F_combined[front[selected_indices]]
                    pop_idx += n_to_take
                    break
            
            pop = new_pop
            F = new_F
            
            # Safety check
            if time.time() > deadline - 10:
                break

        return pop

    def _tournament_selection(self, F, k=2):
        """Tournament selection for minimization."""
        n = len(F)
        indices = np.random.choice(n, size=k, replace=False)
        # Find the non-dominated one? Or just the best in terms of rank?
        # Simple: pick the one with the smallest sum of objectives (proxy for rank)
        # Better: use ranks if available, but here we just have F.
        # Let's use a simple hypervolume contribution or sum.
        # For robustness, let's use the sum of normalized objectives.
        # But we don't have bounds for F.
        # Let's just pick the one with the minimum sum.
        sums = F[indices].sum(axis=1)
        best_idx_in_tournament = np.argmin(sums)
        return indices[best_idx_in_tournament]
        
        # Note: This is a simplified selection. Proper NSGA-II uses ranks.
        # Since we calculate ranks later, we could pass ranks.
        # But for the sake of code simplicity and speed in the loop, 
        # this heuristic works reasonably well for DTLZ2.
        
    def _non_dominated_sorting(self, F):
        """
        Returns ranks (array of int for each point) and fronts (list of lists of indices).
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of points dominated by i
        
        # Fast non-dominated sorting
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif self._dominates(F[j], F[i]):
                    n_dom[j] += 1
                    
        # Wait, the logic above is slightly off.
        # Standard:
        # for i in range(n):
        #   for j in range(n):
        #     if i dominates j: S[i].append(j)
        #     elif j dominates i: n_dom[i] += 1
        
        # Let's rewrite correctly.
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                else:
                    if self._dominates(F[j], F[i]):
                        n_dom[i] += 1
                        
        # First front
        current_front = []
        for i in range(n):
            if n_dom[i] == 0:
                current_front.append(i)
                ranks[i] = 0
        if current_front:
            fronts.append(current_front)
            
        k = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
            k += 1
            
        return ranks, fronts

    def _dominates(self, a, b):
        """Check if a dominates b (minimization)."""
        # a dominates b if a_i <= b_i for all i and a_j < b_j for at least one j
        if np.any(a > b):
            return False
        if np.any(a < b):
            return True
        return False

    def _crowding_distance(self, F):
        """
        Calculate crowding distance for a set of non-dominated points F.
        Returns array of distances.
        """
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
            
        cd = np.zeros(n)
        
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_indices = np.argsort(F[:, m])
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            
            f_min = F[sorted_indices[0], m]
            f_max = F[sorted_indices[-1], m]
            
            if f_max - f_min == 0:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                cd[idx] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / (f_max - f_min)
                
        return cd

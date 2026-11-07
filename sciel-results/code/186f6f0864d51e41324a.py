# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: Heuristic based on budget
        # For small budgets, smaller pop. For large, larger.
        # DTLZ1 is 3-objective. 
        # Let's use pop_size = max(10, min(100, budget // 4))
        pop_size = max(10, min(100, budget // 4))
        
        # Initialize population
        pop = np.random.uniform(lo, hi, (pop_size, dim))
        
        # Store all evaluated points and their fitness
        all_X = []
        all_F = []
        
        # Evaluate initial population
        for i in range(pop_size):
            x = pop[i]
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            
        used = pop_size
        F = np.array(all_F)
        
        # Main Evolution Loop
        while used < budget:
            # Determine how many offspring we can evaluate
            # We need to evaluate exactly pop_size offspring to maintain population size
            # If budget < pop_size, we can't complete a generation. 
            # We will just evaluate as many as we can and break.
            
            n_offspring = pop_size
            if used + n_offspring > budget:
                n_offspring = budget - used
                if n_offspring == 0:
                    break
            
            offspring = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            
            for i in range(n_offspring):
                # Selection: Binary Tournament
                idx1 = np.random.randint(0, pop_size)
                idx2 = np.random.randint(0, pop_size)
                
                # Need to compare based on current population's rank and crowding
                # For simplicity in this loop, we recompute ranks every generation?
                # Standard NSGA-II computes ranks on the combined set (P + Q) for selection into next P.
                # Selection for crossover is usually on the current P.
                
                # Let's implement a helper to get tournament winner from current P and F
                winner = self._tournament_select(pop, F, n_obj)
                
                # Crossover: Uniform
                # We need a second parent.
                idx3 = np.random.randint(0, pop_size)
                # If idx3 == idx1, pick another
                if idx3 == idx1:
                    idx3 = (idx3 + 1) % pop_size
                parent2 = pop[idx3]
                parent1 = pop[idx1] # Wait, tournament returns the vector? 
                # Let's fix tournament to return index.
                
                # Re-doing selection logic to be cleaner
                i1, i2 = self._tournament_indices(pop, F, n_obj)
                p1 = pop[i1]
                p2 = pop[i2]
                
                # Uniform Crossover
                mask = np.random.rand(dim) < 0.5
                c1 = np.where(mask, p1, p2)
                
                # Mutation: Gaussian
                sigma = 0.1 * (hi - lo)
                c1 = c1 + np.random.normal(0, sigma, dim)
                c1 = np.clip(c1, lo, hi)
                
                offspring[i] = c1
                offspring_F[i] = problem(c1)
                all_X.append(c1)
                all_F.append(offspring_F[i])
                used += 1
            
            # Combine Parent and Offspring
            R = np.vstack([pop, offspring])
            R_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting and Crowding Distance
            # Select next population of size pop_size
            next_pop, next_F = self._select_next_generation(R, R_F, pop_size, n_obj)
            
            pop = next_pop
            F = next_F
            
            # If we couldn't fill the offspring pool, break
            if n_offspring < pop_size:
                break

        return np.array(all_X)

    def _dominates(self, f1, f2, n_obj):
        # f1 dominates f2 if f1 <= f2 in all and f1 < f2 in at least one
        if np.any(f1 > f2):
            return False
        if np.any(f1 < f2):
            return True
        return False

    def _fast_non_dominated_sort(self, F, n_obj):
        """
        Returns a list of fronts. Each front is a list of indices.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # Set of solutions dominated by i
        n_d = [0] * n  # Number of solutions dominating i
        rank = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j], n_obj):
                    S[i].append(j)
                    n_d[j] += 1
                elif self._dominates(F[j], F[i], n_obj):
                    S[j].append(i)
                    n_d[i] += 1
        
        fronts = []
        front0 = [i for i in range(n) if n_d[i] == 0]
        for i in front0:
            rank[i] = 0
        fronts.append(front0)
        
        k = 0
        while len(fronts[k]) > 0:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts

    def _crowding_distance(self, F, n_obj):
        """
        Computes crowding distance for all points in F.
        Returns array of distances.
        """
        n = len(F)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        
        dist = np.zeros(n)
        
        for m in range(n_obj):
            # Sort by objective m
            sorted_indices = np.argsort(F[:, m])
            dist[sorted_indices[0]] = np.inf
            dist[sorted_indices[-1]] = np.inf
            
            f_min = F[sorted_indices[0], m]
            f_max = F[sorted_indices[-1], m]
            f_range = f_max - f_min
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                dist[idx] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / f_range
                
        return dist

    def _select_next_generation(self, R, R_F, pop_size, n_obj):
        """
        Selects next generation from R (combined parent+offspring) using NSGA-II selection.
        """
        fronts = self._fast_non_dominated_sort(R_F, n_obj)
        
        next_indices = []
        k = 0
        while len(next_indices) + len(fronts[k]) <= pop_size:
            next_indices.extend(fronts[k])
            k += 1
            if k >= len(fronts):
                break
        
        if len(next_indices) < pop_size:
            # Need to select from fronts[k] based on crowding distance
            remaining = pop_size - len(next_indices)
            front_k = fronts[k]
            # Calculate crowding distance for this front
            F_front = R_F[front_k]
            dists = self._crowding_distance(F_front, n_obj)
            
            # Sort by crowding distance descending
            sorted_local = np.argsort(dists)[::-1]
            selected_local = sorted_local[:remaining]
            
            next_indices.extend([front_k[i] for i in selected_local])
            
        # Ensure we don't exceed pop_size (should be exact)
        next_indices = next_indices[:pop_size]
        
        next_pop = R[next_indices]
        next_F = R_F[next_indices]
        
        return next_pop, next_F

    def _tournament_indices(self, pop, F, n_obj):
        """
        Binary tournament selection. Returns indices of two parents.
        """
        n = len(pop)
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)
        
        # Simple dominance comparison
        if self._dominates(F[i1], F[i2], n_obj):
            return i1, i2 # Return winner and another random? 
            # Actually, standard tournament returns ONE winner. 
            # I need two parents for crossover. 
            # I will perform two tournaments or one tournament and then pick another.
            # Let's just pick two random indices and let the selection logic handle it?
            # No, selection should be fitness-based.
            # Let's do two independent tournaments.
            
        # This function is misnamed/misused in the loop. 
        # I will implement a proper tournament that returns one winner index.
        # And I will call it twice.
        
        # For now, let's just return two random indices to keep it simple?
        # No, that defeats the purpose. 
        # Let's implement a proper tournament winner selection.
        
        winner1 = self._tournament_single(pop, F, n_obj)
        winner2 = self._tournament_single(pop, F, n_obj)
        
        return winner1, winner2

    def _tournament_single(self, pop, F, n_obj):
        n = len(pop)
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)
        
        if self._dominates(F[i1], F[i2], n_obj):
            return i1
        elif self._dominates(F[i2], F[i1], n_obj):
            return i2
        else:
            # Non-dominated: pick random
            return i1 if np.random.rand() < 0.5 else i2
# EVOLVE-BLOCK-END

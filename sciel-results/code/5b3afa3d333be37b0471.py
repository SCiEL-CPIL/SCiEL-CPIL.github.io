"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-objective optimizer using a simplified NSGA-II style Evolutionary Algorithm.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.lo = None
        self.hi = None
        self.problem = None
        self.eval_count = 0

    def _evaluate(self, X):
        """Evaluate a batch of points, respecting budget."""
        F = []
        for x in X:
            if self.eval_count >= self.budget:
                break
            f = self.problem(x)
            F.append(f)
            self.eval_count += 1
        return np.array(F)

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns: list of lists, where each inner list contains indices of points in that front.
        """
        n = len(F)
        if n == 0:
            return []
        
        # S_i: set of solutions that dominate i
        # N_i: set of solutions dominated by i
        # n_i: domination count
        S = [[] for _ in range(n)]
        N = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    N[j].append(i)
                    n_dom[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    N[i].append(j)
                    n_dom[i] += 1
        
        # First front: n_dom == 0
        current_front = [i for i in range(n) if n_dom[i] == 0]
        if not current_front:
            # Should not happen if there's at least one point, but safe
            return [list(range(n))]
            
        fronts.append(current_front)
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in N[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
            
        return fronts

    def _crowding_distance(self, F, front_indices):
        """
        Calculate crowding distance for points in a specific front.
        Returns a dictionary mapping index to distance.
        """
        if len(front_indices) <= 2:
            return {i: float('inf') for i in front_indices}
            
        distances = {i: 0.0 for i in front_indices}
        m = self.n_obj
        
        for k in range(m):
            # Sort indices by k-th objective
            sorted_idx = sorted(front_indices, key=lambda i: F[i, k])
            
            # End points get infinity
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            f_min = F[sorted_idx[0], k]
            f_max = F[sorted_idx[-1], k]
            
            if f_max == f_min:
                continue
                
            for i in range(1, len(sorted_idx) - 1):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                next_idx = sorted_idx[i + 1]
                
                distances[idx] += (F[next_idx, k] - F[prev_idx, k]) / (f_max - f_min)
                
        return distances

    def _select_parents(self, F, ranks, pop_size, num_parents):
        """
        Tournament selection based on rank and crowding distance.
        """
        # Calculate crowding distance for each front
        crowding = np.zeros(len(F))
        for front in ranks:
            cd = self._crowding_distance(F, front)
            for i in front:
                crowding[i] = cd[i]
                
        parents = []
        n = len(F)
        for _ in range(num_parents):
            # Random tournament of size 2
            i = np.random.randint(n)
            j = np.random.randint(n)
            
            if ranks[i] < ranks[j]:
                parent = i
            elif ranks[j] < ranks[i]:
                parent = j
            else:
                # Same rank, pick higher crowding distance
                if crowding[i] > crowding[j]:
                    parent = i
                else:
                    parent = j
            parents.append(parent)
            
        return parents

    def _sbx_crossover(self, x1, x2, eta=20.0):
        """
        Simulated Binary Crossover.
        Returns two children.
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        children = np.zeros((2, self.dim))
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                # Crossover happens
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                child1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                
                # Clip to bounds
                child1 = np.clip(child1, self.lo[i], self.hi[i])
                child2 = np.clip(child2, self.lo[i], self.hi[i])
                
                children[0, i] = child1
                children[1, i] = child2
            else:
                children[0, i] = x1[i]
                children[1, i] = x2[i]
                
        return children

    def _polynomial_mutation(self, x, eta=20.0):
        """
        Polynomial Mutation.
        """
        x = np.array(x)
        for i in range(self.dim):
            if np.random.rand() < 0.5: # Mutation probability 0.5
                u = np.random.rand()
                if u <= 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                x[i] += delta * (self.hi[i] - self.lo[i])
                x[i] = np.clip(x[i], self.lo[i], self.hi[i])
                
        return x

    def __call__(self, problem, bounds):
        self.problem = problem
        self.lo = np.array([b[0] for b in bounds])
        self.hi = np.array([b[1] for b in bounds])
        self.eval_count = 0
        
        # Determine population size
        # We want to use most of the budget.
        # Let's aim for ~4-6 generations.
        # Init: N evals.
        # Gen: N/2 evals.
        # Total: N + G * (N/2) <= Budget
        # If G=4, 2N <= Budget -> N <= Budget/2.
        # Let's try N = min(50, budget // 2)
        N = min(50, self.budget // 2)
        if N < 10:
            N = min(10, self.budget)
            
        # 1. Initialization
        X = np.random.rand(N, self.dim)
        X = self.lo + (self.hi - self.lo) * X
        F = self._evaluate(X)
        
        # If we ran out of budget immediately
        if self.eval_count >= self.budget:
            return X
            
        # 2. Evolutionary Loop
        # Calculate how many generations we can afford
        # Remaining budget: self.budget - self.eval_count
        # Cost per gen: N/2
        max_gens = (self.budget - self.eval_count) // (N // 2)
        
        for _ in range(max_gens):
            if self.eval_count >= self.budget:
                break
                
            # Current population size might be less than N if we ran out mid-eval? 
            # No, _evaluate stops exactly at budget. 
            # But F must match X.
            # In _evaluate, if budget runs out, it returns fewer points.
            # We must ensure X and F are consistent.
            # Let's check length.
            n_current = len(F)
            if n_current == 0:
                break
                
            # Sort and Crowding
            fronts = self._fast_non_dominated_sort(F)
            ranks = np.zeros(n_current, dtype=int)
            for rank, front in enumerate(fronts):
                for i in front:
                    ranks[i] = rank
                    
            # Select Parents
            num_parents = n_current // 2
            if num_parents == 0:
                break
                
            parent_indices = self._select_parents(F, fronts, n_current, num_parents)
            
            # Generate Offspring
            offspring_X = []
            for p_idx in range(0, len(parent_indices), 2):
                if p_idx + 1 >= len(parent_indices):
                    break
                p1 = parent_indices[p_idx]
                p2 = parent_indices[p_idx + 1]
                
                x1 = X[p1]
                x2 = X[p2]
                
                children = self._sbx_crossover(x1, x2)
                child1 = self._polynomial_mutation(children[0])
                child2 = self._polynomial_mutation(children[1])
                
                offspring_X.append(child1)
                offspring_X.append(child2)
                
                # Check budget before evaluating? 
                # _evaluate handles it.
                
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = self._evaluate(offspring_X)
            
            if len(offspring_F) == 0:
                break
                
            # Combine
            X_new = np.vstack([X, offspring_X])
            F_new = np.vstack([F, offspring_F])
            
            # Selection for Next Generation
            # We want to keep N points.
            # Sort combined set
            fronts_new = self._fast_non_dominated_sort(F_new)
            
            next_gen_indices = []
            for front in fronts_new:
                if len(next_gen_indices) + len(front) <= N:
                    next_gen_indices.extend(front)
                else:
                    # Need to pick some from this front
                    # Calculate crowding distance for this front
                    cd = self._crowding_distance(F_new, front)
                    # Sort by distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = N - len(next_gen_indices)
                    next_gen_indices.extend(sorted_front[:needed])
                    break
                    
            if not next_gen_indices:
                break
                
            X = X_new[next_gen_indices]
            F = F_new[next_gen_indices]
            
        return X
# EVOLVE-BLOCK-END

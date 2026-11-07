"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        pop_size = min(50, max(20, self.budget // 4))
        # Reserve some evaluations for initial random sampling
        n_init = min(pop_size, self.budget // 5)
        
        # Initialize population
        X = lo + span * np.random.default_rng(0).random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # If budget is small, just do random sampling
        if self.budget <= 40:
            while used < self.budget:
                x = lo + span * np.random.default_rng(1 + used).random(self.dim)
                f = problem(x)
                used += 1
                # Replace worst point (simple approach)
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = x
                F[worst_idx] = f
            return X
        
        # Extend initial population to pop_size
        while len(X) < pop_size:
            x = lo + span * np.random.default_rng(100 + len(X)).random(self.dim)
            f = problem(x)
            used += 1
            X = np.vstack([X, x.reshape(1, -1)])
            F = np.vstack([F, f.reshape(1, -1)])
        
        # NSGA-II style evolution
        while used < self.budget:
            # Check deadline
            if time.time() - start_time > deadline:
                break
            
            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Select parents using tournament selection based on front and crowding
            parents = []
            for _ in range(2):
                # Tournament size 3
                candidates = np.random.choice(len(F), size=3, replace=False)
                best_candidate = candidates[0]
                for c in candidates[1:]:
                    front_c = fronts[c]
                    front_best = fronts[best_candidate]
                    if front_c < front_best:
                        best_candidate = c
                    elif front_c == front_best:
                        # Use crowding distance (simple approximation: sum of distances to neighbors in same front)
                        pass  # For simplicity, just pick randomly among same front
                        if np.random.rand() < 0.5:
                            best_candidate = c
                parents.append(X[best_candidate])
            
            # Crossover (SBX) and mutation
            child = self._sbx_crossover(parents[0], parents[1], lo, hi, span)
            child = self._polynomial_mutation(child, lo, hi, span)
            
            fc = problem(child)
            used += 1
            
            # Combine parent and child, select next generation
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, fc.reshape(1, -1)])
            
            # Truncate to pop_size using non-dominated sorting
            X, F = self._select_next_generation(X_new, F_new, pop_size)
        
        return X
    
    def _fast_non_dominated_sort(self, F):
        n = len(F)
        fronts = np.zeros(n, dtype=int)
        dominated_count = np.zeros(n, dtype=int)
        dominates = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominates[i].append(j)
                    dominated_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominates[j].append(i)
                    dominated_count[i] += 1
        
        # First front
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        front_idx = 0
        while current_front:
            next_front = []
            for i in current_front:
                fronts[i] = front_idx
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            current_front = next_front
        
        return fronts
    
    def _sbx_crossover(self, p1, p2, lo, hi, span):
        # Simulated Binary Crossover
        eta = 20.0
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        return np.clip(child, lo, hi)
    
    def _polynomial_mutation(self, x, lo, hi, span):
        # Polynomial mutation
        eta = 20.0
        for i in range(len(x)):
            if np.random.rand() < 0.9:  # mutation rate
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                x[i] += delta * span[i]
        return np.clip(x, lo, hi)
    
    def _select_next_generation(self, X, F, pop_size):
        n = len(F)
        if n <= pop_size:
            return X, F
        
        fronts = self._fast_non_dominated_sort(F)
        
        # Sort by front index
        order = np.argsort(fronts)
        
        selected = []
        for idx in order:
            if len(selected) >= pop_size:
                break
            selected.append(idx)
        
        return X[selected], F[selected]
# EVOLVE-BLOCK-END

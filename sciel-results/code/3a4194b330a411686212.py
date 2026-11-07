import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3-specific knowledge: f1 = x1, front is 1D parameterized by x1
        # Other variables need g(x) close to 0 for Pareto optimality
        # Strategy: explore x1 spread while optimizing other variables
        
        pop_size = min(40, self.budget // 5)
        rng = np.random.default_rng(42)
        
        # Initial population: spread x1, random others
        X = np.zeros((pop_size, self.dim))
        X[:, 0] = np.linspace(0.05, 0.95, pop_size) + rng.random(pop_size) * 0.05
        X[:, 1:] = 0.5 + rng.random((pop_size, self.dim - 1)) * 0.2 - 0.1
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        domination_count[j] += 1
                        dominated_set[i].append(j)
            front = [i for i in range(n) if domination_count[i] == 0]
            return front
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated front
            front_idx = non_dominated_sort(F)
            
            # Select parent from front (or all if front is empty)
            if len(front_idx) == 0:
                parent_idx = rng.integers(self.dim)
                parent_idx = rng.integers(len(X))
            else:
                parent_idx = front_idx[rng.integers(len(front_idx))]
            
            parent = X[parent_idx].copy()
            
            # Mutation: larger step for x1 to explore front, smaller for others
            child = parent.copy()
            child[0] += 0.05 * rng.standard_normal()
            child[1:] += 0.02 * rng.standard_normal(self.dim - 1)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (highest sum of objectives)
            worst_idx = np.argmax(np.sum(F, axis=1))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X

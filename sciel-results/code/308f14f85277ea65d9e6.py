"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using NSGA-II-like approach with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT4, x1 is critical (bounds [-5, 5]), others are [-1, 1]
        # f1 = 1 + 10*(n-1) + sum((xi-0.2)^2 - 0.2*cos(20*pi*(xi-0.2))) for i=2..n
        # f2 = (f1+1) * (1 - sqrt(f1/(f1+1)))
        # The Pareto front is f1 in [1, 11], f2 = (f1+1)*(1-sqrt(f1/(f1+1)))
        
        # Strategy: 
        # 1. Initialize with a mix of random and targeted samples for x1
        # 2. Use a simple evolutionary loop with non-dominated sorting
        # 3. Keep track of all evaluated points
        # 4. Return when budget or time is exhausted

        # Population size
        pop_size = min(50, max(10, self.budget // 4))
        
        # Initial population: mix of uniform and targeted
        # x1 should cover [-5, 5] well since it affects f1 through the cosine term
        # Actually, for ZDT4, f1 depends on all variables, but x1 has the same form as others
        # The key is that the Pareto front is achieved when f1 is minimized for each f2
        
        # Let's use a smarter initialization: sample x1 from a distribution that covers
        # the range where f1 can be small. Since f1 = 1 + 10*(n-1) + sum(...), the minimum
        # of f1 is 1 (when all terms are 0, which happens at xi = 0.2 + k for integer k)
        # But actually the minimum achievable f1 depends on the bounds.
        
        # For ZDT4 with standard bounds, f1 ranges from ~1 to ~11+
        # The Pareto front: f1 in [1, 11], f2 = (f1+1)*(1-sqrt(f1/(f1+1)))
        
        # Let's initialize with random points but ensure good coverage
        rng = np.random.default_rng(42)
        
        # Create initial population
        X = np.empty((pop_size, n))
        # First half: uniform random
        X[:pop_size//2] = lo + (hi - lo) * rng.random((pop_size//2, n))
        # Second half: target x1 values that give good f1 spread
        # f1 = 1 + 10*(n-1) + sum((xi-0.2)^2 - 0.2*cos(20*pi*(xi-0.2)))
        # To minimize f1, set xi near 0.2, 2.2, -3.8, etc. (where cos term is -1 and quadratic is small)
        # For xi in [-5, 5] (x1) or [-1, 1] (others), the good points are:
        # x1: 0.2, -3.8 (since 0.2 - 4 = -3.8, and 0.2 + 2 = 2.2 is out of [-1,1] for others but in [-5,5] for x1)
        # Actually let's just use a few targeted points
        
        # Targeted initialization: set variables to values that minimize their contribution
        # For variables in [-1, 1]: 0.2 is optimal (contribution = 0)
        # For x1 in [-5, 5]: 0.2, 2.2, -3.8 are optimal (contribution = 0)
        
        targeted = np.zeros((pop_size - pop_size//2, n))
        # Set all non-x1 variables to 0.2 (optimal for [-1,1] bounds)
        targeted[:, 1:] = 0.2
        # Spread x1 across good values
        good_x1 = [0.2, 2.2, -3.8, 0.2, 2.2, -3.8, 0.2, 2.2, -3.8]
        for i in range(pop_size - pop_size//2):
            targeted[i, 0] = good_x1[i % len(good_x1)]
        X[pop_size//2:] = targeted
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Store all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        
        # Simple evolutionary loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated points in current population
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            # Select parents from non-dominated
            non_dom_mask = ~dominated
            if not np.any(non_dom_mask):
                # All dominated, pick best by sum
                sums = np.sum(F, axis=1)
                parent_idx = np.argmin(sums)
            else:
                parent_indices = np.where(non_dom_mask)[0]
                parent_idx = parent_indices[rng.integers(len(parent_indices))]
            
            # Create child by mutating parent
            parent = X[parent_idx].copy()
            # Adaptive mutation: larger steps early, smaller later
            progress = used / self.budget
            step_size = 0.5 * (1 - progress) + 0.05
            mutation = step_size * (hi - lo) * rng.standard_normal(n)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Add to all evaluated
            all_X = np.vstack([all_X, child.reshape(1, -1)])
            all_F = np.vstack([all_F, fc.reshape(1, -1)])
            
            # Replace worst point in population (by sum of objectives)
            sums = np.sum(F, axis=1)
            worst_idx = np.argmax(sums)
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return all_X
# EVOLVE-BLOCK-END

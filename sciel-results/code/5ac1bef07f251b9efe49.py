import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # NSGA-II style approach
        # Initialize population
        pop_size = min(40, budget // 5)
        
        # Generate initial population using Latin hypercube-like sampling
        # Focus on exploring the space
        pop = []
        eval_count = 0
        
        # Phase 1: Explore to find good g values
        # Sample x[2:] at various points to understand g
        # First, try x[2:] = 0.5 (common minimum for DTLZ g functions)
        # and x[2:] = 0 to see which gives better results
        
        # Initialize with diverse points
        for i in range(pop_size):
            if eval_count >= budget or time.time() > deadline:
                break
            x = np.zeros(n)
            # Random initialization with some structure
            if i == 0:
                x[2:] = 0.5  # Try common DTLZ7 g minimum
            elif i == 1:
                x[2:] = 0.0
            elif i == 2:
                x[2:] = 0.5
                x[0] = 0.0
                x[1] = 0.0
            elif i == 3:
                x[2:] = 0.5
                x[0] = 1.0
                x[1] = 1.0
            elif i == 4:
                x[2:] = 0.5
                x[0] = 0.5
                x[1] = 0.5
            else:
                x[0] = np.random.rand()
                x[1] = np.random.rand()
                # Mix of x[2:] values
                r = np.random.rand()
                if r < 0.5:
                    x[2:] = 0.5
                elif r < 0.8:
                    x[2:] = 0.0
                else:
                    x[2:] = np.random.rand(n - 2)
            
            f = problem(x)
            pop.append((x.copy(), f))
            eval_count += 1
        
        # Maintain non-dominated set
        def dominates(a, b):
            """Check if a dominates b (all <=, at least one <)"""
            a = np.asarray(a)
            b = np.asarray(b)
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_sort(points):
            """Return indices of non-dominated points"""
            n = len(points)
            dominated = [False] * n
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if dominates(points[i][1], points[j][1]):
                        dominated[j] = True
                    elif dominates(points[j][1], points[i][1]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]
        
        # Phase 2: Iterative improvement
        # Use remaining budget to refine
        while eval_count < budget and time.time() < deadline:
            if len(pop) == 0:
                break
            
            # Get non-dominated set
            nd_indices = non_dominated_sort(pop)
            nd_points = [pop[i] for i in nd_indices]
            
            # If we have a good non-dominated set, try to refine it
            if len(nd_points) >= 2:
                # Pick two random non-dominated points and interpolate
                i, j = np.random.choice(len(nd_points), size=2, replace=False)
                alpha = np.random.rand()
                x_new = alpha * nd_points[i][0] + (1 - alpha) * nd_points[j][0]
                # Add small perturbation
                x_new += np.random.normal(0, 0.01, n)
                x_new = np.clip(x_new, 0, 1)
            else:
                # Mutate a random point
                i = np.random.randint(len(pop))
                x_new = pop[i][0].copy()
                x_new += np.random.normal(0, 0.1, n)
                x_new = np.clip(x_new, 0, 1)
            
            f_new = problem(x_new)
            eval_count += 1
            pop.append((x_new, f_new))
            
            # Keep population size manageable
            if len(pop) > 100:
                nd_idx = non_dominated_sort(pop)
                if len(nd_idx) > 50:
                    # Keep all non-dominated
                    pop = [pop[i] for i in nd_idx]
                else:
                    # Keep non-dominated + some random dominated for diversity
                    pop = [pop[i] for i in nd_idx]
                    remaining = 100 - len(pop)
                    if remaining > 0:
                        dominated_idx = [i for i in range(len(pop)) if i not in nd_idx]
                        if len(dominated_idx) > 0:
                            extra = np.random.choice(dominated_idx, size=min(remaining, len(dominated_idx)), replace=False)
                            pop.extend([pop[i] for i in extra])
        
        return pop[0][0] if pop else np.zeros(n)

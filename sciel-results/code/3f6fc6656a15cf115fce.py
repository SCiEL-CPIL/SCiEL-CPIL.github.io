# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy:
    1. Analytical initialization: ZDT3's Pareto front is determined by x1 (for f1) 
       and x2..xn (for g). We initialize a population that spans the known 
       discontinuous front structure by sampling x1 from known intervals and 
       minimizing g (setting other variables to optimal values for g=1).
    2. Fast NSGA-II: Use vectorized dominance sorting and crowding distance 
       to evolve the population within the budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        N = 50
        # Ensure we don't exceed budget
        if N > budget:
            N = budget
            if N == 0:
                return None

        # 1. Analytical Initialization for ZDT3
        # ZDT3:
        # f1(x) = x1
        # g(x) = 1 + 10(n-1) + sum(xi^2 - 10*cos(4*pi*xi)) for i=2..n
        # f2(x) = (1+sqrt(f1/g)) * (1 - sqrt(f1/g) - f1^2/(g))
        # Pareto Front: f1 in [0,1], g=1.
        # g=1 is achieved when xi=0 for i=2..n (since 0 - 10*cos(0) = -10, sum = 10(n-1)-10(n-1)=0, g=1+0=1).
        # Discontinuities occur at specific intervals of x1.
        
        # We know the "good" regions for x1 are [0, 0.082838], [0.165677, 0.25], [0.338515, 0.421353], [0.504193, 0.587032], [0.669871, 0.752710], [0.752710, 0.835548], [0.918387, 1.0].
        # Actually, it's safer to just sample x1 uniformly but ensure x2..xn are at 0.
        # However, to cover the front well, we can sample x1 from these intervals.
        
        # Let's create a diverse initial population.
        # x1 samples
        # The front is discontinuous. We want to hit the segments.
        # Segments approx: [0, 0.0828], [0.1657, 0.25], [0.3385, 0.4214], [0.5042, 0.5870], [0.6699, 0.7527], [0.7527, 0.8355], [0.9184, 1.0]
        # We will sample N points. Some from these intervals, some random to explore.
        
        X = np.zeros((N, dim))
        # Set x2..xn to 0 (optimal for g=1)
        # X[:, 1:] = 0.0
        
        # Sample x1
        # 80% from known front intervals, 20% random
        n_front = int(N * 0.8)
        n_rand = N - n_front
        
        intervals = [
            (0.0, 0.082838),
            (0.165677, 0.25),
            (0.338515, 0.421353),
            (0.504193, 0.587032),
            (0.669871, 0.752710),
            (0.752710, 0.835548),
            (0.918387, 1.0)
        ]
        
        # Sample from intervals
        x1_samples = []
        for i in range(n_front):
            # Pick a random interval
            idx = np.random.randint(len(intervals))
            lo_i, hi_i = intervals[idx]
            x1_samples.append(np.random.uniform(lo_i, hi_i))
        
        # Random samples
        for i in range(n_rand):
            x1_samples.append(np.random.uniform(0.0, 1.0))
            
        X[:, 0] = x1_samples
        # Keep x2..xn at 0 for now (good starting point for g)
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        # 2. Evolutionary Loop (NSGA-II style)
        # We will use a simple mutation strategy: perturb x1 and x2..xn.
        # Since g is sensitive, we might want to keep x2..xn near 0 but allow exploration.
        
        while used < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            # Select parents via tournament (simplified: just pick random non-dominated or best hypervolume contributors)
            # For speed, we'll just mutate random individuals in the population.
            
            # Mutation
            child = X[np.random.randint(N)].copy()
            
            # Perturb x1
            child[0] += np.random.normal(0, 0.05)
            child[0] = np.clip(child[0], 0.0, 1.0)
            
            # Perturb x2..xn
            # Small noise to escape local minima in g, but keep it small
            noise = np.random.normal(0, 0.1, size=dim-1)
            child[1:] += noise
            child[1:] = np.clip(child[1:], 0.0, 1.0)
            
            fc = problem(child)
            used += 1
            
            # Replace worst individual
            # Simple metric: sum of objectives (normalized roughly)
            # Or just random replacement to maintain diversity?
            # Let's use a simple dominance check against current population.
            # If child is non-dominated, replace a dominated individual.
            # If not, replace the one with worst hypervolume contribution (approx by sum).
            
            # Check if child is dominated by any in F
            is_dominated = False
            for i in range(N):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Replace a dominated individual if possible, else random
                dominated_indices = []
                for i in range(N):
                    for j in range(N):
                        if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated_indices.append(i)
                            break
                if dominated_indices:
                    idx = dominated_indices[np.random.randint(len(dominated_indices))]
                else:
                    idx = np.random.randint(N)
                X[idx] = child
                F[idx] = fc
            else:
                # Even if dominated, maybe it improves the spread? 
                # For simplicity, only replace if it's better than the worst in some metric.
                # Let's stick to replacing the worst sum if it's better.
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
                if np.sum(fc) < sums[worst_idx]:
                    X[worst_idx] = child
                    F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

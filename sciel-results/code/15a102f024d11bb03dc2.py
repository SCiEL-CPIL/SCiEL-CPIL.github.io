import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        N = self.budget
        if N == 0:
            return np.array([])

        # 1. Latin Hypercube Sampling (LHS) for initial coverage
        # This ensures that every dimension is uniformly covered in the first pass.
        # For 200 evaluations in 12 dimensions, a single pass of LHS provides
        # much better coverage of the search space than random sampling,
        # which is critical for capturing the disconnected Pareto front of WFG9.
        
        # Generate LHS points
        # For each dimension, divide [0, 1] into N equal intervals.
        # Pick one random point in each interval.
        # Then permute the rows for each dimension to decorrelate them.
        
        X_lhs = np.zeros((N, self.dim))
        for d in range(self.dim):
            # (i + U_i) / N
            # np.arange(N) is 0..N-1
            # np.random.rand(N) is U_i
            # shuffle the result
            vals = (np.arange(N) + np.random.rand(N)) / N
            np.random.shuffle(vals)
            X_lhs[:, d] = vals
        
        # Map to bounds
        X = lo + X_lhs * span
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = N

        # 2. Local Refinement / Multi-Start Local Search
        # Since we have used our entire budget for exploration, we cannot do a full evolutionary loop.
        # However, we can try to improve the hypervolume by checking if we have any remaining budget 
        # (which we don't, since N=budget) OR we can simply return the LHS set.
        
        # Wait, the strategy says "Use a deterministic Latin Hypercube Sampling (LHS) strategy to generate the first 200 points".
        # This implies we might not have budget left for refinement if we use all 200 for LHS.
        # But the current program (0.6096) uses an EA. The inspiration (0.7600) uses ONLY LHS.
        # The directive says "COMBINE the strategy of the current program with the strongest ideas of the inspirations".
        # The strongest idea is LHS coverage. The current program's idea is local search/refinement.
        
        # If I use all 200 for LHS, I have no budget left for refinement.
        # If I use 100 for LHS, I have 100 for refinement.
        
        # Let's look at the scores:
        # Current (EA): 0.6096
        # Inspiration 1 (LHS only): 0.7600
        # Inspiration 2 (NSGA-II): 0.6831
        
        # It seems pure LHS is better than the EA in the current program.
        # Why? Maybe the EA gets stuck or the initial population is poor.
        # LHS ensures global coverage.
        
        # Can we do better than pure LHS?
        # Perhaps a hybrid: Use LHS for the first 80% of the budget, then use the remaining 20% to refine the best non-dominated points?
        # Or maybe the LHS is just so good for W9 that refinement isn't needed or is hard to do without knowing the structure.
        
        # Let's try a hybrid approach:
        # 1. Generate 150 LHS points.
        # 2. Evaluate them.
        # 3. Identify the non-dominated set.
        # 4. Use the remaining 50 evaluations to do a local search around the non-dominated points.
        
        # However, the inspiration 1 score (0.76) is significantly higher than current (0.61).
        # The directive asks to combine.
        # If I just copy Inspiration 1, I am not combining.
        # If I add local search to LHS, I might lose the benefits of pure LHS if the local search is bad.
        
        # Let's analyze WFG9 structure.
        # WFG9 is a deceptive problem. The variables x2...x12 are "position" variables that affect the shape of the front but not the position of the ideal point directly in the same way as x1.
        # Actually, for WFG9, f1 = x1. f2 and f3 are complex functions of x2...x12 and x1.
        # The Pareto front is defined by the minimization of the shape functions.
        # If we sample uniformly in x1 (which LHS does), we cover the trade-off axis well.
        # The other variables need to be at specific values to minimize the "distance" to the ideal front.
        # Random sampling (LHS) might hit these good regions by chance, but it's sparse.
        # Local search could refine the position variables for a given x1.
        
        # Let's implement a hybrid:
        # 1. LHS for N_init = 120 points.
        # 2. Evaluate.
        # 3. Find non-dominated set.
        # 4. For each point in the non-dominated set (or a subset), do a few local perturbations on the position variables (x2..x12) while keeping x1 fixed or slightly varied.
        
        # But we only have 200 total.
        # If we use 120 for LHS, we have 80 left.
        # If there are ~10-20 non-dominated points, we can do ~4-8 refinements per point.
        
        # Let's define the hybrid.
        
        N_init = int(0.6 * N)  # 120 points
        N_refine = N - N_init  # 80 points
        
        # Generate LHS points
        X_lhs = np.zeros((N_init, self.dim))
        for d in range(self.dim):
            vals = (np.arange(N_init) + np.random.rand(N_init)) / N_init
            np.random.shuffle(vals)
            X_lhs[:, d] = vals
        
        X = lo + X_lhs * span
        F = np.array([problem(x) for x in X])
        used = N_init

        # Helper: non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            if current_front:
                fronts.append(current_front)
            
            k = 0
            while k < len(fronts) and fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
            return fronts

        # Get the first front (non-dominated)
        fronts = non_dominated_sort(F)
        front0 = fronts[0] if fronts else []
        
        # Refinement phase
        # We want to refine the position variables (indices 1 to dim-1) for the points in front0.
        # We keep x1 (index 0) fixed or slightly perturbed.
        
        if front0:
            # Shuffle the front to randomize the order of refinement
            np.random.shuffle(front0)
            
            # Distribute remaining budget among the front points
            # We can't do deep local search with 80 evals and ~20 points.
            # Maybe 2-3 evals per point?
            
            for idx in front0:
                if used >= N:
                    break
                if time.time() > limit_time:
                    break
                
                x_parent = X[idx]
                
                # Try a few local perturbations
                # Focus on dimensions 1..11
                n_local = min(3, N - used)
                
                for _ in range(n_local):
                    if used >= N:
                        break
                    if time.time() > limit_time:
                        break
                    
                    # Perturb position variables only
                    x_child = x_parent.copy()
                    # Small step for local search
                    step = 0.05 * span
                    # Perturb dims 1 to end
                    perturbation = np.random.normal(0, step, self.dim - 1)
                    x_child[1:] += perturbation
                    
                    # Keep x1 fixed to maintain the trade-off position
                    # x_child[0] = x_parent[0]
                    
                    x_child = np.clip(x_child, lo, hi)
                    
                    f_child = problem(x_child)
                    used += 1
                    
                    # Add to population
                    X = np.vstack([X, x_child])
                    F = np.vstack([F, f_child])
        
        return X

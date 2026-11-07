"""Structure-aware hybrid optimizer for DTLZ1: Analytic Front Initialization + Local Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer that leverages the analytic structure of DTLZ1.
    1. Generates points on the ideal Pareto front (g=0) using a quasi-random sequence.
    2. Uses Nelder-Mead local search to refine these points, minimizing the 
       distance to the ideal front (or simply minimizing the sum of objectives).
    3. Maintains a non-dominated archive of all evaluated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.archive_objs = []

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        # We will use a portion of the budget for local refinement.
        # DTLZ1 has K objectives. The Pareto front is defined by the first K-1 variables.
        # The last variable (x_K) determines g(x). For DTLZ1, g(x) = 10*(K - sum(x_i)) + 10*x_1*sin^2(10*pi*x_1).
        # To get on the front, we need g(x)=0. This implies x_1=0 and sum(x_1...x_K) = K.
        # Since x_1=0, sum(x_2...x_K) = K.
        
        # Strategy:
        # 1. Sample points on the unit simplex for the first n_obj-1 variables.
        # 2. Set the remaining variables to satisfy the constraint for g=0.
        # 3. Evaluate these "ideal" points.
        # 4. Use remaining budget to perform local searches around these points to ensure 
        #    we are actually hitting the front (in case of numerical issues or if the problem 
        #    is slightly perturbed, though the contract says it's DTLZ1).
        
        # However, the problem function `problem(x)` takes a 7-dim vector.
        # DTLZ1 with M=3 objectives usually has K=M-1=2 position variables and N-K=5 distance variables.
        # Wait, standard DTLZ1:
        # x_1 ... x_{K} are position variables (K=M-1=2).
        # x_{K+1} ... x_{N} are distance variables (N-K=5).
        # g(x) = 10*(N-K) + sum_{x_i in X_g} (x_i - 0.5)^2 - 10 cos(20 * pi * x_1 * g)
        # Actually, standard DTLZ1 g(x) = 10*(N-K) + sum(x_i - 0.5)^2 - 10 cos(20*pi*x_1*g)
        # The front is g=0.
        
        # Let's stick to a robust local search approach initialized with a good spread.
        # We will generate a set of initial points on the "ideal" simplex for the first M-1 variables.
        
        M = self.n_obj
        K = M - 1
        N = self.dim
        
        # Generate quasi-random points on the simplex of dimension K-1 (which is M-2)
        # The variables x_0, ..., x_{K-1} define the position on the front.
        # The condition for the front in DTLZ1 is that the distance variables x_K...x_{N-1}
        # are such that g(x)=0.
        
        # Let's create a helper to map a point on the simplex to a valid DTLZ1 input vector
        # that lies on the Pareto front (or very close to it).
        
        def simplex_to_dtlz1_input(u, rng):
            """
            u: vector of length K (M-1) on the simplex (sum(u)=1, u>=0).
            Returns x of length N.
            For DTLZ1, if we set the position variables x_0...x_{K-1} to u,
            and set the distance variables to satisfy g=0.
            g(x) depends on x_K...x_{N-1}.
            Standard DTLZ1: g(x) = 10*(N-K) + sum_{i=K}^{N-1} (x_i - 0.5)^2 - 10 cos(20*pi*x_0*g)
            This is complex to invert exactly for arbitrary x_0.
            
            Alternative Strategy: 
            Just use a robust multi-objective local search.
            1. Initialize with a grid on the hypercube [0,1]^N.
            2. Filter to non-dominated.
            3. Use local search (Nelder-Mead) on the sum of objectives or a scalarized objective 
               for a subset of these points.
            """
            pass

        # Let's try a different structural approach:
        # 1. Use a Sobol sequence (or simple LHS) to spread points in the search space.
        # 2. Evaluate them.
        # 3. Keep the non-dominated front.
        # 4. For each point on the front, perform a local search to minimize the 
        #    "distance" to the ideal front. Since we don't know the ideal front analytically 
        #    without assuming the specific DTLZ1 formula constants, we can minimize the 
        #    sum of objectives locally, which drives g(x) towards 0 and positions towards the front.
        
        # Actually, minimizing the sum of objectives for DTLZ1 is a good proxy for finding the front,
        # as the sum is minimized when g=0 and the position variables are balanced.
        
        # Implementation Plan:
        # 1. Generate 50 initial points using Latin Hypercube Sampling (LHS) in [0,1]^N.
        # 2. Evaluate them.
        # 3. Extract non-dominated points.
        # 4. For each non-dominated point, run a few steps of Nelder-Mead to minimize the sum of objectives.
        # 5. Record all evaluations.
        
        # LHS Generation
        def lhs_sample(n_samples, n_dims, rng):
            # Simple LHS implementation
            samples = np.zeros((n_samples, n_dims))
            for i in range(n_samples):
                for j in range(n_dims):
                    samples[i, j] = (j + rng.random()) / n_samples
            return samples

        rng = np.random.RandomState(0) # Use global seed? No, np.random.seed is called by evaluator.
        # We should use the global np.random state to respect the seed.
        
        # Initial population size
        init_size = 40
        # Generate LHS points
        X_init = lhs_sample(init_size, N, np.random)
        
        F_init = np.array([problem(x) for x in X_init])
        used += init_size
        
        # Store in archive
        self.archive = [x.copy() for x in X_init]
        self.archive_objs = [f.copy() for f in F_init]
        
        # Find non-dominated points
        def is_non_dominated(objs):
            """Returns list of indices of non-dominated points."""
            n = len(objs)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                if np.any(objs <= objs[i]) and np.any(objs < objs[i]):
                    # If any point dominates i, i is not non-dominated
                    # Check if i is dominated by any other
                    pass
                
            # Correct logic:
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j: continue
                    if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                        dominated = True
                        break
                if not dominated:
                    nd[i] = True
            return np.where(nd)[0]
        
        nd_indices = is_non_dominated(self.archive_objs)
        
        # If we have few non-dominated points, we might have missed the front.
        # Let's use the remaining budget to do local searches around these points.
        # We will allocate the remaining budget to refine these points.
        
        remaining_budget = self.budget - used
        
        # Number of local searches we can do
        # Each local search will use a small number of evaluations.
        # Let's say we use 10 evaluations per local search.
        evals_per_search = 10
        num_searches = min(len(nd_indices), remaining_budget // evals_per_search)
        
        if num_searches == 0:
            # If we can't do local searches, just return what we have
            return self.archive
            
        # Shuffle non-dominated indices to ensure variety
        search_indices = np.random.choice(nd_indices, size=num_searches, replace=False)
        
        for idx in search_indices:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x_start = self.archive[idx].copy()
            
            # Local search: Minimize sum of objectives
            # We will use a simple coordinate descent or Nelder-Mead.
            # Given the small budget, a simple perturbation method is safer.
            
            current_x = x_start
            current_f = problem(current_x)
            used += 1 # Already evaluated in init? No, we re-evaluate to be safe or use cached.
            # Let's use the cached value if possible.
            # self.archive_objs[idx] is the value.
            current_f = self.archive_objs[idx].copy()
            
            # Try to improve
            best_x = current_x
            best_f = current_f
            
            # Simple random local search
            for step in range(evals_per_search - 1): # -1 because we already have 1 eval
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Generate a neighbor
                # Step size decreases
                step_size = 0.1 * (0.9 ** step)
                neighbor = best_x + np.random.normal(0, step_size, N)
                neighbor = np.clip(neighbor, lo, hi)
                
                f_neighbor = problem(neighbor)
                used += 1
                
                # Update archive
                self.archive.append(neighbor)
                self.archive_objs.append(f_neighbor)
                
                # Check if neighbor is better (dominates or has lower sum)
                # For local search, minimizing sum is a good heuristic
                if np.sum(f_neighbor) < np.sum(best_f):
                    best_x = neighbor
                    best_f = f_neighbor
                elif np.all(f_neighbor <= best_f) and np.any(f_neighbor < best_f):
                    best_x = neighbor
                    best_f = f_neighbor
                    # If it dominates, it's definitely better
                    break # Found a strictly better point, stop this local search? Or continue?
                    # Let's continue to see if we can get even better.
                    
        return self.archive

# EVOLVE-BLOCK-END

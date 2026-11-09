import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weights(self, N, M):
        # Generate N weight vectors summing to 1
        # Using Dirichlet distribution for random uniform points on simplex
        rng = np.random.default_rng(42) # Fixed seed for weights to be deterministic? 
        # The task says np.random.seed(seed) is called by evaluator. 
        # We should use the global np.random or a local rng? 
        # "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # It's safer to use np.random directly or a local rng seeded from np.random.
        # Let's use np.random to respect the global seed state if needed, 
        # but local rng is cleaner. However, the prompt implies the evaluator controls the seed.
        # I will use np.random for the search to ensure it varies with the seed.
        
        # Actually, using a local rng seeded from np.random.get_state is overkill.
        # Just use np.random.
        
        weights = np.random.dirichlet(np.ones(M), size=N)
        return weights

    def _compute_neighbors(self, weights, k):
        N = len(weights)
        neighbors = []
        # Compute pairwise distances
        # For small N (20), O(N^2) is fine
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        for i in range(N):
            # Sort indices by distance, exclude self
            sorted_idx = np.argsort(dists[i])
            # Take first k+1 (including self), then remove self
            neigh = sorted_idx[1:k+1]
            neighbors.append(neigh)
        return neighbors

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Budget 200. 
        # If N is too large, we don't get many generations.
        # If N is too small, we lack diversity.
        # N=20 is a good start. 200/20 = 10 generations.
        N = 20
        
        # 1. Generate Weight Vectors
        weights = self._generate_weights(N, self.n_obj)
        
        # 2. Initialize Population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point
        ideal = np.min(F, axis=0)
        
        # 3. Compute Neighbors
        neighbors = self._compute_neighbors(weights, k=5)
        
        # DE Parameters
        F_de = 0.5
        CR = 0.9
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select a random subproblem index
            i = np.random.randint(0, N)
            
            # Get neighbors for i
            neigh = neighbors[i]
            
            # Select 3 distinct parents from neighbors
            # Ensure we have at least 3 unique candidates
            # If neighborhood is small, add random global indices
            candidates = list(neigh)
            while len(candidates) < 3:
                candidates.append(np.random.randint(0, N))
            
            # Shuffle and pick 3
            np.random.shuffle(candidates)
            r1, r2, r3 = candidates[0], candidates[1], candidates[2]
            
            # Ensure distinctness just in case (though shuffle+unique logic above helps)
            # If r1==r2, pick another
            if r1 == r2 or r2 == r3 or r1 == r3:
                 # Fallback to global random distinct
                 indices = np.random.choice(N, size=3, replace=False)
                 r1, r2, r3 = indices
            
            # DE Mutation
            v = X[r1] + F_de * (X[r2] - X[r3])
            
            # Crossover
            jrand = np.random.randint(0, self.dim)
            cross_mask = np.random.random(self.dim) <= CR
            cross_mask[jrand] = True
            
            child = np.where(cross_mask, v, X[i])
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            used += 1
            
            # Update Ideal Point
            ideal = np.minimum(ideal, f_child)
            
            # Tchebycheff Distance
            # g(x) = max_j( w_j * |f_j(x) - ideal_j| )
            
            # Calculate for child
            dist_child = np.max(weights[i] * np.abs(f_child - ideal))
            
            # Calculate for current X[i]
            dist_current = np.max(weights[i] * np.abs(F[i] - ideal))
            
            # Accept if better
            if dist_child < dist_current:
                X[i] = child
                F[i] = f_child
                
        return X

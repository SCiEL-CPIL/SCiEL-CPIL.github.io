import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use global numpy random state to respect evaluator's seed
        # np.random.seed is called by evaluator, so we use np.random directly
        
        M = 20 # Population size / Number of subproblems
        n_neighbors = 5
        
        # 1. Generate Weight Vectors
        # Uniform points on simplex for n_obj=3
        # We can use a grid or random. Let's use a simple grid for 3D.
        # For 3 objectives, we can generate points such that sum is 1.
        # A common way is to use a recursive subdivision or just random points and normalize.
        # For reproducibility and coverage, let's generate M random points on the simplex.
        # To ensure good spread, we might add a few extreme points (0,0,1), (0,1,0), (1,0,0).
        
        weights = []
        # Add corners
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        
        # Add random points to fill up to M
        while len(weights) < M:
            # Random Dirichlet(1,1,1)
            w = np.random.dirichlet([1.0]*self.n_obj)
            weights.append(w)
        
        weights = np.array(weights)
        
        # 2. Build Neighbor List
        # Calculate pairwise distances between weights
        dists = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        neighbors = []
        for i in range(M):
            # Get indices of closest n_neighbors, excluding self
            idx = np.argsort(dists[i])
            idx = idx[1:n_neighbors+1] # Exclude self (index 0)
            neighbors.append(idx)
            
        # 3. Initialize Population
        X = lo + (hi - lo) * np.random.random((M, self.dim))
        F = np.array([problem(x) for x in X])
        
        used = M
        
        # Ideal point (min for each objective)
        ideal = np.min(F, axis=0)
        
        # Tchebycheff aggregation function
        def tchebycheff(f, w, ideal):
            # g = max_k (w_k * |f_k - ideal_k|)
            # Note: Standard Tchebycheff is max_k (w_k * (f_k - ideal_k)) if minimizing
            # But since we want to minimize, and f_k >= ideal_k usually, 
            # we use max_k (w_k * (f_k - ideal_k)).
            # However, if f_k < ideal_k (impossible if ideal is min), it would be negative.
            # To be safe, use max_k (w_k * abs(f_k - ideal_k)) or just (f_k - ideal_k).
            # Given ideal is the min seen so far, f_k >= ideal_k.
            vals = w * (f - ideal)
            return np.max(vals)
        
        # Calculate current Tchebycheff values for each subproblem
        # t_val[i] is the Tchebycheff value of X[i] with weight weights[i]
        t_vals = np.array([tchebycheff(F[i], weights[i], ideal) for i in range(M)])
        
        # 4. Evolution Loop
        while used < self.budget:
            # Determine how many offspring we can generate in this "generation"
            # We generate 1 offspring per subproblem per generation step.
            # But we might not have enough budget for a full generation.
            # Let's process subproblems one by one until budget is exhausted.
            
            for i in range(M):
                if used >= self.budget:
                    break
                
                # Select a neighbor j
                j = neighbors[i][np.random.randint(n_neighbors)]
                
                # Mutation: DE-like
                # Pick two other random individuals r1, r2
                r1 = np.random.randint(M)
                r2 = np.random.randint(M)
                # Ensure r1, r2 are not i or j? Not strictly necessary but good.
                # Simple DE: v = X[r1] + F * (X[r2] - X[j])
                F_factor = 0.8
                v = X[r1] + F_factor * (X[r2] - X[j])
                
                # Add some random perturbation to ensure exploration
                # v = v + 0.1 * (hi - lo) * np.random.randn(self.dim)
                
                v = np.clip(v, lo, hi)
                
                fv = problem(v)
                used += 1
                
                # Update Ideal Point
                ideal = np.minimum(ideal, fv)
                
                # Update Tchebycheff values for affected individuals
                # We need to re-calc t_vals for i and j if we replace them.
                # Also, replacing an individual changes the ideal point, which changes
                # the t_vals for ALL individuals. 
                # This is expensive. 
                # Optimization: We can keep the old ideal for the comparison step 
                # and update the global ideal only after the full generation?
                # Or update ideal and re-calc t_vals for i and j only?
                # Actually, if ideal changes, the relative ordering might change.
                # However, for small M, re-calculating all t_vals is O(M^2 * n_obj) which is fine.
                
                # Let's check if v is better than X[i] for subproblem i
                t_v_i = tchebycheff(fv, weights[i], ideal)
                t_x_i = tchebycheff(F[i], weights[i], ideal)
                
                replace_i = False
                if t_v_i < t_x_i:
                    X[i] = v
                    F[i] = fv
                    replace_i = True
                    
                # Check if v is better than X[j] for subproblem j
                t_v_j = tchebycheff(fv, weights[j], ideal)
                t_x_j = tchebycheff(F[j], weights[j], ideal)
                
                replace_j = False
                if t_v_j < t_x_j:
                    X[j] = v
                    F[j] = fv
                    replace_j = True
                    
                # If ideal changed, we should update t_vals for all?
                # For simplicity and correctness of the "better" check in subsequent steps,
                # we should update the stored t_vals.
                # But we don't store t_vals as a separate array that is used for selection in this loop,
                # we calculate on the fly.
                # The `t_x_i` and `t_x_j` used above were calculated with the NEW ideal.
                # This is consistent.
                
        # 5. Extract Non-Dominated Solutions
        # Find non-dominated points in F
        n = len(F)
        nondom_indices = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nondom_indices.append(i)
                
        return F[nondom_indices]

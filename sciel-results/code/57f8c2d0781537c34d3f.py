import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use N=21 for H=5 grid in 3 objectives
        N = 21
        if self.budget < N:
            # Fallback if budget is too small
            N = self.budget
            
        # Generate weight vectors
        # For 3 objectives, weights (w1, w2, w3) sum to 1.
        # Grid with H=5: i+j+k=5, i,j,k >= 0.
        # Points: (i/5, j/5, k/5)
        weights = []
        for i in range(6):
            for j in range(6 - i):
                k = 5 - i - j
                if k >= 0:
                    weights.append(np.array([i/5.0, j/5.0, k/5.0]))
        # We have 21 weights.
        # If N < 21, we need to subset. If N > 21, we duplicate or use finer grid.
        # Here N=21 matches exactly.
        if N != len(weights):
             # Resample or truncate
             if N < len(weights):
                 # Randomly select N weights
                 indices = np.random.choice(len(weights), N, replace=False)
                 weights = [weights[i] for i in indices]
             else:
                 # Duplicate last weight? Or just use N=21
                 pass
                 
        weights = np.array(weights[:N])
        
        # Neighborhood
        T = 5
        # Calculate distances between weights
        # dist[i][j] = ||weights[i] - weights[j]||
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i][j] = np.linalg.norm(weights[i] - weights[j])
        
        # Find T nearest neighbors for each i
        neighborhoods = []
        for i in range(N):
            # Sort indices by distance to i
            order = np.argsort(dists[i])
            # Exclude i itself? Usually neighborhood includes i or not.
            # Standard MOEA/D: B_i contains T individuals closest to i, including i? 
            # Often B_i is defined as T closest neighbors. Let's include i.
            # If we include i, we pick T from the sorted list starting with i.
            # But we need to select parents from B_i.
            # Let's define B_i as the T closest indices, including i.
            b_i = order[:T]
            neighborhoods.append(b_i)

        # Initialize Population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Ideal Point
        z_star = np.min(F, axis=0)
        
        # Tchebycheff Distance Function
        def tchebycheff(f, w, z):
            # f: (n_obj,)
            # w: (n_obj,)
            # z: (n_obj,)
            # g = max_k (w_k * |f_k - z_k|)
            # If w_k is 0, the term is 0.
            # We should avoid division by zero if we normalized, but we didn't.
            # Weights sum to 1.
            diff = np.abs(f - z)
            # Handle case where w_k = 0
            terms = w * diff
            return np.max(terms)
        
        # Main Loop
        while evals + N <= self.budget:
            if time.time() > deadline:
                break
                
            for i in range(N):
                if time.time() > deadline:
                    break
                    
                # Select Parents
                # Parent 1 from Neighborhood of i
                # Parent 2 from Whole Population (or Neighborhood)
                # Let's use Neighborhood for both to keep it tight, or one global.
                # Standard: 0.9 prob select from neighborhood, 0.1 from whole.
                # Or simply: p1 from B_i, p2 from whole.
                
                idx1 = neighborhoods[i][np.random.randint(T)]
                
                # For p2, let's pick from whole population to encourage global mix
                idx2 = np.random.randint(N)
                
                p1 = X[idx1]
                p2 = X[idx2]
                
                # Crossover
                c = self._sbx_crossover(p1, p2, lo, hi)
                
                # Mutation
                c = self._poly_mutation(c, lo, hi)
                
                # Evaluate
                f_c = problem(c)
                evals += 1
                
                # Update Ideal Point
                z_star = np.minimum(z_star, f_c)
                
                # Update Neighborhood
                # Check all j in B_i (including i)
                for j in neighborhoods[i]:
                    # Calculate Tchebycheff distance for current X[j] and new c
                    # Note: We must use the UPDATED z_star? 
                    # In MOEA/D, z* is updated globally.
                    # The comparison is: if g(c, w_j, z*) < g(X[j], w_j, z*)
                    # However, z* changes during the generation.
                    # This is standard.
                    
                    g_c = tchebycheff(f_c, weights[j], z_star)
                    g_curr = tchebycheff(F[j], weights[j], z_star)
                    
                    if g_c < g_curr:
                        X[j] = c
                        F[j] = f_c
                        
        return X

    def _sbx_crossover(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if p1[i] != p2[i]:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / 15)
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / 15)
                    
                    c1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    c2[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1

    def _poly_mutation(self, x, lo, hi):
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / 20) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / 20)
                
                x[i] = x[i] + delta * (hi[i] - lo[i])
        
        return np.clip(x, lo, hi)

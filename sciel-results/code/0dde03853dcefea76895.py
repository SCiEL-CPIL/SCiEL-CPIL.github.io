import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weights(self, N, M):
        # Generate N evenly distributed weight vectors on the simplex
        # Using a grid-based approach for better coverage
        if M == 3:
            # For 3 objectives, use a triangular grid
            weights = []
            # Number of points along each edge
            H = int(np.ceil((N**0.5)))
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    if i + j + k == H:
                        w = np.array([i / H, j / H, k / H])
                        weights.append(w)
            # If we have more than needed, subsample
            if len(weights) > N:
                idx = np.random.choice(len(weights), N, replace=False)
                weights = [weights[i] for i in idx]
            elif len(weights) < N:
                # Add random points to fill
                while len(weights) < N:
                    w = np.random.dirichlet(np.ones(M))
                    weights.append(w)
            return np.array(weights)
        else:
            # General case: use Dirichlet
            return np.random.dirichlet(np.ones(M), size=N)

    def _compute_neighbors(self, weights, k):
        N = len(weights)
        neighbors = []
        # Compute pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        for i in range(N):
            sorted_idx = np.argsort(dists[i])
            neigh = sorted_idx[1:k+1]
            neighbors.append(neigh)
        return neighbors

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
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
        
        # DE Parameters - will be adaptive
        F_de_start = 0.8
        F_de_end = 0.3
        CR = 0.9
        
        total_budget = self.budget - N
        gen = 0
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Adaptive F_de
            progress = min(1.0, used / max(1, total_budget))
            F_de = F_de_start + (F_de_end - F_de_start) * progress
            
            # Select a random subproblem index
            i = np.random.randint(0, N)
            
            # Get neighbors for i
            neigh = neighbors[i]
            
            # Select 3 distinct parents from neighbors
            candidates = list(neigh)
            while len(candidates) < 3:
                candidates.append(np.random.randint(0, N))
            
            np.random.shuffle(candidates)
            r1, r2, r3 = candidates[0], candidates[1], candidates[2]
            
            if r1 == r2 or r2 == r3 or r1 == r3:
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
            gen += 1
            
            # Update Ideal Point
            ideal = np.minimum(ideal, f_child)
            
            # Chebyshev Distance
            dist_child = np.max(weights[i] * np.abs(f_child - ideal))
            dist_current = np.max(weights[i] * np.abs(F[i] - ideal))
            
            # Accept if better in Chebyshev distance
            if dist_child < dist_current:
                X[i] = child
                F[i] = f_child
        
        # Final polish: local search around the best points
        if used < self.budget and time.time() < deadline:
            # Find the non-dominated set
            # Simple non-domination check
            nondom_idx = []
            for i in range(N):
                dominated = False
                for j in range(N):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nondom_idx.append(i)
            
            if len(nondom_idx) > 0:
                # Pick the best non-dominated point for local refinement
                # Use the point with smallest sum of objectives
                sums = F[nondom_idx].sum(axis=1)
                best_local = nondom_idx[np.argmin(sums)]
                
                # Local search with small perturbations
                local_budget = min(20, self.budget - used)
                for _ in range(local_budget):
                    if time.time() > deadline:
                        break
                    # Small Gaussian perturbation
                    sigma = 0.05 * (hi - lo)
                    child = X[best_local] + np.random.normal(0, sigma)
                    child = np.clip(child, lo, hi)
                    f_child = problem(child)
                    used += 1
                    ideal = np.minimum(ideal, f_child)
                    
                    # Accept if it improves the Chebyshev distance for that subproblem
                    # Find which weight this point is closest to
                    dists = np.max(weights * np.abs(f_child - ideal), axis=1)
                    best_w = np.argmin(dists)
                    
                    dist_current = np.max(weights[best_w] * np.abs(F[best_local] - ideal))
                    if dists[best_w] < dist_current:
                        X[best_local] = child
                        F[best_local] = f_child
        
        return X

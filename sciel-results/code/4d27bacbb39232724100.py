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
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 30 # Population size
        # Generate weight vectors
        W = rng.random((N, self.n_obj))
        W = W / W.sum(axis=1, keepdims=True)
        
        # Initial population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point
        z_star = F.min(axis=0)
        
        # Define Tchebycheff
        def tcheb(f, w, z):
            return np.max(w * np.abs(f - z))
            
        # Calculate initial Tchebycheff values
        G = np.array([tcheb(F[i], W[i], z_star) for i in range(N)])
        
        # Neighborhood (simple: all are neighbors for small N, or define by distance)
        # For N=30, we can just use all other points as potential neighbors or a subset.
        # Let's use a simple strategy: when updating, compare with all points? No, that's O(N^2).
        # Standard MOEA/D uses a neighborhood list. Let's create it based on weight distance.
        # Distance between weight vectors
        # For simplicity and robustness with small N, let's just allow replacement of any point 
        # if it improves the Tchebycheff value for that specific weight.
        # Actually, standard MOEA/D updates neighbors. Let's define neighbors as those with distance < threshold.
        
        # Calculate pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i,j] = np.linalg.norm(W[i] - W[j])
                
        # Threshold: top 10 nearest neighbors
        T = 10
        neighbors = []
        for i in range(N):
            idxs = np.argsort(dists[i])
            neighbors.append(idxs[:T])
            
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a weight vector
            k = rng.integers(N)
            nbrs = neighbors[k]
            
            # Select parents from neighborhood
            p1_idx = nbrs[rng.integers(len(nbrs))]
            p2_idx = nbrs[rng.integers(len(nbrs))]
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            
            # SBX Crossover
            child = np.zeros_like(p1)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    child[i] = p1[i]
                else:
                    child[i] = p2[i]
            # Apply SBX properly
            # Simplified SBX
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (15 + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (15 + 1))
                
            for i in range(self.dim):
                if rng.random() < 0.5:
                    child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                else:
                    child[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
            # Polynomial Mutation
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    delta1 = rng.random()
                    delta2 = 1 - delta1
                    delta = (2 * delta1) ** (1 / (20 + 1)) - 1
                    delta = (2 * delta2) ** (1 / (20 + 1)) - 1
                    delta = delta if rng.random() < 0.5 else -delta
                    child[i] = child[i] + delta * (hi[i] - lo[i])
                    
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update ideal point
            z_star = np.minimum(z_star, fc)
            
            # Update population
            # Compare with neighbors
            for j in nbrs:
                # If child is better than X[j] for weight W[j]
                if tcheb(fc, W[j], z_star) < G[j]:
                    X[j] = child
                    F[j] = fc
                    G[j] = tcheb(fc, W[j], z_star)
                    
        return X

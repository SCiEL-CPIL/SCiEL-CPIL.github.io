import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        N = 20
        
        # 1. Generate Weight Vectors on Simplex
        # For n_obj=3, we can generate uniform points on the simplex
        weights = []
        # Use a grid or random. Random is fine if N is small, but grid is more structured.
        # Let's use a simple recursive simplex generation or just random + deduplication.
        # Actually, for 3 objectives, a grid of H=4 gives 10 points, H=5 gives 21.
        # Let's generate 21 points and pick 20, or just use 20 random points.
        # Random points on simplex:
        while len(weights) < N:
            # Generate random point in simplex
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            w = np.array([r1, r2, 1 - r1 - r2])
            # Check distance to existing weights to ensure spread
            if all(np.linalg.norm(w - ew) > 0.1 for ew in weights):
                weights.append(w)
        
        weights = np.array(weights)
        
        # 2. Define Neighbors
        # Find 5 nearest neighbors for each weight vector
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Sort indices by distance, exclude self
            sorted_idx = np.argsort(dists)
            neigh = [sorted_idx[j] for j in range(1, 6) if sorted_idx[j] != i]
            neighbors.append(neigh)
        
        # 3. Initialize Population using Latin Hypercube Sampling
        # For dim=4, N=20
        X = np.zeros((N, dim))
        for d in range(dim):
            # Create 20 bins
            edges = np.linspace(lo[d], hi[d], N + 1)
            centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N)])
            # Shuffle to decorrelate dimensions
            np.random.shuffle(centers)
            X[:, d] = centers
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        # Initialize Ideal Point (z*)
        z_star = np.min(F, axis=0)
        
        # 4. Evolution Loop
        # We have budget - used evaluations left.
        # We will iterate through individuals cyclically.
        
        i_counter = 0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            i = i_counter % N
            i_counter += 1
            
            # Select parents from neighbors of i
            # If neighbors list is empty or small, use whole population
            candidate_indices = neighbors[i]
            if len(candidate_indices) < 3:
                # Pick random from whole population
                candidate_indices = np.random.choice(N, 3, replace=False)
            else:
                # Pick 3 distinct from neighbors
                candidate_indices = np.random.choice(candidate_indices, 3, replace=False)
            
            r1, r2, r3 = candidate_indices
            
            # DE Mutation
            F_mut = 0.8 # Mutation factor
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Crossover (Binomial)
            # For dim=4, we can just use v directly if we want, 
            # or do a simple blend.
            # Standard DE uses binomial crossover.
            j_rand = np.random.randint(dim)
            CR = 0.5 # Crossover rate
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1
            
            # Update Ideal Point
            z_star = np.minimum(z_star, f_trial)
            
            # Update Population (MOEA/D strategy)
            # Check current individual i and its neighbors
            indices_to_check = [i] + neighbors[i]
            
            for k in indices_to_check:
                # Calculate Tchebycheff distance for trial and current X[k]
                # d_te(f, z*, w) = max(w_i * |f_i - z*_i|)
                
                # We need to handle the case where z* is updated.
                # The formula uses the current z*.
                
                d_trial = np.max(weights[k] * np.abs(f_trial - z_star))
                d_current = np.max(weights[k] * np.abs(F[k] - z_star))
                
                if d_trial < d_current:
                    X[k] = trial
                    F[k] = f_trial

        return X

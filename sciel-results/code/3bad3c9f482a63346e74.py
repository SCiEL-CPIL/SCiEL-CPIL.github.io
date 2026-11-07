"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using MOEA/D (Decomposition-based)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Generate weight vectors for decomposition
        # For 3 objectives, we use a simplex lattice.
        # Number of subproblems N determines the resolution.
        # With budget 200, we can afford N=50 subproblems (4 evaluations per subproblem initially, then 1 each).
        # Let's use N=40 to be safe and allow more iterations.
        N = 40
        
        # Generate weights for 3 objectives that sum to 1
        # Using a grid on the simplex
        weights = []
        # Step size for the grid. 
        # We want roughly N points.
        # For 3D simplex, number of points with step size h (integers) is (K+2 choose 2) where K = 1/h
        # We want (K+2)(K+1)/2 approx N
        # K=8 -> 45 points. K=7 -> 36 points.
        # Let's use K=8 (step 0.125) for 45 weights, or just pick N random points on simplex.
        # Random points on simplex are easier to generate for arbitrary N.
        for _ in range(N):
            # Dirichlet distribution with alpha=1 gives uniform points on simplex
            w = rng.dirichlet(np.ones(self.n_obj))
            weights.append(w)
        weights = np.array(weights)
        
        # 2. Initialize population
        # Each subproblem i has a solution x_i
        X = np.array([lo + (hi - lo) * rng.random(self.dim) for _ in range(N)])
        F = np.array([problem(x) for x in X])
        
        # Record all evaluations
        all_F = [F]
        used = N
        
        # 3. Define neighborhood
        # In MOEA/D, each subproblem interacts with neighbors in weight space.
        # We compute pairwise distances in weight space and pick top T neighbors.
        T = 20  # Top 20 neighbors
        dists = np.linalg.norm(weights[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
        neighbors = []
        for i in range(N):
            # Sort indices by distance, exclude self
            idx = np.argsort(dists[i])
            # Skip first (self)
            neighbors.append(idx[1:T+1])
        
        # 4. Main Loop
        # Budget remaining: 200 - 40 = 160
        # We will perform 160 iterations.
        # In each iteration, we update one subproblem.
        # We cycle through subproblems or pick randomly.
        # Picking randomly ensures all get explored.
        
        # Adaptive mutation step
        # Start large, shrink over time
        init_step = 0.5
        final_step = 0.05
        
        for k in range(self.budget - N):
            if time.time() > deadline:
                break
            
            # Select subproblem to update
            i = rng.integers(N)
            
            # Select parent from neighborhood of i
            # With probability 0.5, select from neighbors, else from global population?
            # Standard MOEA/D selects from neighbors.
            neigh = neighbors[i]
            parent_idx = neigh[rng.integers(len(neigh))]
            
            # Crossover: Simulated Binary Crossover (SBX) or just blending
            # Let's use a simple blend (BLX-alpha) or mutation only for simplicity and speed
            # Given low dim (5), mutation is very effective.
            # Let's use Gaussian mutation with adaptive step size.
            
            progress = k / (self.budget - N)
            step = init_step + (final_step - init_step) * progress
            
            # Mutation
            x_parent = X[parent_idx]
            x_child = x_parent + step * (hi - lo) * rng.standard_normal(self.dim)
            x_child = np.clip(x_child, lo, hi)
            
            # Evaluate
            f_child = problem(x_child)
            all_F.append(f_child)
            used += 1
            
            # Update
            # In MOEA/D, we update x_i if the new solution is better in terms of the decomposition function (e.g., Tchebycheff)
            # Decomposition function: g(x, w) = max_j (w_j * |f_j(x) - z_j^*) )
            # We need ideal point z^*. Update z^* as min(F, axis=0).
            
            # Update ideal point
            z_star = np.min(F, axis=0)
            
            # Calculate decomposition value for child
            w_i = weights[i]
            # Tchebycheff distance
            # Note: We minimize objectives, so we want small f.
            # g = max_j (w_j * (f_j - z_j)) is not quite right if z is ideal (min).
            # Standard Tchebycheff: g(x, w, z) = max_j (w_j * |f_j(x) - z_j|)
            # Since we minimize, and z is the lower bound (ideal), f_j >= z_j usually.
            # So |f_j - z_j| = f_j - z_j.
            
            child_val = np.max(w_i * (f_child - z_star))
            current_val = np.max(w_i * (F[i] - z_star))
            
            # If child is better (smaller decomposition value), replace
            if child_val < current_val:
                X[i] = x_child
                F[i] = f_child
                # Update z_star if child improved any objective
                F_new_min = np.minimum(F, f_child)
                # Actually z_star is just min of current F. Since we updated F[i], 
                # we can just recompute min or update incrementally.
                # Recomputing min of 40x3 array is cheap.
                # But we already computed z_star before update. 
                # The condition used the OLD z_star. 
                # This is standard for MOEA/D: use current z*.
                
        return X

# EVOLVE-BLOCK-END

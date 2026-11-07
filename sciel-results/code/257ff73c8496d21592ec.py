"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Decomposition-based Multi-Objective Optimization (MOEA/D) with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate weight vectors for decomposition
        # For 3 objectives, we can use a simplex-lattice or just a grid
        # Let's use a simple grid approach for weight vectors
        n_points_per_dim = 3
        weights = []
        for i in range(n_points_per_dim):
            for j in range(n_points_per_dim):
                k = n_points_per_dim - 1 - i - j
                if k >= 0:
                    weights.append(np.array([i, j, k]) / (n_points_per_dim - 1))
        W = np.array(weights)
        n_subproblems = len(W)
        
        # Initialize population with one solution per subproblem (or random if budget allows)
        # We have 200 budget. Let's start with n_subproblems random points
        # If n_subproblems > budget, we can't even initialize properly, but 3x3x3 grid is small.
        # Actually, for 3 objectives, a grid of 3x3x3 = 19 points is reasonable.
        
        X = np.random.uniform(lo, hi, size=(n_subproblems, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_subproblems
        
        # Check if we can proceed
        if used >= self.budget:
            return X
        
        # MOEA/D loop
        # We will iterate through subproblems, generating one child each per generation
        # Total evaluations = 200. Init = 19. Remaining = 181.
        # Generations = 181 / 19 = 9.5 generations.
        
        # Define neighborhood
        # For simplicity, we can use a fixed neighborhood size or just all other subproblems
        # A common choice is T = 20 neighbors. Here we have few subproblems, so we can use all.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Randomly select a subproblem to solve
            m = np.random.randint(n_subproblems)
            
            # Select parent from neighborhood
            # Since n_subproblems is small, pick a random different subproblem
            parents_idx = [i for i in range(n_subproblems) if i != m]
            p1 = parents_idx[np.random.randint(len(parents_idx))]
            
            # Crossover (Simulated Binary Crossover or simple blend)
            # Let's use a simple convex combination
            alpha = np.random.rand()
            child = alpha * X[p1] + (1 - alpha) * X[m]
            
            # Mutation
            # Gaussian mutation
            sigma = 0.1
            child = child + sigma * (hi - lo) * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            f_child = problem(child)
            used += 1
            
            # Update: 
            # In MOEA/D, we typically replace solutions in the neighborhood 
            # if the child is better according to the subproblem's aggregation function.
            # Aggregation: Tchebycheff or Weighted Sum.
            # Let's use Weighted Sum for simplicity: f_child @ W[m]
            
            # Compare with current solution of subproblem m
            score_child = np.dot(f_child, W[m])
            score_parent = np.dot(F[m], W[m])
            
            if score_child < score_parent: # Minimizing
                X[m] = child
                F[m] = f_child
            
            # Also, consider updating neighbors if child is non-dominated
            # To keep it fast and simple, we'll do a limited neighborhood update
            # Check against all other solutions to see if it dominates them
            # This is O(N) per evaluation, which is fine for N=19.
            
            for i in range(n_subproblems):
                if i == m:
                    continue
                # Check if child dominates F[i]
                # Dominance: f_child <= F[i] and f_child < F[i] (strictly in at least one)
                if np.all(f_child <= F[i]) and np.any(f_child < F[i]):
                    # Child dominates i. Replace i?
                    # In standard MOEA/D, we only replace if the aggregation value is better 
                    # for the subproblem i, or if we are using a specific update strategy.
                    # Let's use the strategy: replace if aggregation value is better for subproblem i
                    score_child_i = np.dot(f_child, W[i])
                    score_i = np.dot(F[i], W[i])
                    if score_child_i < score_i:
                        X[i] = child
                        F[i] = f_child
                # Check if F[i] dominates child? If so, don't replace m with child?
                # We already replaced m based on W[m]. 
                # If F[i] dominates child, it's good to keep F[i].
                # The logic above handles the case where child is better for subproblem i.

        return X
# EVOLVE-BLOCK-END

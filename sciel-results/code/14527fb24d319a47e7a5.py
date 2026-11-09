"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using MOEA/D (Decomposition-based) on ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Population size
        N = 20
        # Generate weight vectors for MOEA/D
        # For 2 objectives, weights are (w1, w2) where w1+w2=1
        weights = np.array([i / (N - 1) for i in range(N)])
        lambda_vectors = np.column_stack([weights, 1 - weights])

        # Initialize population
        X = np.zeros((N, self.dim))
        for i in range(N):
            X[i] = lo + rng.random(self.dim) * span
            
        F = np.array([problem(x) for x in X])
        used = N

        # Neighborhood size
        T = 5
        # Precompute neighbors
        neighbors = [[] for _ in range(N)]
        for i in range(N):
            dists = np.sum(np.abs(lambda_vectors - lambda_vectors[i]), axis=1)
            neighbors[i] = np.argsort(dists)[1:T+1] # Exclude self

        # Tchebycheff scalarization
        # g(x, lambda) = max_j { lambda_j * |f_j(x) - z_j^*| }
        # z^* is the ideal point (minimum of each objective)
        
        # Adaptive parameters
        base_mutation_eta = 20.0
        jump_prob = 0.1

        while used < self.budget:
            # Update ideal point
            z_star = np.min(F, axis=0)
            
            # Calculate current fitness for each individual
            # g_i = max_j { lambda^i_j * |F_i_j - z_star_j| }
            # F shape: (N, n_obj)
            # lambda_vectors shape: (N, n_obj)
            # We need to compute for each i, max over j of lambda_vectors[i,j] * abs(F[i,j] - z_star[j])
            
            # Vectorized calculation
            # abs_diff shape (N, n_obj)
            abs_diff = np.abs(F - z_star)
            # weighted shape (N, n_obj)
            weighted = lambda_vectors * abs_diff
            # fitness shape (N,)
            fitness = np.max(weighted, axis=1)

            # For each individual, try to generate an offspring
            for i in range(N):
                if used >= self.budget:
                    break
                    
                # Determine parent: either i or a neighbor
                # Standard MOEA/D: mutate x_i or x_j from neighbors
                # Let's mutate x_i with some prob, or a neighbor
                if rng.random() < 0.5:
                    parent_idx = i
                else:
                    parent_idx = neighbors[i][rng.integers(len(neighbors[i]))]
                
                x_parent = X[parent_idx].copy()

                # Jumping mechanism to escape local optima (ZDT4 specific)
                if rng.random() < jump_prob:
                    x_child = lo + rng.random(self.dim) * span
                else:
                    x_child = x_parent.copy()
                    # Polynomial mutation
                    eta_m = base_mutation_eta
                    for d in range(self.dim):
                        if rng.random() < 0.5: # Mutation rate 0.5
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            x_child[d] += delta_q * span[d]
                    x_child = np.clip(x_child, lo, hi)

                f_child = problem(x_child)
                used += 1

                # Calculate fitness of child for i and neighbors
                # g_child_i = max_j { lambda^i_j * |f_child_j - z_star_j| }
                abs_diff_child = np.abs(f_child - z_star)
                
                # Update individual i
                g_child_i = np.max(lambda_vectors[i] * abs_diff_child)
                if g_child_i < fitness[i]:
                    X[i] = x_child
                    F[i] = f_child
                    fitness[i] = g_child_i
                    # Update z_star
                    z_star = np.min(F, axis=0)
                    
                # Update neighbors
                for j in neighbors[i]:
                    g_child_j = np.max(lambda_vectors[j] * abs_diff_child)
                    if g_child_j < fitness[j]:
                        X[j] = x_child
                        F[j] = f_child
                        fitness[j] = g_child_j
                        # Update z_star
                        z_star = np.min(F, axis=0)

        return X
# EVOLVE-BLOCK-END

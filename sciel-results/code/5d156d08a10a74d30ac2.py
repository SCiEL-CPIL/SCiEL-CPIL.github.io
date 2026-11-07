"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_obj = self.n_obj
        dim = self.dim

        # Population size (number of subproblems)
        N = 20
        
        # Generate weight vectors
        # Random weights normalized to sum to 1
        raw_weights = rng.random((N, n_obj))
        weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        
        # Find neighbors for each weight vector (5 nearest)
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Sort indices by distance, exclude self
            sorted_idx = np.argsort(dists)
            # Take first 5, skipping self if it's at 0 (it will be)
            nb = [idx for idx in sorted_idx if idx != i][:5]
            neighbors.append(nb)

        # Initial population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Ideal point z*
        z_star = F.min(axis=0)

        # Best solution for each subproblem
        best_X = X.copy()
        best_F = F.copy()

        # Helper: Tchebycheff scalarizing function
        def tchebycheff(f, w, z):
            # f: (n_obj,), w: (n_obj,), z: (n_obj,)
            # Standard: max_i w_i * |f_i - z_i|
            return np.max(w * np.abs(f - z))

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if abs(x1[i] - x2[i]) < 1e-14:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Polynomial mutation
        def polynomial_mutation(x, eta=20.0):
            x = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    lo_i = lo[i]
                    hi_i = hi[i]
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x[i] += delta_q * (hi_i - lo_i)
                    x[i] = np.clip(x[i], lo_i, hi_i)
            return x

        # Main loop
        while used < self.budget:
            # Select a subproblem to update
            i = rng.integers(0, N)
            
            # Select 2 parents from neighbors of i
            # Include i itself in candidate pool
            candidate_pool = [i] + neighbors[i]
            p1_idx = candidate_pool[rng.integers(0, len(candidate_pool))]
            p2_idx = candidate_pool[rng.integers(0, len(candidate_pool))]
            
            # Crossover and Mutation
            c1, c2 = sbx_crossover(best_X[p1_idx], best_X[p2_idx])
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            # Evaluate c1
            f1 = problem(c1)
            used += 1
            
            # Update ideal point
            z_star = np.minimum(z_star, f1)
            
            # Update subproblems in neighborhood of i
            # Check against current best for each neighbor
            update_set = [i] + neighbors[i]
            for j in update_set:
                # Compare c1 with best_X[j] using weight w_j
                val_c1 = tchebycheff(f1, weights[j], z_star)
                val_best = tchebycheff(best_F[j], weights[j], z_star)
                
                if val_c1 < val_best:
                    best_X[j] = c1
                    best_F[j] = f1
            
            # If budget allows, evaluate c2 and update similarly
            if used < self.budget:
                f2 = problem(c2)
                used += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, f2)
                
                # Update subproblems in neighborhood of i
                for j in update_set:
                    val_c2 = tchebycheff(f2, weights[j], z_star)
                    val_best = tchebycheff(best_F[j], weights[j], z_star)
                    
                    if val_c2 < val_best:
                        best_X[j] = c2
                        best_F[j] = f2

        return best_X
# EVOLVE-BLOCK-END

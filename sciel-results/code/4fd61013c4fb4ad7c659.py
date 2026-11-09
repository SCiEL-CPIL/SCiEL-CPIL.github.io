"""Rewritten optimizer for ZDT1 using MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D style multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.budget
        
        # Population size for MOEA/D
        # With 2 objectives, a moderate population size allows good coverage.
        # Budget 200. Let's use N=50.
        # 50 initial evals. 150 iterations.
        N = 50
        
        # Generate weight vectors
        # For 2 objectives, weights are (lambda_1, lambda_2)
        # lambda_1 + lambda_2 = 1
        # We want to cover the front.
        # Uniform distribution on the simplex.
        weights = np.zeros((N, self.n_obj))
        for i in range(N):
            # Map i to [0, 1]
            # Avoid 0 and 1 exactly to ensure both objectives are considered?
            # Actually, endpoints are good for extremes.
            # lambda_1 = i / (N - 1)
            # lambda_2 = 1 - lambda_1
            # To avoid division by zero issues or degenerate weights, we can use:
            # But for 2D, simple linear spacing is fine.
            # Let's include endpoints.
            if N > 1:
                w1 = i / (N - 1)
            else:
                w1 = 0.5
            weights[i, 0] = w1
            weights[i, 1] = 1.0 - w1
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = N
        
        # Ideal point z* (min values seen so far)
        z_star = np.min(F, axis=0)
        
        # Calculate current Tchebycheff values
        def tchebycheff(f, w, z):
            return np.max(w * np.abs(f - z))
        
        G = np.array([tchebycheff(F[i], weights[i], z_star) for i in range(N)])
        
        # Define neighborhood
        # For N points in a line (1D simplex), neighbors are adjacent points.
        # Let's define T neighbors on each side.
        T = 5 # Neighborhood size
        neighbors = []
        for i in range(N):
            nbrs = []
            for j in range(N):
                if i != j:
                    # Distance in weight space
                    dist = np.linalg.norm(weights[i] - weights[j])
                    nbrs.append((dist, j))
            nbrs.sort(key=lambda x: x[0])
            neighbors.append([idx for dist, idx in nbrs[:T]])
        
        # Helper: SBX Crossover
        def sbx(x1, x2, rng, eta=20.0):
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x2[i] - x1[i]) > 1e-15:
                        if rng.random() <= 1.0 / (eta + 1.0):
                            beta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * rng.random())) ** (1.0 / (eta + 1.0))
                        c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                        c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, rng, eta=20.0, prob=1.0 / self.dim):
            for i in range(self.dim):
                if rng.random() < prob:
                    dx = x[i] - lo[i]
                    dxh = hi[i] - x[i]
                    rand = rng.random()
                    if rand < 0.5:
                        xy = (2.0 * rand) ** (1.0 / (eta + 1.0))
                        delta = (1.0 - xy) ** (eta + 1.0)
                        x[i] -= delta * dx
                    else:
                        xy = (2.0 * (1.0 - rand)) ** (1.0 / (eta + 1.0))
                        delta = (1.0 - xy) ** (eta + 1.0)
                        x[i] += delta * dxh
            return np.clip(x, lo, hi)

        # Main Loop
        while used < n:
            # Select a subproblem to update
            # Random selection is standard
            i = rng.integers(N)
            
            # Select parents from neighbors of i
            nbrs = neighbors[i]
            # Pick 2 parents from neighbors (or from whole pop if neighbors are small, but T=5 is enough)
            # If i is not in nbrs, we should probably include i? 
            # Usually, we pick from the neighborhood which includes i or not.
            # Let's pick 2 distinct parents from the list of neighbors + i
            candidate_pool = list(set(nbrs + [i]))
            p1_idx = candidate_pool[rng.integers(len(candidate_pool))]
            p2_idx = candidate_pool[rng.integers(len(candidate_pool))]
            
            # Crossover
            c1, c2 = sbx(X[p1_idx], X[p2_idx], rng)
            
            # Mutation
            c1 = polynomial_mutation(c1, rng)
            c2 = polynomial_mutation(c2, rng)
            
            # Evaluate c1
            f1 = problem(c1)
            used += 1
            
            # Update logic for c1
            # 1. Check if c1 improves the ideal point
            new_z_star = z_star.copy()
            for k in range(self.n_obj):
                if f1[k] < z_star[k]:
                    new_z_star[k] = f1[k]
            
            # 2. Check if c1 is better than current solution of subproblem i
            # Note: If z_star changed, we should recalculate G for all? 
            # In standard MOEA/D, z* is global. If it changes, G values for all subproblems change.
            # This is expensive. 
            # Alternative: Use the current z_star for comparison, or update z_star and only compare locally?
            # Usually, z* is updated, and the scalarizing function uses the new z*.
            # However, recalculating G for all N points every time z* changes is O(N * n_obj).
            # With N=50, this is cheap.
            
            if np.any(new_z_star < z_star):
                z_star = new_z_star
                # Recalculate G for all
                G = np.array([tchebycheff(F[k], weights[k], z_star) for k in range(N)])
            
            # Compare c1 with X[i]
            g_c1 = tchebycheff(f1, weights[i], z_star)
            if g_c1 < G[i]:
                X[i] = c1
                F[i] = f1
                G[i] = g_c1
                
                # Update neighbors
                for j in nbrs:
                    if j == i: continue
                    g_c1_j = tchebycheff(f1, weights[j], z_star)
                    if g_c1_j < G[j]:
                        X[j] = c1
                        F[j] = f1
                        G[j] = g_c1_j
                        
            # If we have budget left, evaluate c2
            if used < n:
                f2 = problem(c2)
                used += 1
                
                # Update logic for c2 (same as c1)
                new_z_star = z_star.copy()
                for k in range(self.n_obj):
                    if f2[k] < z_star[k]:
                        new_z_star[k] = f2[k]
                        
                if np.any(new_z_star < z_star):
                    z_star = new_z_star
                    G = np.array([tchebycheff(F[k], weights[k], z_star) for k in range(N)])
                
                g_c2 = tchebycheff(f2, weights[i], z_star)
                if g_c2 < G[i]:
                    X[i] = c2
                    F[i] = f2
                    G[i] = g_c2
                    
                    # Update neighbors
                    for j in nbrs:
                        if j == i: continue
                        g_c2_j = tchebycheff(f2, weights[j], z_star)
                        if g_c2_j < G[j]:
                            X[j] = c2
                            F[j] = f2
                            G[j] = g_c2_j

        return X
# EVOLVE-BLOCK-END

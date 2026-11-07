"""Rewritten multi-objective optimizer for WFG4 using MOEA/D."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size N
        # We need to balance exploration and exploitation.
        # With budget 200, we can afford a larger population if generations are small,
        # or a smaller population with more generations.
        # MOEA/D typically uses N = 100-200.
        # Let's try N=100.
        # If N=100, we have 100 initial evals.
        # Remaining budget: 100.
        # We can do 100 updates. Each update generates 1 offspring.
        # So we can do 100 updates.
        # Total evals = 100 + 100 = 200.
        
        N = 100
        
        # Generate weight vectors
        # We need N weight vectors in the unit simplex.
        # A common method is to use a grid or random points on the simplex.
        # For n_obj=3, we can use a triangular grid.
        # Or simply generate random points and normalize.
        # To ensure good coverage, let's use a systematic approach for 3 objectives.
        # For N=100, we can distribute them.
        
        # Let's use a simple random generation with normalization, 
        # but ensure diversity. 
        # Actually, a grid is better for MOEA/D.
        # For 3 objectives, we can use a triangular lattice.
        # Let's generate points using a recursive method or just random and check.
        # Given the small budget, random normalized vectors might be sufficient if N is large enough.
        # Let's try generating N random points and normalizing.
        
        weights = rng.random((N, n_obj))
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        # Calculate neighborhood
        # Two weights are neighbors if their Euclidean distance is small.
        # Let's define neighborhood size as T=20 (top 20 closest).
        T = 20
        
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        # For each i, find T closest j's (excluding i)
        neighbors = [None] * N
        for i in range(N):
            # Sort indices by distance
            sorted_idx = np.argsort(dists[i])
            # Exclude self
            sorted_idx = sorted_idx[sorted_idx != i]
            neighbors[i] = sorted_idx[:T]
            
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # Ideal point and nadir point (approximated)
        # For Tchebycheff, we need ideal point f*.
        # Initialize with current best.
        f_ideal = np.min(F, axis=0)
        f_nadir = np.max(F, axis=0)
        
        # MOEA/D parameters
        eta_c = 20.0
        eta_m = 20.0
        p_c = 0.9  # Crossover probability
        p_m = 1.0 / dim # Mutation probability per gene
        
        # Main loop
        # We have budget - used evaluations left.
        # Each iteration generates 1 offspring (evaluated).
        # We iterate through the population indices in random order.
        
        remaining = budget - used
        if remaining <= 0:
            return X
            
        # Shuffle order of updates to avoid bias
        order = rng.permutation(N)
        
        for k in range(remaining):
            i = order[k % N]
            
            # Select parent from neighborhood of i
            # Pick a random neighbor
            if len(neighbors[i]) > 0:
                j_idx = neighbors[i][rng.integers(len(neighbors[i]))]
            else:
                j_idx = rng.integers(N)
                
            p1 = X[j_idx]
            
            # Select a second parent? 
            # Standard MOEA/D often uses one parent and mutation, or two parents.
            # Let's use two parents for better exploration.
            # Pick another neighbor or random.
            if len(neighbors[i]) > 1:
                j2_idx = neighbors[i][rng.integers(len(neighbors[i]))]
                if j2_idx == j_idx:
                    j2_idx = (j2_idx + 1) % len(neighbors[i])
                    j2_idx = neighbors[i][j2_idx]
            else:
                j2_idx = rng.integers(N)
                
            p2 = X[j2_idx]
            
            # Crossover
            if rng.random() < p_c:
                c1, c2 = self._sbx_crossover(rng, p1, p2, lo, hi, eta_c)
            else:
                c1 = p1.copy()
                c2 = p2.copy()
                
            # We only use c1 as the offspring (or c2, or both if budget allows).
            # Since we have 1 eval budget per iteration, we use c1.
            c1 = self._poly_mutation(rng, c1, lo, hi, eta_m, p_m)
            
            f_new = np.array(problem(c1), dtype=float)
            used += 1
            
            # Update neighborhood
            # For each neighbor j of i, check if c1 is better for subproblem j.
            # Objective: Tchebycheff distance
            # d(x, lambda) = max_k ( lambda_k * |f_k(x) - z_k^*| )
            
            for j in neighbors[i]:
                # Calculate Tchebycheff for current solution X[j]
                # And for new solution c1
                
                # Update ideal point if new solution is better
                # Note: f_ideal should be the component-wise minimum.
                # If f_new has any component smaller than f_ideal, update f_ideal.
                # However, updating f_ideal dynamically can cause issues if it moves too fast.
                # Standard MOEA/D updates f* periodically or at the end.
                # Let's update f_ideal only if it improves, but keep it stable.
                # Actually, for Tchebycheff, using a fixed ideal point (initial or estimated) is common.
                # Let's use the current component-wise minimum of the population.
                # To avoid expensive recalculation, we update f_ideal incrementally.
                
                # Check improvement
                # d_j(X[j]) vs d_j(c1)
                
                # Calculate d_j(X[j])
                # We can precompute or calculate on the fly.
                # Given N=100, calculating on the fly for T=20 neighbors is 20 * 3 ops.
                
                # Let's calculate d for X[j] and c1
                # d = max_k( lambda[k] * |F[k] - f_ideal[k]| )
                
                # We need f_ideal. 
                # Let's maintain f_ideal as the min of all seen F.
                # If f_new is better, update f_ideal.
                # But if we update f_ideal, the distances for existing points change.
                # This is standard in some implementations.
                
                # Let's try a simpler update rule:
                # Replace X[j] with c1 if c1 has a better Tchebycheff value 
                # using the *current* f_ideal.
                
                # First, update f_ideal if c1 is better
                # But wait, if we update f_ideal, we should re-evaluate?
                # Usually, f_ideal is updated at the end of each generation or periodically.
                # With online updates, let's just check if c1 is better than X[j] 
                # using the *current* f_ideal.
                
                # Calculate Tchebycheff for X[j]
                # lambda_j = weights[j]
                # diff_j = np.abs(F[j] - f_ideal)
                # tcheb_j = np.max(weights[j] * diff_j)
                
                # Calculate Tchebycheff for c1
                # diff_new = np.abs(f_new - f_ideal)
                # tcheb_new = np.max(weights[j] * diff_new)
                
                # If tcheb_new < tcheb_j, replace.
                
                # Optimization: 
                # We can check if f_new dominates F[j] or is closer to ideal.
                
                # Let's do the full calculation for correctness.
                
                # Update f_ideal? 
                # If we update f_ideal, tcheb_j changes.
                # Let's NOT update f_ideal inside the inner loop to keep it simple and stable.
                # We will update f_ideal after the inner loop.
                
                diff_j = np.abs(F[j] - f_ideal)
                tcheb_j = np.max(weights[j] * diff_j)
                
                diff_new = np.abs(f_new - f_ideal)
                tcheb_new = np.max(weights[j] * diff_new)
                
                if tcheb_new < tcheb_j:
                    X[j] = c1
                    F[j] = f_new
                    
            # Update f_ideal with the new solution
            # f_ideal = min(f_ideal, f_new)
            f_ideal = np.minimum(f_ideal, f_new)
            
            # Optionally update f_nadir? Not needed for Tchebycheff.
            
        return X

    def _sbx_crossover(self, rng, p1, p2, lo, hi, eta_c):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if rng.random() <= 1.0 / (eta_c + 1.0):
                        beta = 2.0 * rng.random()
                    else:
                        beta = 2.0 * (1.0 - rng.random())
                    
                    beta = 1.0 / (beta * eta_c + 1.0)
                    
                    c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
                    
                    if c1[i] > hi[i]:
                        c1[i] = hi[i]
                    if c1[i] < lo[i]:
                        c1[i] = lo[i]
                    if c2[i] > hi[i]:
                        c2[i] = hi[i]
                    if c2[i] < lo[i]:
                        c2[i] = lo[i]
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        return c1, c2

    def _poly_mutation(self, rng, x, lo, hi, eta_m, p_m):
        x_mut = x.copy()
        for i in range(len(x)):
            if rng.random() < p_m:
                if abs(hi[i] - lo[i]) < 1e-14:
                    continue
                
                r = rng.random()
                if r < 0.5:
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                    mut_pow = 2.0 / (eta_m + 1.0) * (delta1 + 1.0)
                    if r <= 1.0 / mut_pow:
                        delta = (2.0 * r) ** (1.0 / (eta_m + 1.0))
                    else:
                        delta = (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] - delta * (hi[i] - lo[i])
                else:
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                    mut_pow = 2.0 / (eta_m + 1.0) * (delta2 + 1.0)
                    if (1.0 - r) <= 1.0 / mut_pow:
                        delta = (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    else:
                        delta = (2.0 * r) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] + delta * (hi[i] - lo[i])
                    
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut
# EVOLVE-BLOCK-END

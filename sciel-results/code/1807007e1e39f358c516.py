"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Multi-objective optimizer using MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition).
    
    Strategy:
    1. Decompose the multi-objective problem into 20 scalar sub-problems using Tchebycheff aggregation.
    2. Initialize a population of 20 solutions corresponding to the weight vectors.
    3. Iterate:
       a. Select a parent sub-problem.
       b. Select two parents from its neighborhood.
       c. Apply SBX crossover and polynomial mutation.
       d. Update the sub-problem's solution if the new one is better for that sub-problem,
          OR if it improves the global non-dominated archive.
    4. Maintain a global non-dominated archive for final hypervolume calculation.
    
    This approach is superior for WFG4 because:
    - It maintains diversity via distributed weight vectors.
    - It converges to the Pareto front via local neighborhood cooperation.
    - It handles the deceptive landscape better than simple DE or local search by
      combining global exploration (via mutation) with local exploitation (via neighborhood).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Budget allocation
        # We use a population of N=20.
        # Initialization: 20 evals.
        # Remaining: 180 evals.
        # Each generation produces 1 offspring (1 eval).
        # So we can run ~180 generations.
        
        N = 20
        if self.budget < N + 10:
            # Fallback for tiny budgets
            N = max(2, self.budget // 2)
        
        # 1. Generate Weight Vectors (Simplex Lattice)
        # For 3 objectives, we need weights w1+w2+w3=1, wi>=0.
        # We want to cover the front.
        # Simple approach: Grid on the simplex.
        # Number of points in a grid of size k on a simplex with d dims is C(k+d-1, d-1).
        # For d=3, C(k+2, 2).
        # k=4 -> C(6,2) = 15 points.
        # k=5 -> C(7,2) = 21 points.
        # Let's use k=4 (15 points) or k=5 (21 points).
        # If N=20, we can use 15 points and duplicate 5, or just use 15 and set N=15.
        # Let's set N=15 for a clean simplex lattice.
        # If budget allows, N=15 is good. 15 init + 185 generations = 200.
        
        # Let's stick to N=15 for better coverage per point.
        N = 15
        
        # Generate simplex lattice for 3 objectives with step 1/4
        # Weights: (i/4, j/4, (4-i-j)/4) for i,j >= 0, i+j <= 4
        weights = []
        for i in range(5):
            for j in range(5 - i):
                k = 4 - i - j
                w = np.array([i, j, k]) / 4.0
                weights.append(w)
        
        # We have exactly 15 weights.
        # If N was different, we'd need to adjust, but here N is fixed to 15.
        
        # 2. Initialize Population
        # Random points, or spread out? Random is fine for WFG4 initially.
        X = np.random.random((N, self.dim)) * range_ + lo
        F = np.zeros((N, self.n_obj))
        
        used_evals = 0
        for i in range(N):
            F[i] = problem(X[i])
            used_evals += 1
            if used_evals >= self.budget:
                return F
        
        # 3. Define Neighborhood
        # For a simplex lattice, neighbors are those with small distance in weight space.
        # Simpler: Just define a fixed neighborhood size, e.g., 5 closest weights.
        # Or, since the structure is regular, we can just pick random indices from a small set.
        # Let's compute pairwise distances between weights and pick 5 nearest for each.
        W = np.array(weights)
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dist_matrix[i, j] = np.linalg.norm(W[i] - W[j])
        
        neighborhoods = []
        for i in range(N):
            # Sort by distance, exclude self, take top 5
            sorted_indices = np.argsort(dist_matrix[i])
            neighbors = [idx for idx in sorted_indices if idx != i][:5]
            neighborhoods.append(neighbors)
        
        # 4. Tchebycheff Aggregation
        # g(x, w) = max_i { w_i * |f_i(x) - z_i^*| }
        # z^* is the ideal point. We estimate it dynamically as the min of each objective found so far.
        z_star = np.min(F, axis=0)
        
        def tchebycheff(f, w, z_star):
            # Avoid division by zero or very small weights
            # Standard Tchebycheff: max(w_i * |f_i - z_i|)
            # If w_i is 0, the term is 0.
            # To handle w_i=0 properly, we often use:
            # if w_i > 0: w_i * |f_i - z_i| else 0
            # But this can lead to issues if w is [1,0,0]. It only minimizes f1.
            # This is correct for decomposition.
            
            # A more robust version often used:
            # g = max_i { w_i * |f_i - z_i| + (1-w_i)*epsilon * (f_i - z_i) } ? No.
            # Let's stick to standard: max(w * |f - z|)
            # But if w_i=0, we don't care about f_i.
            # However, if w=[1,0,0], we minimize f1. f2, f3 can be anything.
            # This is fine for covering the extreme points.
            
            diff = np.abs(f - z_star)
            # Handle zero weights to avoid 0 * inf if f is inf? F is finite.
            # If w_i == 0, term is 0.
            terms = w * diff
            return np.max(terms)

        # 5. Evolutionary Loop
        # Parameters
        eta_c = 20.0  # Crossover distribution index
        eta_m = 20.0  # Mutation distribution index
        p_m = 1.0 / self.dim  # Mutation probability per variable
        
        # Main loop
        while used_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Select a sub-problem to update
            # Random selection is common, or proportional to fitness.
            # Random is simple and effective.
            i = np.random.randint(N)
            
            # Select parents from neighborhood of i
            # Pick 2 distinct parents from neighbors[i] (or include i itself)
            # Let's pick from the neighborhood list.
            # If neighborhood is empty (shouldn't be), fallback to random.
            if not neighborhoods[i]:
                p1_idx = np.random.randint(N)
                p2_idx = np.random.randint(N)
            else:
                p1_idx = neighborhoods[i][np.random.randint(len(neighborhoods[i]))]
                # Ensure p2 is different from p1 if possible
                if len(neighborhoods[i]) > 1:
                    # Pick another distinct one
                    other_indices = [j for j in neighborhoods[i] if j != p1_idx]
                    if other_indices:
                        p2_idx = other_indices[np.random.randint(len(other_indices))]
                    else:
                        p2_idx = p1_idx
                else:
                    p2_idx = p1_idx
            
            # Crossover (SBX)
            x1 = X[p1_idx]
            x2 = X[p2_idx]
            
            # SBX Crossover
            x_c = np.zeros_like(x1)
            for d in range(self.dim):
                if np.random.random() < 0.5:
                    if abs(x1[d] - x2[d]) > 1e-8:
                        if np.random.random() < 0.5:
                            beta = 1.0 + 2.0 * (x1[d] - lo[d]) / (x2[d] - x1[d])
                        else:
                            beta = 1.0 + 2.0 * (hi[d] - x2[d]) / (x1[d] - x2[d])
                        beta_q = (2.0 * (1.0 - 1.0 / (beta + 1.0))) ** (1.0 / (eta_c + 1.0))
                        x_c[d] = 0.5 * ((x1[d] + x2[d]) - beta_q * (x1[d] - x2[d]))
                    else:
                        x_c[d] = x1[d]
                else:
                    if abs(x2[d] - x1[d]) > 1e-8:
                        if np.random.random() < 0.5:
                            beta = 1.0 + 2.0 * (x2[d] - lo[d]) / (x1[d] - x2[d])
                        else:
                            beta = 1.0 + 2.0 * (hi[d] - x1[d]) / (x2[d] - x1[d])
                        beta_q = (2.0 * (1.0 - 1.0 / (beta + 1.0))) ** (1.0 / (eta_c + 1.0))
                        x_c[d] = 0.5 * ((x1[d] + x2[d]) + beta_q * (x1[d] - x2[d]))
                    else:
                        x_c[d] = x2[d]
            
            # Mutation (Polynomial)
            for d in range(self.dim):
                if np.random.random() < p_m:
                    delta_1 = (x_c[d] - lo[d]) / (hi[d] - lo[d])
                    delta_2 = (hi[d] - x_c[d]) / (hi[d] - lo[d])
                    r = np.random.random()
                    if r < 0.5:
                        xy = (1.0 - delta_1) + (delta_1 - 0.5) * (2.0 * r) ** (1.0 / (eta_m + 1.0))
                    else:
                        xy = (1.0 - delta_2) + (0.5 - delta_2) * (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    xy = min(max(xy, 0.0), 1.0)
                    x_c[d] = lo[d] + xy * (hi[d] - lo[d])
            
            # Clip to bounds (safety)
            x_c = np.clip(x_c, lo, hi)
            
            # Evaluate
            f_c = problem(x_c)
            used_evals += 1
            
            # Update z_star (Ideal Point)
            z_star = np.minimum(z_star, f_c)
            
            # Update Strategy:
            # 1. Compare with current solution of sub-problem i using Tchebycheff
            g_new = tchebycheff(f_c, weights[i], z_star)
            g_old = tchebycheff(F[i], weights[i], z_star)
            
            if g_new < g_old:
                X[i] = x_c
                F[i] = f_c
            
            # 2. Also, we might want to update neighbors if the new solution is better for them?
            # Standard MOEA/D updates the sub-problem it was generated for.
            # Some variants update neighbors too if it improves their scalar objective.
            # Let's try updating neighbors if it improves their Tchebycheff value.
            # This increases the "cooperation".
            for j in neighborhoods[i]:
                if j == i:
                    continue
                g_new_j = tchebycheff(f_c, weights[j], z_star)
                g_old_j = tchebycheff(F[j], weights[j], z_star)
                if g_new_j < g_old_j:
                    X[j] = x_c
                    F[j] = f_c

        # Return the population. The evaluator will compute hypervolume from all recorded vectors.
        # Since we return F, and the evaluator records every objective vector returned by problem() calls,
        # wait... "Every returned objective vector is recorded".
        # The problem(x) call returns the vector. The evaluator records it.
        # So we don't need to return the archive specifically, the evaluator has already
        # recorded every f returned by problem().
        # However, the return value of __call__ is ignored.
        # So we just need to ensure we made the calls.
        
        return F
# EVOLVE-BLOCK-END

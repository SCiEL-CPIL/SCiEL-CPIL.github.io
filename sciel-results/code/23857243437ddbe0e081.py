# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-style decomposition-based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Population size
        # For WFG9 with 3 objectives, a population of 100-200 is often good.
        # With budget 200, we can afford a smaller population and more iterations or
        # a larger population and fewer iterations.
        # Let's try N=100, so we have 100 initial + 100 offspring = 200 total.
        # Or N=50, 4 generations.
        # MOEA/D typically benefits from larger populations.
        # Let's use N=100. This uses exactly the budget if we do 1 generation of offspring.
        # Wait, if N=100, init=100, offspring=100. Total 200.
        # This is a single generation MOEA/D, which is basically a parallel hill climbing.
        # To get evolutionary pressure, we need multiple generations.
        # Let's try N=50. Init 50.
        # Gen 1: 50 offspring. Total 100.
        # Gen 2: 50 offspring. Total 150.
        # Gen 3: 50 offspring. Total 200.
        # 3 generations might be too few for MOEA/D to converge well.
        
        # Alternative: Use a smaller population for more generations.
        # N=20. Init 20.
        # 9 generations of 20 offspring. Total 200.
        # 9 generations is better.
        
        pop_size = 20

        # Generate reference points on simplex
        # For 3 objectives, we can generate points on the 2D simplex.
        # Number of points = pop_size
        ref_points = self._generate_reference_points(n_obj, pop_size)

        # Initialize population
        # Random initialization
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])

        # Track evaluations used
        used = pop_size

        # Ideal point (lower bound)
        # Initialize to a large value or update dynamically
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)
        
        # Normalize objectives for Tchebycheff
        # We need to update ideal and nadir as we go.
        # Usually, we keep the best known ideal point.
        
        # Neighborhood
        # For MOEA/D, each subproblem has a neighborhood of other subproblems.
        # We need to define which indices are neighbors.
        # Simple approach: Sort reference points and take k nearest neighbors.
        # Or just use random neighbors.
        # Let's use a fixed neighborhood size T=10 (half the population).
        T = 10
        
        # Precompute neighbors
        # Calculate pairwise distances between reference points
        ref_diff = ref_points[:, np.newaxis, :] - ref_points[np.newaxis, :, :]
        ref_dist = np.sum(ref_diff ** 2, axis=2)
        
        # For each i, find T nearest neighbors
        neighbors = []
        for i in range(pop_size):
            # Sort indices by distance to i
            dists = ref_dist[i, :]
            # Exclude i itself
            dists[i] = np.inf
            sorted_idx = np.argsort(dists)
            neighbors.append(sorted_idx[:T])

        # Main loop
        while used + pop_size <= budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Update ideal point
            # ideal = np.min(F, axis=0)
            # nadir = np.max(F, axis=0)
            # To avoid division by zero or small ranges, add epsilon
            eps = 1e-6
            range_obj = nadir - ideal
            range_obj[range_obj < eps] = eps

            # Generate offspring
            # For each subproblem i, select parents from neighbors of i
            for i in range(pop_size):
                if used >= budget:
                    break
                    
                # Select parents from neighbors of i
                # Randomly pick 2 from neighbors[i]
                n_idx = neighbors[i]
                p1_idx = n_idx[np.random.randint(len(n_idx))]
                p2_idx = n_idx[np.random.randint(len(n_idx))]
                
                # Ensure p1 != p2
                if p1_idx == p2_idx:
                    # Pick another
                    other_idx = [j for j in n_idx if j != p1_idx]
                    if other_idx:
                        p2_idx = other_idx[np.random.randint(len(other_idx))]
                    else:
                        p2_idx = p1_idx

                # SBX Crossover
                child = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi, 20.0)

                # Mutation
                # Polynomial mutation or Gaussian
                # Gaussian mutation with sigma = 0.1 * span
                mut_mask = np.random.random(dim) < 0.5
                if np.any(mut_mask):
                    child[mut_mask] += 0.1 * span[mut_mask] * np.random.standard_normal(np.sum(mut_mask))
                    child = np.clip(child, lo, hi)

                offspring_F = problem(child)
                used += 1

                # Update population
                # Check if child is better than current X[i] for subproblem i
                # And potentially update neighbors if better for them too
                
                # Calculate Tchebycheff value for child for subproblem i
                # v_i(x) = max_j (lambda_ij * |f_j(x) - z_j^i|)
                # Actually, standard Tchebycheff uses normalized objectives:
                # g(x, lambda) = max_j (lambda_j * (f_j(x) - z_j^*) / (z_j^max - z_j^*))
                
                # Let's use the standard MOEA/D update rule:
                # A new solution x replaces x_i if g(x, lambda_i) < g(x_i, lambda_i)
                # Or if it improves the fitness of any neighbor.
                
                # Calculate g for child and current X[i]
                # Normalize child and X[i]
                # f_norm = (f - ideal) / (nadir - ideal)
                
                child_norm = (offspring_F - ideal) / range_obj
                curr_norm = (F[i] - ideal) / range_obj
                
                # Tchebycheff distance
                # lambda_i is ref_points[i]
                lambda_i = ref_points[i]
                
                # g_child = max(lambda_i * child_norm)
                g_child = np.max(lambda_i * child_norm)
                g_curr = np.max(lambda_i * curr_norm)
                
                # If child is better for i, update i
                if g_child < g_curr:
                    X[i] = child
                    F[i] = offspring_F
                    # Update ideal and nadir
                    ideal = np.minimum(ideal, offspring_F)
                    nadir = np.maximum(nadir, offspring_F)
                    
                # Check neighbors
                # If child is better for neighbor k, update k
                for k in neighbors[i]:
                    if k == i:
                        continue
                    lambda_k = ref_points[k]
                    curr_k_norm = (F[k] - ideal) / range_obj
                    g_curr_k = np.max(lambda_k * curr_k_norm)
                    g_child_k = np.max(lambda_k * child_norm)
                    
                    if g_child_k < g_curr_k:
                        X[k] = child
                        F[k] = offspring_F
                        # Note: We don't update ideal/nadir here again to save time, 
                        # or we can. It's cheap.
                        # ideal = np.minimum(ideal, offspring_F)
                        # nadir = np.maximum(nadir, offspring_F)

        return X

    def _generate_reference_points(self, n_obj, num_points):
        """Generate reference points on the simplex for n_obj objectives."""
        if n_obj == 3:
            # Generate points on 2D simplex
            # Use Fibonacci sphere or similar distribution projected to simplex
            # Or simply random points normalized
            # For deterministic behavior, we could use a fixed grid, but random is fine if seeded.
            # Let's use a stratified sampling or just random normalization.
            
            # Random points in [0,1]^3, then normalize to sum=1
            points = np.random.random((num_points, n_obj))
            points = points / points.sum(axis=1, keepdims=True)
            return points
        else:
            # General case
            points = np.random.random((num_points, n_obj))
            points = points / points.sum(axis=1, keepdims=True)
            return points

    def _sbx_crossover(self, x1, x2, lo, hi, eta):
        """Simulated Binary Crossover."""
        child = np.zeros_like(x1)
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) < 1e-14:
                    child[i] = x1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                child[i] = np.clip(child[i], lo[i], hi[i])
            else:
                if abs(x1[i] - x2[i]) < 1e-14:
                    child[i] = x1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
                child[i] = np.clip(child[i], lo[i], hi[i])
        return child
# EVOLVE-BLOCK-END

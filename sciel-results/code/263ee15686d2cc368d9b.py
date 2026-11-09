"""Multi-Objective Optimizer for RE21 using MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization
        # Use a significant portion of the budget for initial exploration.
        # For dim=4, n_obj=2, a population of 40 is reasonable for MOEA/D.
        pop_size = 40
        if self.budget < 50:
            pop_size = max(10, self.budget // 2)
            
        X = self._latin_hypercube(pop_size, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        
        # Track total evaluations
        n_evals = pop_size
        
        # 2. MOEA/D Setup
        # Generate weight vectors on the simplex for 2 objectives.
        # We use a grid of weights to ensure coverage.
        # For 2 objectives, weights are (w1, 1-w1).
        # We create pop_size weight vectors evenly spaced.
        # To avoid boundary issues in scalarization, we use slightly interior points or handle boundaries carefully.
        # Standard MOEA/D uses a grid.
        # Let's use pop_size weights.
        weights = np.linspace(0.0, 1.0, pop_size).reshape(-1, 1)
        weights = np.hstack([weights, 1.0 - weights])
        
        # Ideal point (minimum of each objective so far)
        # We update this dynamically.
        ideal_point = np.min(F, axis=0)
        
        # Neighbor set: For each weight, find the closest neighbors.
        # In 2D, the neighbors are simply the adjacent weights in the grid.
        # We define a neighborhood size. For a grid of size N, neighbors are i-1 and i+1.
        # We will use a small neighborhood, e.g., 2 neighbors on each side.
        # However, with 40 points, a neighborhood of 2 (total 5 including self) is standard.
        # Let's precompute neighbor indices.
        neighbors = np.zeros((pop_size, 5), dtype=int)
        for i in range(pop_size):
            # Indices: i-2, i-1, i, i+1, i+2
            # Handle boundaries
            idxs = [i-2, i-1, i, i+1, i+2]
            valid_idxs = [j for j in idxs if 0 <= j < pop_size]
            # Pad if necessary (shouldn't be necessary with 5 slots if we just repeat or leave 0)
            # But for simplicity, we just use the valid ones and repeat the closest if needed?
            # Actually, standard MOEA/D uses a fixed neighborhood size.
            # Let's just use the 2 closest neighbors (i-1, i+1) plus self.
            # So neighborhood size 3.
            # Let's stick to a simple strategy: update using the best neighbor.
            pass # We will implement a simpler update rule: pick a random neighbor from the current best.

        # 3. Main Loop
        # We have budget - pop_size evaluations left.
        # Each iteration of MOEA/D typically involves generating offspring.
        # To fit the budget, we will generate one offspring per iteration, 
        # or a small batch, and replace the worst individual in its neighborhood.
        
        # Strategy:
        # 1. Select a parent index i based on the weight vector i.
        # 2. Generate an offspring using a local search operator (e.g., Gaussian perturbation) around X[i].
        # 3. Evaluate the offspring.
        # 4. Update the ideal point.
        # 5. Find the neighbor j in the neighborhood of i that is dominated by the offspring or has the worst Tchebycheff value.
        # 6. Replace X[j], F[j] with the offspring if it is better.
        
        # To be more efficient and ensure convergence, we can use a "polishing" step or larger steps initially.
        # Let's use an adaptive step size.
        step_size = 0.5 # Fraction of the range
        
        # We will run the loop until budget or time is exhausted.
        # Each loop iteration consumes 1 evaluation (the offspring).
        # We can afford ~160 iterations.
        
        # To speed up convergence, we can occasionally do a "restart" or larger jump if no improvement is seen.
        # But let's stick to the standard MOEA/D logic first.
        
        # Precompute Tchebycheff objective function
        def tchebycheff(f, w, z):
            # z is the ideal point
            # Handle division by zero or negative range
            # Since we are minimizing, f - z should be >= 0 if z is the min.
            # If f < z, it means we found a better point, so f-z is negative.
            # The Tchebycheff function is max_i w_i * (f_i - z_i) / (z_i_max - z_i_min)
            # But we don't have z_max easily. 
            # Standard MOEA/D uses (f_i - z_i) / (z_i_max - z_i) or just (f_i - z_i) if normalized.
            # Let's use the normalized form: (f_i - z_i) / (max_j(f_j_i) - z_i)
            # We need to track the max of each objective to normalize.
            pass # We will implement this inside the loop or as a helper with state.

        # Let's refine the Tchebycheff approach.
        # We need to track the minimum (ideal) and maximum (anti-ideal) of each objective in the population.
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0 # Avoid division by zero
        
        def tchebycheff_value(f, w, f_min, f_range):
            # Normalize f
            # If f < f_min, it's better than the current min. 
            # We should update f_min first, then compute.
            # Standard: h_i = w_i * (f_i - z_i) / (z_i_max - z_i)
            # Let's use: h_i = w_i * (f_i - f_min) / f_range
            # If f_i < f_min, the term is negative, which is good (we want to minimize h).
            # So we just compute max of these terms.
            norm_f = (f - f_min) / f_range
            terms = w * norm_f
            return np.max(terms)

        # Loop
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Select a parent index. 
            # To ensure exploration, we can select randomly, or select the one with the worst Tchebycheff value.
            # Selecting the worst one encourages improvement in underperforming regions.
            # Let's select a random index to maintain diversity, but bias towards those with worse fitness?
            # Standard MOEA/D often selects randomly from the neighborhood of a random individual.
            # Let's pick a random parent index i.
            i = np.random.randint(pop_size)
            
            # Generate offspring
            # Use a Gaussian perturbation.
            # The step size can be adaptive. Start large, shrink if no improvement.
            # For simplicity, let's use a fixed step size relative to the range, e.g., 0.2 * range.
            # Or use a smaller step for local refinement.
            # Let's try a step size of 0.1 * range initially.
            sigma = 0.1 * span
            x_offspring = X[i] + np.random.normal(0, 1, self.dim) * sigma
            
            # Clip to bounds
            x_offspring = np.clip(x_offspring, lo, hi)
            
            # Evaluate
            f_offspring = problem(x_offspring)
            n_evals += 1
            
            # Update ideal point and range
            f_min = np.minimum(f_min, f_offspring)
            f_max = np.maximum(f_max, f_offspring)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            
            # Find the best neighbor to replace
            # Neighborhood of i: i-1, i, i+1 (circular or clipped)
            # Let's use a neighborhood of size 5: i-2, i-1, i, i+1, i+2
            neighbor_indices = [i-2, i-1, i, i+1, i+2]
            valid_neighbors = [j for j in neighbor_indices if 0 <= j < pop_size]
            
            # Calculate Tchebycheff value for offspring with the weight vector of the neighbors?
            # No, in MOEA/D, each individual is associated with a weight vector.
            # The offspring is associated with the weight vector of its parent i.
            # We compare the offspring against the neighbors using the weight vector of the neighbor?
            # Actually, the standard rule is:
            # Replace the individual j in the neighborhood if the offspring dominates it, 
            # OR if the offspring has a better Tchebycheff value with respect to weight vector w_j.
            
            # Let's compute the Tchebycheff value of the offspring with respect to the parent's weight w_i.
            # And compare it to the Tchebycheff values of the neighbors with respect to their own weights?
            # This is complex. 
            # Simpler rule: 
            # 1. If offspring dominates any neighbor, replace the dominated one.
            # 2. Else, replace the neighbor with the worst Tchebycheff value (using that neighbor's weight).
            
            best_j = -1
            best_score = np.inf
            
            for j in valid_neighbors:
                # Check dominance
                if np.all(f_offspring <= F[j]) and np.any(f_offspring < F[j]):
                    # Offspring dominates neighbor j
                    best_j = j
                    break
                else:
                    # Calculate Tchebycheff value of neighbor j using weight w_j
                    # h_j = max_k w_j_k * (F[j]_k - f_min_k) / f_range_k
                    # We want to minimize this.
                    # We compare the offspring's Tchebycheff value (using w_j) with the neighbor's?
                    # Standard: Compare offspring's Tchebycheff value (with w_j) vs Neighbor's Tchebycheff value (with w_j).
                    # If offspring is better, it's a candidate.
                    
                    # Calculate Tchebycheff for offspring with weight w_j
                    norm_off = (f_offspring - f_min) / f_range
                    t_off = np.max(weights[j] * norm_off)
                    
                    # Calculate Tchebycheff for neighbor j with weight w_j
                    norm_nei = (F[j] - f_min) / f_range
                    t_nei = np.max(weights[j] * norm_nei)
                    
                    if t_off < t_nei:
                        # Offspring is better in terms of Tchebycheff for this weight
                        # We want to find the neighbor that is "most improvable" or just pick the first one?
                        # Usually, we pick the one with the worst Tchebycheff value relative to the offspring?
                        # Let's just track the best candidate for replacement.
                        # If t_off < t_nei, offspring is better.
                        # We can replace j.
                        # To be selective, we can pick the j where the improvement is largest, or just the first one.
                        # Let's pick the one with the worst current Tchebycheff value (highest t_nei) to maximize gain?
                        # Or just replace the first one found?
                        # Let's replace the one with the highest t_nei (worst performer in that niche).
                        if t_nei > best_score:
                            best_score = t_nei
                            best_j = j
            
            if best_j != -1:
                X[best_j] = x_offspring
                F[best_j] = f_offspring
                
                # Optional: Local refinement
                # If we have time, we could do a quick local search from the new point.
                # But each eval counts.
                # Let's stick to 1 eval per loop for now to stay within budget safely.

        return F

    def _latin_hypercube(self, n, dim, lo, hi):
        pts = np.zeros((n, dim))
        for i in range(dim):
            intervals = np.linspace(0, 1, n + 1)
            # Shuffle to ensure independence
            order = np.random.permutation(n)
            pts[:, i] = np.array([np.random.uniform(intervals[j], intervals[j+1]) for j in order])
        return lo + pts * (hi - lo)

# EVOLVE-BLOCK-END

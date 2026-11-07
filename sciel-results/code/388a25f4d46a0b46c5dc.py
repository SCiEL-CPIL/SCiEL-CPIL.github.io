# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for WFG7 using a hybrid approach:
    1. Fast non-dominated sorting (vectorized)
    2. Epsilon-dominance for diversity
    3. Adaptive mutation strength
    4. Deadline guard to ensure completion within 300s
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Use a fixed seed for reproducibility within the run
        rng = np.random.default_rng(12345)
        
        # Population size: balance exploration and exploitation
        # With budget=200, we can afford smaller populations for more generations
        pop_size = 30
        
        # Initialize population with good spread
        X = self._initialize_population(rng, lo, hi, pop_size)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best known points for hypervolume
        archive_X = X.copy()
        archive_F = F.copy()
        
        # Adaptive parameters
        eta_c = 20.0  # Crossover distribution index
        eta_m = 30.0  # Mutation distribution index
        prob_c = 0.9  # Crossover probability
        base_prob_m = 1.0 / self.dim  # Mutation probability per gene
        eps = 0.1  # Epsilon for dominance
        
        def fast_non_dominated_sort(F_mat):
            """
            Fast non-dominated sorting using vectorized operations.
            Returns list of fronts, each a list of indices.
            """
            n = len(F_mat)
            if n == 0:
                return []
            
            # Initialize
            S = [[] for _ in range(n)]  # Dominated sets
            n_d = np.zeros(n, dtype=np.int32)  # Number of dominators
            fronts = [[]]
            
            # Compute dominance relations
            # For each pair, check if i dominates j
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if i dominates j
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    # Check if j dominates i
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            # First front: points not dominated by any other
            for i in range(n):
                if n_d[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            return fronts

        def crowding_distance(F_mat, indices):
            """
            Calculate crowding distance for a set of indices in the same front.
            """
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_idx = sorted(indices, key=lambda i: F_mat[i][m])
                # Set boundary distances to inf
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_min = F_mat[sorted_idx[0]][m]
                f_max = F_mat[sorted_idx[-1]][m]
                range_f = f_max - f_min
                if range_f < 1e-14:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    dist[idx] += (F_mat[sorted_idx[i+1]][m] - F_mat[sorted_idx[i-1]][m]) / range_f
            return dist

        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if rng.random() > prob_c:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        beta = 1.0
                    else:
                        if rng.random() <= 0.5:
                            alpha = (x1[i] - lo[i]) / (x1[i] - x2[i] + 1e-14)
                            beta = 1 + 2 * alpha
                        else:
                            alpha = (hi[i] - x2[i]) / (x2[i] - x1[i] + 1e-14)
                            beta = 1 + 2 * alpha
                        
                        if rng.random() <= 1.0 / (beta + 1.0):
                            beta_q = (beta * rng.random()) ** (1.0 / (eta_c + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - beta * rng.random())) ** (1.0 / (eta_c + 1.0))
                    
                    c1[i] = 0.5 * ((1 + beta_q) * x1[i] + (1 - beta_q) * x2[i])
                    c2[i] = 0.5 * ((1 - beta_q) * x1[i] + (1 + beta_q) * x2[i])
            
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        def polynomial_mutation(x, prob_m):
            """Polynomial Mutation."""
            for i in range(self.dim):
                if rng.random() < prob_m:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-14)
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-14)
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                        delta_q = (val ** (1.0 / (eta_m + 1.0)) - 1.0)
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                        delta_q = 1.0 - (val ** (1.0 / (eta_m + 1.0)))
                    
                    x[i] += delta_q * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        def tournament_select(F_mat, fronts, k=2):
            """Tournament selection based on rank and crowding distance."""
            # Assign ranks
            rank = np.zeros(len(F_mat), dtype=np.int32)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Calculate crowding distances for each front
            cd = np.zeros(len(F_mat))
            for front in fronts:
                if len(front) > 2:
                    front_cd = crowding_distance(F_mat, front)
                    for i, idx in enumerate(front):
                        cd[idx] = front_cd[i]
                else:
                    for idx in front:
                        cd[idx] = float('inf')
            
            # Tournament
            candidates = rng.integers(0, len(F_mat), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and cd[c] > cd[best]:
                    best = c
            return best

        def update_archive(archive_X, archive_F, new_X, new_F):
            """Update archive with new points, keeping only non-dominated."""
            combined_X = np.vstack([archive_X, new_X])
            combined_F = np.vstack([archive_F, new_F])
            
            # Find non-dominated points in combined set
            n = len(combined_F)
            is_dominated = np.zeros(n, dtype=bool)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(combined_F[j] <= combined_F[i]) and np.any(combined_F[j] < combined_F[i]):
                        is_dominated[i] = True
                        break
            
            keep_idx = np.where(~is_dominated)[0]
            return combined_X[keep_idx], combined_F[keep_idx]

        # Main evolution loop
        gen = 0
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Sort current population
            fronts = fast_non_dominated_sort(F)
            
            # Create offspring
            offspring_X = []
            for _ in range(pop_size):
                # Select parents using tournament selection
                p1_idx = tournament_select(F, fronts)
                p2_idx = tournament_select(F, fronts)
                
                # Crossover and mutation
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Adaptive mutation: increase mutation rate if stuck
                stagnation = (gen > 5 and np.allclose(F[0], archive_F[0], atol=0.01))
                prob_m = base_prob_m * (1.5 if stagnation else 1.0)
                
                c1 = polynomial_mutation(c1, prob_m)
                c2 = polynomial_mutation(c2, prob_m)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            gen += 1
            
            # Combine parents and offspring
            X_comb = np.vstack((X, offspring_X))
            F_comb = np.vstack((F, offspring_F))
            
            # Select next generation
            fronts_comb = fast_non_dominated_sort(F_comb)
            
            X_next = []
            F_next = []
            for front in fronts_comb:
                if len(X_next) + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                else:
                    # Need to truncate
                    need = pop_size - len(X_next)
                    # Calculate crowding distance for this front
                    cd = crowding_distance(F_comb, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(need):
                        idx = front[sorted_front[i]]
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)
            
            # Update archive
            archive_X, archive_F = update_archive(archive_X, archive_F, X, F)

        # Return the archive (best non-dominated points found)
        return archive_X

    def _initialize_population(self, rng, lo, hi, size):
        """Initialize population with good spread using Latin hypercube sampling."""
        X = np.zeros((size, self.dim))
        for i in range(size):
            for j in range(self.dim):
                # Latin hypercube: divide each dimension into size equal intervals
                # and sample one point from each interval
                interval = (hi[j] - lo[j]) / size
                X[i, j] = lo[j] + (i + rng.random()) * interval
        return X

# EVOLVE-BLOCK-END

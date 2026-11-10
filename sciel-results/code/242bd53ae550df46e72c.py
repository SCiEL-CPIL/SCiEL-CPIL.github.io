"""Multi-Objective Optimizer for RE21 using NSGA-II."""
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
        deadline = start_time + 295.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization: Latin Hypercube Sampling for better initial coverage
        # With 200 budget, we can afford a decent initial population.
        # Let's use 40 points for initialization to ensure diversity.
        n_init = 40
        X_init = self._latin_hypercube(n_init, self.dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        
        # Archive to store all evaluated points (for final hypervolume calculation)
        # The task says "Every returned objective vector is recorded".
        # So we should keep all points we have evaluated, even if dominated.
        # However, NSGA-II typically keeps a population. 
        # But for the final score, we need the set of all non-dominated points found.
        # Actually, the evaluator records EVERY vector returned by problem(x).
        # So we just need to ensure we call problem(x) for points we want to be considered.
        # The archive for NSGA-II is the population.
        # We will maintain a population of size P.
        
        P = 40  # Population size
        X_pop = X_init
        F_pop = F_init
        n_evals = n_init
        
        # NSGA-II Loop
        # We have budget - n_init = 160 evaluations left.
        # Each generation creates P offspring.
        # So we can do at most 160 / 40 = 4 generations.
        # This is very few. 
        # Maybe a smaller population or more generations?
        # If P=20, we can do 8 generations.
        # If P=10, we can do 16 generations.
        
        # Let's try P=20 to allow more generations for evolution.
        # But we already used 40 for init.
        # Let's reduce init to 20 and P=20.
        # Then we have 180 left -> 9 generations.
        
        # Let's restart with smaller init.
        n_init = 20
        X_init = self._latin_hypercube(n_init, self.dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        
        P = 20
        X_pop = X_init
        F_pop = F_init
        n_evals = n_init
        
        # Standard deviation for mutation
        sigma = 0.1 * span
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Fast Non-Dominated Sorting
            ranks = self._fast_non_dominated_sort(F_pop)
            
            # 2. Crowding Distance
            cd = self._crowding_distance(F_pop, ranks)
            
            # 3. Selection (Tournament Selection based on rank and crowding)
            parents = []
            while len(parents) < P:
                p1 = self._tournament_select(X_pop, ranks, cd)
                p2 = self._tournament_select(X_pop, ranks, cd)
                parents.append(p1)
                parents.append(p2)
            
            # 4. Crossover and Mutation to create offspring
            offspring = np.zeros((P, self.dim))
            for i in range(P):
                p1 = parents[2*i]
                p2 = parents[2*i+1]
                # Simulated Binary Crossover (SBX)
                if np.random.random() < 0.9:
                    child = self._sbx(p1, p2, lo, hi)
                else:
                    child = p1.copy()
                
                # Polynomial Mutation
                for j in range(self.dim):
                    if np.random.random() < 1.0/self.dim:
                        if np.random.random() < 0.5:
                            child[j] = self._polynomial_mutation(child[j], lo[j], hi[j], lo[j])
                        else:
                            child[j] = self._polynomial_mutation(child[j], lo[j], hi[j], hi[j])
                
                offspring[i] = np.clip(child, lo, hi)
            
            # 5. Evaluate Offspring
            F_offspring = np.array([problem(x) for x in offspring])
            n_evals += P
            
            # 6. Environmental Selection
            # Combine parents and offspring
            X_combined = np.vstack([X_pop, offspring])
            F_combined = np.vstack([F_pop, F_offspring])
            
            # Sort combined population
            ranks_combined = self._fast_non_dominated_sort(F_combined)
            
            # Select top P individuals
            X_new_pop = np.zeros((P, self.dim))
            F_new_pop = np.zeros((P, self.n_obj))
            
            # We need to pick P individuals from 2P based on ranks.
            # If the last rank is partially filled, use crowding distance.
            cd_combined = self._crowding_distance(F_combined, ranks_combined)
            
            # Sort indices by rank, then by crowding distance (descending)
            # We want the best P.
            # Create a list of (rank, -cd, index)
            indices = np.arange(len(F_combined))
            # Stable sort by rank
            sorted_indices = indices[np.argsort(ranks_combined, kind='mergesort')]
            
            # We need to handle the boundary of the last front carefully.
            # Simple approach: take first P from sorted by rank.
            # But if rank k has more than remaining slots, we pick based on CD.
            
            selected_indices = []
            remaining = P
            for rank_val in np.unique(ranks_combined):
                mask = ranks_combined == rank_val
                count_in_rank = np.sum(mask)
                if count_in_rank <= remaining:
                    # Take all
                    selected_indices.extend(np.where(mask)[0])
                    remaining -= count_in_rank
                else:
                    # Take top 'remaining' by crowding distance
                    idx_in_rank = np.where(mask)[0]
                    cd_in_rank = cd_combined[idx_in_rank]
                    # Sort by CD descending
                    top_idx = idx_in_rank[np.argsort(-cd_in_rank)[:remaining]]
                    selected_indices.extend(top_idx)
                    remaining = 0
                    break
            
            X_pop = X_combined[selected_indices]
            F_pop = F_combined[selected_indices]
            
            # Optional: Local polish on the best non-dominated points?
            # With only ~9 generations, maybe not worth the complexity and risk of burning budget.
            # But if we have time, we could try.
            # For now, stick to pure NSGA-II.

        return X_pop

    def _latin_hypercube(self, n, dim, lo, hi):
        pts = np.zeros((n, dim))
        for i in range(dim):
            intervals = np.linspace(0, 1, n + 1)
            order = np.random.permutation(n)
            pts[:, i] = np.array([np.random.uniform(intervals[j], intervals[j+1]) for j in order])
        return lo + pts * (hi - lo)

    def _fast_non_dominated_sort(self, F):
        """
        Returns an array of ranks for each point in F.
        Rank 0 is the first non-dominated front.
        """
        n = len(F)
        S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of points that dominate i
        rank = np.zeros(n, dtype=int)
        
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
        
        # First front
        front_idx = 0
        current_front = [i for i in range(n) if n_dom[i] == 0]
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_idx += 1
        
        # Assign ranks
        for idx, front in enumerate(fronts):
            for i in front:
                rank[i] = idx
        
        return rank

    def _crowding_distance(self, F, ranks):
        """
        Calculate crowding distance for each point.
        Points with higher crowding distance are in less crowded regions.
        """
        n = len(F)
        cd = np.zeros(n)
        
        for front_idx in np.unique(ranks):
            front = [i for i in range(n) if ranks[i] == front_idx]
            if len(front) <= 2:
                # If front has 1 or 2 points, set CD to infinity (or a large number)
                for i in front:
                    cd[i] = np.inf
                continue
            
            for m in range(self.n_obj):
                # Sort front by objective m
                front_sorted = sorted(front, key=lambda x: F[x, m])
                
                # Assign infinity to extreme points
                cd[front_sorted[0]] = np.inf
                cd[front_sorted[-1]] = np.inf
                
                # Normalize objective values
                f_min = F[front_sorted[0], m]
                f_max = F[front_sorted[-1], m]
                f_range = f_max - f_min
                if f_range < 1e-12:
                    f_range = 1e-12
                
                # Calculate distance for intermediate points
                for k in range(1, len(front_sorted) - 1):
                    prev_val = F[front_sorted[k-1], m]
                    next_val = F[front_sorted[k+1], m]
                    dist = (next_val - prev_val) / f_range
                    cd[front_sorted[k]] += dist
        
        return cd

    def _tournament_select(self, X, ranks, cd, k=2):
        """
        Tournament selection.
        """
        n = len(X)
        candidates = np.random.choice(n, size=k, replace=False)
        
        # Find the best candidate
        best_idx = candidates[0]
        for i in range(1, k):
            idx = candidates[i]
            if ranks[idx] < ranks[best_idx]:
                best_idx = idx
            elif ranks[idx] == ranks[best_idx]:
                if cd[idx] > cd[best_idx]:
                    best_idx = idx
        
        return X[best_idx]

    def _sbx(self, p1, p2, lo, hi, eta=20.0):
        """
        Simulated Binary Crossover.
        """
        child = np.zeros_like(p1)
        for i in range(self.dim):
            if np.random.random() < 0.5:
                # Calculate child value
                if abs(p1[i] - p2[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    
                    child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            else:
                if abs(p1[i] - p2[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    
                    child[i] = 0.5 * ((1 + beta) * p2[i] + (1 - beta) * p1[i])
        
        return np.clip(child, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, x_center, eta=20.0):
        """
        Polynomial Mutation.
        """
        delta1 = (x - lo) / (hi - lo)
        delta2 = (hi - x) / (hi - lo)
        
        u = np.random.random()
        if u < 0.5:
            xy = 1.0 - delta1
            val = 1.0 - (xy) ** (eta + 1.0)
        else:
            xy = 1.0 - delta2
            val = 1.0 - (xy) ** (eta + 1.0)
        
        mut_val = x + (hi - lo) * (val - 0.5)
        
        return np.clip(mut_val, lo, hi)

# EVOLVE-BLOCK-END

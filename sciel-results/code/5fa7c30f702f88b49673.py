import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        pop_size = 20
        
        # 1. Initialization: Latin Hypercube Sampling
        # LHS provides better coverage than random for small N
        def lhs_sample(n, dim):
            # Create n samples in [0,1] with stratification
            # Simple LHS: divide [0,1] into n intervals, pick one random point in each
            # For dim dimensions, we permute the intervals
            samples = np.zeros((n, dim))
            for d in range(dim):
                intervals = np.linspace(0, 1, n + 1)
                for i in range(n):
                    samples[i, d] = rng.uniform(intervals[i], intervals[i+1])
            return samples

        X = lo + (hi - lo) * lhs_sample(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used += pop_size
        
        # 2. NSGA-II Loop
        while used + pop_size <= self.budget:
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                ranks[front] = i
            
            # Crowding distance
            distances = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    d = self._crowding_distance(F[front], front)
                    distances[front] = d
            
            # Selection
            parents = self._tournament_select(ranks, distances, pop_size, rng)
            
            # Crossover and Mutation
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                p1 = parents[i]
                p2 = parents[i+1]
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, rng)
                c1 = self._poly_mutation(c1, lo, hi, rng)
                c2 = self._poly_mutation(c2, lo, hi, rng)
                offspring_X[i] = c1
                offspring_X[i+1] = c2
            
            F_off = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine and Select
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, F_off])
            
            fronts_comb = self._fast_non_dominated_sort(F_comb)
            ranks_comb = np.zeros(len(F_comb), dtype=int)
            for i, front in enumerate(fronts_comb):
                ranks_comb[front] = i
            
            distances_comb = np.zeros(len(F_comb))
            for front in fronts_comb:
                if len(front) > 2:
                    d = self._crowding_distance(F_comb[front], front)
                    distances_comb[front] = d
            
            # Select best pop_size from combined
            # Sort by rank, then by distance (descending)
            indices = np.argsort(ranks_comb - distances_comb * 1e9) # Rank is primary, distance secondary
            # Wait, rank is int, distance is float. 
            # Better: sort by rank, then within same rank sort by distance desc.
            selected_indices = []
            for i in range(len(fronts_comb)):
                front = fronts_comb[i]
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Need to pick only some from this front
                    needed = pop_size - len(selected_indices)
                    # Sort front by distance descending
                    front_sorted = sorted(front, key=lambda idx: distances_comb[idx], reverse=True)
                    selected_indices.extend(front_sorted[:needed])
                    break
            
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]

        # 3. Local Polish
        remaining = self.budget - used
        if remaining > 0:
            # Get non-dominated points
            fronts = self._fast_non_dominated_sort(F)
            nd_indices = fronts[0]
            
            # Polish the best few points
            # We will try to improve each non-dominated point using small steps
            # Strategy: For each point, try moving each variable by +/- step_size
            # If the new point is non-dominated (with respect to current archive) or improves hypervolume, accept.
            
            # Simple approach: Just do a few rounds of mutation on the ND front
            # Since we have limited budget, let's just mutate the ND points with very small steps
            step_size = 0.01 * (hi - lo)
            
            # We have 'remaining' evaluations.
            # Let's create a queue of candidate mutations
            candidates = []
            for idx in nd_indices:
                for d in range(self.dim):
                    for sign in [-1, 1]:
                        new_x = X[idx].copy()
                        new_x[d] += sign * step_size[d]
                        new_x = np.clip(new_x, lo, hi)
                        candidates.append(new_x)
            
            # Shuffle candidates to avoid bias
            rng.shuffle(candidates)
            
            # Evaluate until budget is exhausted
            for new_x in candidates:
                if used >= self.budget:
                    break
                new_f = problem(new_x)
                used += 1
                
                # Check if new_f is non-dominated by current F
                # And if it dominates any point in F, replace it
                # Simple check: Is new_f dominated by any point in F?
                dominated = False
                for i in range(len(F)):
                    if np.all(F[i] <= new_f) and np.any(F[i] < new_f):
                        dominated = True
                        break
                
                if not dominated:
                    # It's a candidate for the front.
                    # Does it dominate any existing point?
                    for i in range(len(F)):
                        if np.all(new_f <= F[i]) and np.any(new_f < F[i]):
                            # Replace F[i] with new_f, X[i] with new_x
                            # This maintains population size
                            F[i] = new_f
                            X[i] = new_x
                            # Break? No, it might dominate multiple. But we only have one new point.
                            # Actually, if it dominates multiple, we should remove all dominated.
                            # But our population size is fixed. 
                            # Standard NSGA-II selection handles this. 
                            # Here, a simple replacement is a heuristic.
                            pass 
                    # If it doesn't dominate anyone but is not dominated, 
                    # it's a new point on the front. 
                    # We should replace the worst point (lowest crowding distance) 
                    # or just keep it if we have space? 
                    # Our population size is fixed. 
                    # Let's find the point with the lowest "value" (e.g., sum of objectives) 
                    # that is dominated by new_f? 
                    # If no point is dominated by new_f, we should replace the point 
                    # with the lowest crowding distance among those that are dominated 
                    # by new_f? No, if it doesn't dominate anyone, it's just an addition.
                    # To keep size constant, we must remove someone.
                    # Let's remove the point that is dominated by new_f if any.
                    # If none, remove the point with the lowest crowding distance 
                    # that is in the same or worse front?
                    
                    # Simpler: Just add to a temporary list and re-select?
                    # Too complex. 
                    # Let's just replace the point that is dominated by new_f.
                    # If multiple, replace the first.
                    # If none, replace the point with the highest sum of objectives (worst).
                    
                    replaced = False
                    for i in range(len(F)):
                        if np.all(new_f <= F[i]) and np.any(new_f < new_f): # Typo in thought, should be F[i]
                            pass
                    
                    # Correct logic:
                    dominated_indices = [i for i in range(len(F)) if np.all(new_f <= F[i]) and np.any(new_f < F[i])]
                    if dominated_indices:
                        # Replace the one with the largest sum (worst) among dominated
                        worst_idx = max(dominated_indices, key=lambda i: np.sum(F[i]))
                        F[worst_idx] = new_f
                        X[worst_idx] = new_x
                    else:
                        # Not dominating anyone. 
                        # Replace the worst point overall (highest sum)
                        # This is a heuristic to keep the best points.
                        worst_idx = np.argmax(np.sum(F, axis=1))
                        F[worst_idx] = new_f
                        X[worst_idx] = new_x

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [set() for _ in range(n)]  # Dominated by
        n_dom = np.zeros(n, dtype=int)  # Number of points dominated by i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        while front:
            fronts.append(np.array(front))
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            front = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        m = self.n_obj
        for k in range(m):
            # Sort indices by objective k
            sorted_indices = sorted(indices, key=lambda i: F[i, k])
            # Assign infinite distance to extremes
            distances[sorted_indices[0] - indices[0]] = np.inf 
            # Wait, indices are global indices. F is the full array.
            # Let's work with local indices or map back.
            # It's easier to work with the subset.
            
            # Let's rewrite this function to take F_subset and return distances for that subset.
            pass
            
        # Refined Crowding Distance
        F_sub = F[indices]
        n_sub = len(F_sub)
        if n_sub <= 2:
            return np.full(n_sub, np.inf)
        
        d_sub = np.zeros(n_sub)
        for k in range(self.n_obj):
            # Sort by objective k
            sort_order = np.argsort(F_sub[:, k])
            # Extremes
            d_sub[sort_order[0]] = np.inf
            d_sub[sort_order[-1]] = np.inf
            
            range_k = F_sub[sort_order[-1], k] - F_sub[sort_order[0], k]
            if range_k == 0:
                continue
                
            for i in range(1, n_sub - 1):
                d_sub[sort_order[i]] += (F_sub[sort_order[i+1], k] - F_sub[sort_order[i-1], k]) / range_k
                
        return d_sub

    def _tournament_select(self, ranks, distances, pop_size, rng):
        selected = []
        for _ in range(pop_size):
            # Pick 2 random candidates
            i1 = rng.integers(0, len(ranks))
            i2 = rng.integers(0, len(ranks))
            
            if ranks[i1] < ranks[i2]:
                selected.append(i1)
            elif ranks[i2] < ranks[i1]:
                selected.append(i2)
            else:
                # Same rank, pick higher distance
                if distances[i1] > distances[i2]:
                    selected.append(i1)
                else:
                    selected.append(i2)
        return selected

    def _sbx_crossover(self, p1, p2, lo, hi, rng):
        c1 = p1.copy()
        c2 = p2.copy()
        
        eta = 20.0
        for i in range(self.dim):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1[i] = (beta * x2 + x1) / (1 + beta)
                    c2[i] = (beta * x1 + x2) / (1 + beta)
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutation(self, x, lo, hi, rng):
        eta = 20.0
        for i in range(self.dim):
            if rng.random() < 0.5:
                u = rng.random()
                if u < 0.5:
                    delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                x[i] += delta_q * (hi[i] - lo[i])
        
        return np.clip(x, lo, hi)

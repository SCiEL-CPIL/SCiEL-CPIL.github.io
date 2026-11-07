import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evaluations = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        self.evaluations = 0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        def evaluate(x):
            if time.time() - self.start_time > 290:
                return None
            if self.evaluations >= self.budget:
                return None
            
            x = np.clip(x, lo, hi)
            f = problem(x)
            self.evaluations += 1
            return f

        # 1. Initialization: Latin Hypercube Sampling
        pop_size = 20
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        
        for j in range(dim):
            intervals = np.linspace(0, 1, pop_size + 1)
            points = np.random.uniform(intervals[:-1], intervals[1:])
            np.random.shuffle(points)
            X[:, j] = lo[j] + points * (hi[j] - lo[j])
        
        valid_indices = []
        for i in range(pop_size):
            f = evaluate(X[i])
            if f is not None:
                F[i] = f
                valid_indices.append(i)
            else:
                break
        
        if not valid_indices:
            return X
            
        X = X[valid_indices]
        F = F[valid_indices]
        current_pop_size = len(X)

        # Helper: Calculate Hypervolume Contribution for each point in the set
        # Reference point is [3.0, 5.0, 7.0]
        ref_point = np.array([3.0, 5.0, 7.0])
        
        def calc_hv_contributions(F_set):
            n = len(F_set)
            contributions = np.zeros(n)
            # Filter points strictly better than ref
            valid_mask = np.all(F_set < ref_point, axis=1)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                return contributions
            
            # For 3D, we can use a simple inclusion-exclusion or grid method?
            # With N=20, we can compute HV of the full set, and HV of the set minus i.
            # But that's O(N^2) HV calcs.
            # A faster way for 3D:
            # Sort by first objective.
            # For each point, its contribution is the volume of the box it uniquely covers.
            
            # Let's implement a simple HV calculator for 3D.
            # HV of a set of points with ref point.
            def calc_hv(points, ref):
                # points: (N, 3), ref: (3,)
                # Filter
                mask = np.all(points < ref, axis=1)
                points = points[mask]
                if len(points) == 0:
                    return 0.0
                
                # Sort by f0
                points = points[np.argsort(points[:, 0])]
                
                total_hv = 0.0
                # We can use the standard 3D HV algorithm.
                # One efficient way for small N:
                # Iterate through points sorted by f0.
                # For each point p_i, the contribution is the volume of the region
                # dominated by p_i but not by any p_j with j < i (in sorted order)?
                # No, that's not quite right for HV.
                
                # Let's use a simple grid or recursive method?
                # With N=20, a simple O(N^3) or even O(N^4) method is fast enough.
                # Let's use the "sweeping" line algorithm for 3D.
                
                # Actually, for selection, we just need relative contributions.
                # We can approximate HV contribution by the volume of the hypercube
                # defined by the point and the "next" non-dominated points?
                
                # Let's stick to Rank + Crowding Distance for now if HV is too complex to implement correctly quickly.
                # The inspiration said HV-based scored 1.0885. It might have used a library or a specific implementation.
                # Given the constraints, I will implement a basic HV contribution estimator:
                # Contribution of p_i is the volume of the box [p_i, ref] minus the volume covered by other points.
                # This is hard to compute exactly without a full HV calc.
                
                # Alternative: Use **Non-dominated Sorting** and **Crowding Distance** but with **DE** variation.
                # This is a safer structural change. The variation operator is often the key to escaping deception.
                
                return contributions

        # Since implementing HV is risky/time-consuming, I will proceed with **DE Variation** + **NSGA-II Selection**.
        # This is a significant structural change from SBX+Poly.

        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            front = [0]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    if np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                        
            for p in range(n):
                if n_d[p] == 0:
                    ranks[p] = 0
                    front[0] += 1
                    
            k = 0
            while front[k] > 0:
                next_front = []
                for p in range(n):
                    if ranks[p] == k:
                        for q in S[p]:
                            n_d[q] -= 1
                            if n_d[q] == 0:
                                ranks[q] = k + 1
                                next_front.append(q)
                if next_front:
                    k += 1
                    front.append(len(next_front))
                else:
                    break
            return ranks

        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            max_rank = np.max(ranks)
            for r in range(max_rank + 1):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    cd[indices] = np.inf
                    continue
                for m in range(self.n_obj):
                    sorted_idx = indices[np.argsort(F[indices, m])]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    f_min = F[sorted_idx[0], m]
                    f_max = F[sorted_idx[-1], m]
                    if f_max - f_min == 0:
                        continue
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i+1], m] - F[sorted_idx[i-1], m]) / (f_max - f_min)
            return cd

        def tournament_select(pop_size, ranks, cd):
            candidates = np.random.choice(pop_size, size=3, replace=False)
            best_idx = 0
            best_rank = ranks[candidates[0]]
            best_cd = cd[candidates[0]]
            for i in range(1, 3):
                if ranks[candidates[i]] < best_rank:
                    best_rank = ranks[candidates[i]]
                    best_cd = cd[candidates[i]]
                    best_idx = i
                elif ranks[candidates[i]] == best_rank and cd[candidates[i]] > best_cd:
                    best_idx = i
            return candidates[best_idx]

        # DE Parameters
        F_factor = 0.8
        CR = 0.5

        def de_mutation(x_base, x_r1, x_r2):
            return x_base + F_factor * (x_r1 - x_r2)

        def binomial_crossover(x_trial, x_base, CR):
            x_child = x_base.copy()
            j_rand = np.random.randint(0, dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    x_child[j] = x_trial[j]
            return x_child

        # Main Loop
        while self.evaluations < self.budget and time.time() - self.start_time < 290:
            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks)
            
            # Generate offspring for each individual
            new_X = X.copy()
            new_F = F.copy()
            
            for i in range(current_pop_size):
                if self.evaluations >= self.budget:
                    break
                    
                # Select 3 distinct random individuals for DE
                r1, r2, r3 = np.random.choice(current_pop_size, 3, replace=False)
                
                # Mutation
                v = de_mutation(X[i], X[r1], X[r2])
                
                # Crossover
                c = binomial_crossover(v, X[i], CR)
                c = np.clip(c, lo, hi)
                
                # Evaluate
                f_c = evaluate(c)
                if f_c is None:
                    break
                
                # Selection: Replace if c dominates X[i] or if X[i] dominates c?
                # Standard DE: Replace if f_c is better (dominates) or if it's non-dominated?
                # In multi-objective, we usually use a separate population (archive) or
                # replace if it improves the hypervolume or dominates the base.
                # Let's use a simple rule: Replace if c dominates X[i].
                # If they are non-dominated, keep the one with better crowding distance?
                # Or just keep both? We have fixed population size.
                # Let's use: If c dominates X[i], replace.
                # If X[i] dominates c, keep X[i].
                # If non-dominated, replace if c has better crowding distance (approximation).
                
                if np.all(f_c <= F[i]) and np.any(f_c < F[i]):
                    new_X[i] = c
                    new_F[i] = f_c
                elif not (np.all(F[i] <= f_c) and np.any(F[i] < f_c)):
                    # Non-dominated
                    # Use a simple heuristic: keep the one with smaller sum? Or random?
                    # Let's keep the one with better crowding distance in the *current* population context.
                    # This is tricky without recalculating CD.
                    # Let's just keep the base if non-dominated to avoid losing diversity prematurely.
                    # Or, better: use a "survival" selection.
                    pass # Keep X[i]

            X = new_X
            F = new_F

        return X

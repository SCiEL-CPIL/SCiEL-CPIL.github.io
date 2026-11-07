import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            # Ensure x is within bounds
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F):
            """
            Returns list of fronts (lists of indices).
            F is an array of shape (N, n_obj).
            """
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
                    elif dominates(F[q], F[p]):
                        S[q].append(p)
                        n_dom[p] += 1
            
            fronts = []
            current_front = []
            for p in range(n):
                if n_dom[p] == 0:
                    current_front.append(p)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                current_front = next_front
                
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for a front.
            """
            if not front_indices:
                return {}
            
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            
            for k in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])
                
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for i in range(1, m - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / f_range
            
            return dist

        # --- Main Algorithm: Differential Evolution with Pareto Selection ---
        
        # Population size
        N = 20
        CR = 0.8  # Crossover rate
        F_factor = 0.8 # Differential weight (slightly higher for exploration)
        
        # 1. Initialize population
        # Use a mix of random and grid points for better initial coverage
        pop_X = []
        pop_F = []
        
        # First, a small grid to ensure coverage
        grid_step = 3 # 3^4 = 81 points, too many for N=20? 
        # Let's just use random for N=20, it's standard and effective with DE.
        
        for _ in range(N):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)
            pop_X.append(x)
            pop_F.append(f)
            
        # 2. Evolution loop
        while used < budget and time.time() < deadline:
            # Check if we can afford a full generation
            if used + N > budget:
                # If not, reduce population size for this generation?
                # Or break? 
                # Better to just run fewer offspring if budget is tight, 
                # but DE needs N parents to generate N offspring.
                # If budget < N, we can't do a full DE step.
                # We'll just break to avoid partial messy steps.
                break
                
            offspring_X = []
            offspring_F = []
            
            for i in range(N):
                if used >= budget or time.time() > deadline:
                    break
                
                # Pick 3 distinct random indices from 0 to N-1, excluding i
                candidates = [j for j in range(N) if j != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                
                # Differential Evolution: v = x_r1 + F * (x_r2 - x_r3)
                v = pop_X[r1] + F_factor * (pop_X[r2] - pop_X[r3])
                
                # Binomial Crossover
                trial = np.zeros(dim)
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop_X[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                f = evaluate(trial)
                offspring_X.append(trial)
                offspring_F.append(f)
            
            if not offspring_X:
                break
                
            # Combine parents and offspring
            combined_X = pop_X + offspring_X
            combined_F = pop_F + offspring_F
            
            # Sort combined population
            combined_F_arr = np.array(combined_F)
            fronts = non_dominated_sort(combined_F_arr)
            
            # Build rank map
            rank_map = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    rank_map[idx] = rank
            
            # Calculate crowding distance for each front
            crowd_dists = {}
            for front in fronts:
                dists = crowding_distance(combined_F_arr, front)
                crowd_dists.update(dists)
            
            # Select next population of size N
            new_pop_X = []
            new_pop_F = []
            selected_indices = []
            
            for rank, front in enumerate(fronts):
                if len(new_pop_X) + len(front) <= N:
                    # Take all from this front
                    for idx in front:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                        selected_indices.append(idx)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = N - len(new_pop_X)
                    # Sort front by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: crowd_dists.get(i, 0.0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                        selected_indices.append(idx)
                    break
            
            # Update population
            pop_X = new_pop_X
            pop_F = new_pop_F
            
            # Safety check
            if time.time() > deadline - 5.0:
                break

        return all_X

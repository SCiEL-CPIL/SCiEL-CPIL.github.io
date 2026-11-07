import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0
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
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
            
            fronts = []
            current_front = []
            for p in range(n):
                if n_dom[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = rank[p] + 1
                            next_front.append(q)
                current_front = next_front
                
            return fronts

        def crowding_distance(F, front_indices):
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

        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            primes = [2, 3, 5, 7]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        # --- Model-and-Optimize approach with Halton initialization ---
        
        N = 12  # Population size
        CR = 0.8
        F_factor = 0.7

        # Initialize population using Halton sequence for better coverage
        pop_X = []
        pop_F = []
        
        init_size = N
        for i in range(1, init_size + 1):
            if used >= budget or time.time() > deadline:
                break
            p = halton_point(i, dim)
            x = lo + (hi - lo) * p
            f = evaluate(x)
            pop_X.append(x)
            pop_F.append(f)
        
        if used >= budget or time.time() > deadline:
            return all_X

        # Estimate ideal and nadir points to create reference directions
        def estimate_bounds(F):
            ideal = np.min(F, axis=0)
            nadir = np.max(F, axis=0)
            return ideal, nadir

        # Reference vector approach:
        # Create a set of reference vectors in the normalized objective space.
        # For 2 objectives, we can create K vectors evenly spaced between 0 and pi/2.
        n_refs = 12  # Number of reference directions

        # Main loop
        while used < budget and time.time() < deadline:
            # Estimate current bounds
            if len(pop_F) > 0:
                ideal, nadir = estimate_bounds(np.array(pop_F))
                # Handle division by zero
                span = nadir - ideal
                span[span == 0] = 1.0
            
            # Create reference vectors
            # For 2 objectives: directions are (cos(theta), sin(theta)) for theta in [0, pi/2]
            thetas = np.linspace(0, np.pi/2, n_refs)
            ref_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)])
            
            # Normalize population objectives
            F_arr = np.array(pop_F)
            F_norm = (F_arr - ideal) / span
            
            # Assign each individual to the closest reference vector
            # and calculate distance to the reference line
            ref_ids = np.zeros(len(pop_X), dtype=int)
            ref_dists = np.zeros(len(pop_X))
            
            for i, f_norm in enumerate(F_norm):
                # Dot product with reference vectors (normalized)
                dots = np.dot(f_norm, ref_vecs)
                # Distance to the reference line is sqrt(1 - dot^2)
                # But we want to minimize this distance
                # Find closest reference vector
                ref_id = np.argmax(dots)
                ref_ids[i] = ref_id
                ref_dists[i] = np.sqrt(1 - dots[ref_id]**2)
            
            # Selection: pick N individuals, ensuring diversity across reference vectors
            # Strategy: for each reference vector, pick the best individual (closest to the line)
            # If there are more individuals than reference vectors, some vectors might not be represented
            # We'll try to pick one from each reference vector if possible
            
            # Sort individuals by reference id and distance
            # Create a list of (ref_id, dist, idx)
            candidates = []
            for i in range(len(pop_X)):
                candidates.append((ref_ids[i], ref_dists[i], i))
            
            # Sort by ref_id, then by distance
            candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Pick one from each reference vector, starting with the closest
            new_pop_idx = set()
            for ref_id in range(n_refs):
                if len(new_pop_idx) >= N:
                    break
                for (rid, dist, idx) in candidates:
                    if rid == ref_id and idx not in new_pop_idx:
                        new_pop_idx.add(idx)
                        break
            
            # If we still need more individuals, pick the remaining best ones
            if len(new_pop_idx) < N:
                for (rid, dist, idx) in candidates:
                    if idx not in new_pop_idx:
                        new_pop_idx.add(idx)
                        if len(new_pop_idx) >= N:
                            break
            
            # Create new population
            pop_X = [pop_X[i] for i in sorted(new_pop_idx)]
            pop_F = [pop_F[i] for i in sorted(new_pop_idx)]
            
            # Generate offspring using DE
            offspring_X = []
            offspring_F = []
            
            for i in range(N):
                if used >= budget or time.time() > deadline:
                    break
                
                # Pick 3 distinct random indices
                candidates_idx = [j for j in range(N) if j != i]
                if len(candidates_idx) < 3:
                    continue
                r1, r2, r3 = np.random.choice(candidates_idx, size=3, replace=False)
                
                # DE mutation
                v = pop_X[r1] + F_factor * (pop_X[r2] - pop_X[r3])
                
                # Crossover
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
                
            # Combine and select
            combined_X = pop_X + offspring_X
            combined_F = pop_F + offspring_F
            
            fronts = non_dominated_sort(np.array(combined_F))
            
            # Select next population
            new_pop_X = []
            new_pop_F = []
            
            for front in fronts:
                if len(new_pop_X) + len(front) <= N:
                    for idx in front:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                else:
                    remaining = N - len(new_pop_X)
                    # Use crowding distance for selection within the front
                    crowd_dists = crowding_distance(np.array(combined_F), front)
                    sorted_front = sorted(front, key=lambda i: crowd_dists.get(i, 0.0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                    break
            
            pop_X = new_pop_X
            pop_F = new_pop_F

        return all_X

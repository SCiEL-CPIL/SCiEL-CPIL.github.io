import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Population size
        N = 10
        
        if N > budget:
            N = budget
            
        # 1. Initialization: Latin Hypercube Sampling
        # LHS ensures spread
        perm = np.random.permutation(N)
        pop = (perm + np.random.random(N)) / N
        pop = pop * (hi - lo) + lo
        
        evals = 0
        F = np.array([problem(x) for x in pop])
        evals += N
        
        # Efficient 3D Hypervolume Calculation
        def calc_hv(points, ref):
            if len(points) == 0:
                return 0.0
            
            # Filter points dominated by ref
            pts = []
            for p in points:
                if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]:
                    pts.append(p)
            
            if not pts:
                return 0.0
            
            # Sort by x
            pts = sorted(pts, key=lambda x: x[0])
            
            # Get unique x coordinates
            xs = sorted(set([p[0] for p in pts] + [ref[0]]))
            
            total_vol = 0.0
            for i in range(len(xs) - 1):
                x1, x2 = xs[i], xs[i+1]
                dx = x2 - x1
                if dx <= 0:
                    continue
                
                # Points with x <= x1
                active = [p for p in pts if p[0] <= x1]
                if not active:
                    continue
                
                # 2D Area in y-z plane
                # Sort by y
                active_y = sorted(active, key=lambda p: p[1])
                ys = sorted(set([p[1] for p in active_y] + [ref[1]]))
                
                area = 0.0
                for j in range(len(ys) - 1):
                    y1, y2 = ys[j], ys[j+1]
                    dy = y2 - y1
                    if dy <= 0:
                        continue
                    
                    # Points with y <= y1
                    active_yz = [p for p in active_y if p[1] <= y1]
                    if not active_yz:
                        continue
                    
                    # 1D Length in z (Union of intervals)
                    # Intervals are [p[2], ref[2]]
                    intervals = [(p[2], ref[2]) for p in active_yz]
                    if not intervals:
                        continue
                    
                    intervals.sort(key=lambda x: x[0])
                    merged = [list(intervals[0])]
                    for k in range(1, len(intervals)):
                        if intervals[k][0] <= merged[-1][1]:
                            if intervals[k][1] > merged[-1][1]:
                                merged[-1][1] = intervals[k][1]
                        else:
                            merged.append(list(intervals[k]))
                    
                    length = sum(m[1] - m[0] for m in merged)
                    area += dy * length
                
                total_vol += dx * area
            
            return total_vol

        def greedy_hv_select_idx(F_list, k, ref):
            """Select k indices from F_list that maximize HV."""
            if len(F_list) <= k:
                return list(range(len(F_list)))
            
            # Pre-filter: remove points dominated by ref
            valid_indices = [i for i, f in enumerate(F_list) if np.all(f < ref)]
            if not valid_indices:
                return []
            
            remaining = valid_indices
            selected = []
            
            for _ in range(k):
                best_idx = -1
                best_hv = -1
                for idx in remaining:
                    p = F_list[idx]
                    
                    # Check if p is dominated by any in selected
                    dominated = False
                    for s_idx in selected:
                        s_p = F_list[s_idx]
                        if np.all(s_p <= p) and np.any(s_p < p):
                            dominated = True
                            break
                    
                    if dominated:
                        continue
                    
                    # Calculate HV of selected + p
                    if not selected:
                        hv = calc_hv([p], ref)
                    else:
                        # To speed up, we can pass a list
                        trial = [F_list[s] for s in selected] + [p]
                        hv = calc_hv(trial, ref)
                    
                    if hv > best_hv:
                        best_hv = hv
                        best_idx = idx
                
                if best_idx != -1:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                else:
                    break
            
            return selected

        # Main Evolution Loop
        while evals < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            offspring_F = []
            
            # Check if we can generate full N
            if evals + N > budget:
                # Generate only as many as we can
                n_gen = budget - evals
            else:
                n_gen = N
                
            for i in range(n_gen):
                # DE Mutation
                # Need at least 3 parents
                if N < 3:
                    # Fallback to random mutation
                    v = pop[i] + np.random.normal(0, 0.1, dim)
                else:
                    idxs = np.random.choice(N, 3, replace=False)
                    a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
                    F_factor = 0.8
                    v = a + F_factor * (b - c)
                
                # Crossover
                CR = 0.9
                j_rand = np.random.randint(dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                evals += 1
                
                offspring.append(u)
                offspring_F.append(f_u)
                
            if not offspring:
                break
                
            # Combine Parent and Offspring
            combined_X = np.vstack([pop, np.array(offspring)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Select N best
            selected_indices = greedy_hv_select_idx(combined_F, N, ref_point)
            
            # If selection failed or returned too few, fill with random
            if len(selected_indices) < N:
                used = set(selected_indices)
                for i in range(len(combined_F)):
                    if i not in used:
                        selected_indices.append(i)
                        if len(selected_indices) == N:
                            break
            
            # Update Population
            pop = combined_X[selected_indices]
            F = combined_F[selected_indices]
            
            # If we ran out of budget, break
            if evals >= budget:
                break

        return list(pop)

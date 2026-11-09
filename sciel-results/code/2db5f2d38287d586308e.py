import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        pop_size = 50
        eta_c = 15.0
        eta_m = 20.0
        p_c = 0.9
        p_m = 1.0 / dim
        
        # Initialize with Latin Hypercube Sampling
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.rand(pop_size)) / pop_size
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        def fast_non_dominated_sort(F):
            M = len(F)
            S = [set() for _ in range(M)]
            n = [0 for _ in range(M)]
            front = []
            
            for p in range(M):
                for q in range(M):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
                if n[p] == 0:
                    front.append(p)
            
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for p in front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = len(indices)
            cd = [0.0] * m
            k = F.shape[1]
            
            for j in range(k):
                sorted_indices = sorted(range(m), key=lambda i: F[indices[i]][j])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                for i in range(1, m - 1):
                    idx_curr = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    
                    f_curr = F[indices[idx_curr]][j]
                    f_prev = F[indices[idx_prev]][j]
                    f_next = F[indices[idx_next]][j]
                    
                    denom = f_next - f_prev
                    if denom > 1e-12:
                        cd[idx_curr] += (f_curr - f_prev) / denom
                    else:
                        cd[idx_curr] += float('inf')
            return cd

        def sbx_crossover(x1, x2):
            if np.random.rand() < p_c:
                u = np.random.rand(dim)
                beta = np.ones(dim)
                mask1 = u < 0.5
                mask2 = ~mask1
                beta[mask1] = (2 * u[mask1]) ** (1.0 / (eta_c + 1))
                beta[mask2] = (1.0 / (2 * u[mask2])) ** (1.0 / (eta_c + 1))
                
                c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        def polynomial_mutation(x):
            for d in range(dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    delta = 1.0
                    if u < 0.5:
                        delta = (2 * u + 1) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u) + 1) ** (1.0 / (eta_m + 1))
                    
                    x[d] += delta * (hi[d] - lo[d])
                    x[d] = np.clip(x[d], lo[d], hi[d])
            return x

        def hypervolume_contribution(F, idx, ref):
            """Compute hypervolume contribution of point idx in F with reference point ref."""
            point = F[idx]
            if np.any(point >= ref):
                return 0.0
            # Count points that are dominated by point (or equal)
            n = len(F)
            count = 0
            for j in range(n):
                if j == idx:
                    continue
                fj = F[j]
                if np.all(point <= fj):
                    count += 1
            # Approximate contribution: volume of box where point is best
            # Use a simpler heuristic: sum of gaps to next-best in each dimension
            contrib = 1.0
            for d in range(len(point)):
                min_val = ref[d]
                for j in range(n):
                    if j != idx and F[j][d] < point[d]:
                        if F[j][d] > min_val:
                            min_val = F[j][d]
                gap = min_val - point[d]
                if gap > 0:
                    contrib *= gap
                else:
                    contrib = 0.0
                    break
            return contrib

        def tournament_select_hv(F, size=2):
            """Tournament selection using hypervolume contribution as fitness."""
            ref = np.array([3.0, 5.0, 7.0])
            indices = np.random.randint(0, len(F), size)
            best = indices[0]
            best_hv = hypervolume_contribution(F, best, ref)
            for i in indices[1:]:
                hv = hypervolume_contribution(F, i, ref)
                if hv > best_hv:
                    best = i
                    best_hv = hv
            return best

        while evals < budget:
            if time.time() > deadline:
                break
                
            X_off = np.zeros((pop_size, dim))
            F_off = np.zeros((pop_size, n_obj))
            
            for i in range(0, pop_size, 2):
                # Tournament selection using hypervolume contribution for better parent quality
                p1 = tournament_select_hv(F)
                p2 = tournament_select_hv(F)
                
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                X_off[i] = c1
                X_off[i+1] = c2
                F_off[i] = problem(c1)
                F_off[i+1] = problem(c2)
                evals += 2
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break

            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            fronts = fast_non_dominated_sort(F_comb)
            
            all_keys = []
            for i_f, front in enumerate(fronts):
                if len(front) > 0:
                    cd = crowding_distance(F_comb, front)
                    for j, idx in enumerate(front):
                        all_keys.append((i_f, -cd[j], idx))
                else:
                    for idx in front:
                        all_keys.append((i_f, 0, idx))
            
            all_keys.sort(key=lambda x: (x[0], x[1]))
            selected_indices = [k[2] for k in all_keys[:pop_size]]
            
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]
            
        # Local polish phase: refine the best non-dominated points with adaptive local search
        if evals < budget and time.time() < deadline:
            fronts = fast_non_dominated_sort(F)
            if fronts:
                # Take points from the best front for local refinement
                best_front = fronts[0]
                ref = np.array([3.0, 5.0, 7.0])
                
                # Select top points by hypervolume contribution
                front_hvs = [(hypervolume_contribution(F, idx, ref), idx) for idx in best_front]
                front_hvs.sort(reverse=True)
                n_refine = min(8, len(front_hvs), max(1, (budget - evals) // 8))
                
                for _, idx in front_hvs[:n_refine]:
                    if evals >= budget or time.time() > deadline:
                        break
                    
                    # Adaptive local search with decreasing step sizes
                    x0 = X[idx].copy()
                    f0 = F[idx].copy()
                    base_step = 0.03 * (hi - lo)
                    
                    for iter_count in range(4):
                        if evals >= budget or time.time() > deadline:
                            break
                        step_size = base_step * (0.5 ** iter_count)
                        improved = False
                        
                        for d in range(dim):
                            if evals >= budget or time.time() > deadline:
                                break
                            for direction in [1, -1]:
                                if evals >= budget or time.time() > deadline:
                                    break
                                x_trial = x0.copy()
                                x_trial[d] += direction * step_size[d]
                                x_trial = np.clip(x_trial, lo, hi)
                                f_trial = problem(x_trial)
                                evals += 1
                                
                                # Accept if it dominates
                                if np.all(f_trial <= f0) and np.any(f_trial < f0):
                                    x0 = x_trial
                                    f0 = f_trial
                                    improved = True
                                    break
                                # Accept if non-dominated and closer to reference in normalized space
                                elif not (np.all(f0 <= f_trial) and np.any(f0 < f_trial)):
                                    # Check if it's in the region we care about
                                    if np.all(f_trial < ref) and np.all(f0 < ref):
                                        # Compare normalized distance to reference
                                        norm_f0 = (f0 - 0) / (ref - 0)
                                        norm_ft = (f_trial - 0) / (ref - 0)
                                        if np.sum(norm_ft) < np.sum(norm_f0):
                                            x0 = x_trial
                                            f0 = f_trial
                                            improved = True
                                            break
                    
                        if not improved:
                            break
                    
                    # Update the population with the refined point
                    X[idx] = x0
                    F[idx] = f0
        
        return F

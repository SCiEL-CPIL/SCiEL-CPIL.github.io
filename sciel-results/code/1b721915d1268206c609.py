import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        ref = np.array([3.0, 5.0, 7.0])
        
        # Allocate budget: 70% global search, 30% local refinement
        global_budget = int(budget * 0.70)
        local_budget = budget - global_budget
        
        pop_size = 40
        eta_c = 12.0
        eta_m = 15.0
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

        def compute_dominance_matrix(F):
            """Precompute dominance relationships for faster selection."""
            n = len(F)
            dominated_by = [set() for _ in range(n)]
            dominates = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominates[i].add(j)
                        dominated_by[j].add(i)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominates[j].add(i)
                        dominated_by[i].add(j)
            return dominates, dominated_by

        def hv_contribution_fast(F, idx, dominates, dominated_by):
            """Fast HV contribution using precomputed dominance."""
            point = F[idx]
            if np.any(point >= ref):
                return 0.0
            
            # Only count points that are dominated by this point
            n_dom = len(dominates[idx])
            
            # Approximate contribution based on gaps
            contrib = 1.0
            for d in range(len(point)):
                min_val = ref[d]
                for j in dominates[idx]:
                    if F[j][d] < point[d]:
                        if F[j][d] > min_val:
                            min_val = F[j][d]
                gap = min_val - point[d]
                if gap > 0:
                    contrib *= gap
                else:
                    contrib = 0.0
                    break
            return contrib

        def tournament_select(F, dominates, dominated_by, size=2):
            """Tournament selection using precomputed dominance."""
            indices = np.random.randint(0, len(F), size)
            best = indices[0]
            best_hv = hv_contribution_fast(F, best, dominates, dominated_by)
            for i in indices[1:]:
                hv = hv_contribution_fast(F, i, dominates, dominated_by)
                if hv > best_hv:
                    best = i
                    best_hv = hv
            return best

        # Global search phase
        gen = 0
        while evals < global_budget:
            if time.time() > deadline:
                break
            
            # Precompute dominance for selection
            dominates, dominated_by = compute_dominance_matrix(F)
            
            X_off = np.zeros((pop_size, dim))
            F_off = np.zeros((pop_size, n_obj))
            
            for i in range(0, pop_size, 2):
                if time.time() > deadline:
                    break
                
                p1 = tournament_select(F, dominates, dominated_by)
                p2 = tournament_select(F, dominates, dominated_by)
                
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
            gen += 1
        
        # Local refinement phase: Nelder-Mead style local search on best front points
        if evals < budget and time.time() < deadline:
            fronts = fast_non_dominated_sort(F)
            if fronts:
                best_front = fronts[0]
                
                # Select top points by simple HV proxy
                front_scores = []
                for idx in best_front:
                    point = F[idx]
                    if np.any(point >= ref):
                        front_scores.append((0.0, idx))
                    else:
                        # Simple proxy: sum of normalized distances
                        score = np.sum((ref - point) / ref)
                        front_scores.append((score, idx))
                
                front_scores.sort(reverse=True)
                n_refine = min(12, len(front_scores), max(1, local_budget // 10))
                
                for _, idx in front_scores[:n_refine]:
                    if evals >= budget or time.time() > deadline:
                        break
                    
                    # Nelder-Mead style local search
                    x0 = X[idx].copy()
                    f0 = F[idx].copy()
                    
                    # Create initial simplex
                    simplex = [x0.copy()]
                    f_simplex = [f0.copy()]
                    step = 0.02 * (hi - lo)
                    
                    for d in range(min(dim, 6)):  # Limit simplex size
                        if evals >= budget or time.time() > deadline:
                            break
                        x_trial = x0.copy()
                        x_trial[d] += step[d]
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = problem(x_trial)
                        evals += 1
                        simplex.append(x_trial)
                        f_simplex.append(f_trial)
                    
                    # Iterate simplex
                    max_iter = min(15, (budget - evals) // len(simplex))
                    for it in range(max_iter):
                        if evals >= budget or time.time() > deadline:
                            break
                        
                        # Sort simplex by sum of objectives
                        order = np.argsort([np.sum(f) for f in f_simplex])
                        simplex = [simplex[i] for i in order]
                        f_simplex = [f_simplex[i] for i in order]
                        
                        # Centroid of all but worst
                        centroid = np.mean(simplex[:-1], axis=0)
                        
                        # Reflection
                        xr = centroid + 1.0 * (centroid - simplex[-1])
                        xr = np.clip(xr, lo, hi)
                        fr = problem(xr)
                        evals += 1
                        
                        # Check if reflection is better than worst
                        if np.sum(fr) < np.sum(f_simplex[-1]):
                            # Check if better than best
                            if np.sum(fr) < np.sum(f_simplex[0]):
                                # Expansion
                                xe = centroid + 2.0 * (centroid - simplex[-1])
                                xe = np.clip(xe, lo, hi)
                                fe = problem(xe)
                                evals += 1
                                if np.sum(fe) < np.sum(fr):
                                    simplex[-1] = xe
                                    f_simplex[-1] = fe
                                else:
                                    simplex[-1] = xr
                                    f_simplex[-1] = fr
                            else:
                                simplex[-1] = xr
                                f_simplex[-1] = fr
                        else:
                            # Contraction
                            xc = centroid + 0.5 * (simplex[-1] - centroid)
                            xc = np.clip(xc, lo, hi)
                            fc = problem(xc)
                            evals += 1
                            if np.sum(fc) < np.sum(f_simplex[-1]):
                                simplex[-1] = xc
                                f_simplex[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    f_simplex[i] = problem(simplex[i])
                                    evals += 1
                                    if evals >= budget or time.time() > deadline:
                                        break
                        
                        # Update best point
                        best_in_simplex = np.argmin([np.sum(f) for f in f_simplex])
                        if np.sum(f_simplex[best_in_simplex]) < np.sum(f0):
                            x0 = simplex[best_in_simplex].copy()
                            f0 = f_simplex[best_in_simplex].copy()
                    
                    # Update population
                    X[idx] = x0
                    F[idx] = f0
        
        return F

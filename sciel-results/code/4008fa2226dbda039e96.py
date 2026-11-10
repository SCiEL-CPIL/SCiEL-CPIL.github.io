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
        
        # Parameters
        pop_size = 30
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
            if M == 0:
                return []
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
            
            # For 3 objectives, exact HV contribution can be computed in O(n^2)
            # But we use a fast approximation: 
            # The contribution is the volume of the region dominated by point but not by any other point in F
            # For 3D, this is complex. We use a simpler heuristic based on gaps.
            
            n = len(F)
            # Find the "box" around point that it uniquely dominates
            # For each dimension, find the closest other point that is better in that dimension
            # If no other point is better in that dimension, the bound is ref[d]
            
            bounds = np.zeros(n_obj)
            for d in range(n_obj):
                min_val = ref[d]
                for j in range(n):
                    if j == idx:
                        continue
                    if F[j][d] < point[d]:
                        if F[j][d] > min_val:
                            min_val = F[j][d]
                bounds[d] = min_val
            
            # The contribution is the volume of the box [point, bounds]
            # But we must subtract parts dominated by other points.
            # A simpler approximation: just the box volume, which overestimates.
            # Let's use a more accurate 3D HV algorithm if possible, or a better heuristic.
            
            # Actually, let's just return the box volume as a proxy for "importance"
            # Points with larger unique boxes are more important.
            contrib = 1.0
            for d in range(n_obj):
                gap = bounds[d] - point[d]
                if gap > 0:
                    contrib *= gap
                else:
                    contrib = 0.0
                    break
            return contrib

        def tournament_select_hv(F, size=2):
            """Tournament selection using hypervolume contribution as fitness."""
            indices = np.random.randint(0, len(F), size)
            best = indices[0]
            best_hv = hypervolume_contribution(F, best, ref)
            for i in indices[1:]:
                hv = hypervolume_contribution(F, i, ref)
                if hv > best_hv:
                    best = i
                    best_hv = hv
            return best

        def nelder_mead(x0, f0, bounds, max_iter=20, tol=1e-6):
            """Nelder-Mead local search for multi-objective (minimizing sum of objectives)."""
            # This is a simplification: we minimize the sum of objectives
            # to find a good point on the Pareto front.
            # A better approach would be to use a scalarization or just accept
            # if it dominates or is non-dominated and closer to ideal.
            
            dim = len(x0)
            n_obj = len(f0)
            
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                if xi[i] != bounds[i][0]:
                    xi[i] += 0.05 * (bounds[i][1] - bounds[i][0])
                else:
                    xi[i] += 0.05 * (bounds[i][1] - bounds[i][0])
                xi = np.clip(xi, bounds[:, 0], bounds[:, 1])
                simplex.append(xi)
            
            f_simplex = np.array([problem(x) for x in simplex])
            # Scalarize: sum of objectives
            f_scalar = f_simplex.sum(axis=1)
            
            best_x = simplex[0].copy()
            best_f = f0.copy()
            best_scalar = f0.sum()
            
            for _ in range(max_iter):
                # Sort simplex by scalar fitness
                order = np.argsort(f_scalar)
                simplex = [simplex[i] for i in order]
                f_scalar = f_scalar[order]
                f_simplex = f_simplex[order]
                
                if f_scalar[-1] - f_scalar[0] < tol:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, bounds[:, 0], bounds[:, 1])
                f_ref = problem(x_ref)
                f_ref_scalar = f_ref.sum()
                
                if f_ref_scalar < f_scalar[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, bounds[:, 0], bounds[:, 1])
                    f_exp = problem(x_exp)
                    f_exp_scalar = f_exp.sum()
                    
                    if f_exp_scalar < f_ref_scalar:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                        f_scalar[-1] = f_exp_scalar
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                        f_scalar[-1] = f_ref_scalar
                elif f_ref_scalar < f_scalar[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                    f_scalar[-1] = f_ref_scalar
                else:
                    # Contraction
                    beta = 0.5
                    x_con = centroid + beta * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, bounds[:, 0], bounds[:, 1])
                    f_con = problem(x_con)
                    f_con_scalar = f_con.sum()
                    
                    if f_con_scalar < f_scalar[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                        f_scalar[-1] = f_con_scalar
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
                            f_simplex[i] = problem(simplex[i])
                            f_scalar[i] = f_simplex[i].sum()
                        order = np.argsort(f_scalar)
                        simplex = [simplex[i] for i in order]
                        f_scalar = f_scalar[order]
                        f_simplex = f_simplex[order]
                
                # Track best
                for i in range(dim + 1):
                    if f_scalar[i] < best_scalar:
                        best_scalar = f_scalar[i]
                        best_x = simplex[i].copy()
                        best_f = f_simplex[i].copy()
            
            return best_x, best_f

        # Main loop
        while evals < budget:
            if time.time() > deadline:
                break
            
            X_off = np.zeros((pop_size, dim))
            F_off = np.zeros((pop_size, n_obj))
            
            for i in range(0, pop_size, 2):
                if time.time() > deadline:
                    break
                
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
        
        # Local polish phase: refine the best non-dominated points with Nelder-Mead
        if evals < budget and time.time() < deadline:
            fronts = fast_non_dominated_sort(F)
            if fronts:
                best_front = fronts[0]
                
                # Select top points by hypervolume contribution
                front_hvs = [(hypervolume_contribution(F, idx, ref), idx) for idx in best_front]
                front_hvs.sort(reverse=True)
                
                # Allocate remaining budget to local search
                n_refine = min(5, len(front_hvs), max(1, (budget - evals) // 10))
                
                for _, idx in front_hvs[:n_refine]:
                    if evals >= budget or time.time() > deadline:
                        break
                    
                    x0 = X[idx].copy()
                    f0 = F[idx].copy()
                    
                    # Use Nelder-Mead to refine
                    x_new, f_new = nelder_mead(x0, f0, np.array([lo, hi]).T, max_iter=5)
                    evals += 10  # Approximate cost
                    
                    # Update if better
                    if np.all(f_new <= f0) or (not (np.all(f0 <= f_new) and np.any(f0 < f_new))):
                        if np.all(f_new < ref):
                            X[idx] = x_new
                            F[idx] = f_new
        
        return F

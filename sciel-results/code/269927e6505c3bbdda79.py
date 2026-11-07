import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 305.0  # Safety margin (evaluator kills at 330s)
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
            all_F.append(f.copy())
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
                    elif dominates(F[q], F[p]):
                        S[q].append(p)
                        n_dom[p] += 1
            
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

        def local_polish(x_start, f_start, max_evals):
            """Nelder-Mead style local search on a single point."""
            if used >= budget or time.time() > deadline:
                return x_start, f_start
            
            # Initialize simplex
            simplex_X = [x_start.copy()]
            simplex_F = [f_start.copy()]
            
            # Add perturbed points
            for d in range(dim):
                x_pert = x_start.copy()
                step = 0.05 * (hi[d] - lo[d])
                x_pert[d] += step
                x_pert = np.clip(x_pert, lo, hi)
                f_pert = evaluate(x_pert)
                simplex_X.append(x_pert)
                simplex_F.append(f_pert)
            
            n_simplex = dim + 1
            max_iter = min(max_evals - dim, 50)
            
            for iteration in range(max_iter):
                if used >= budget or time.time() > deadline:
                    break
                
                # Sort by composite objective (sum of normalized objectives)
                order = np.argsort([np.sum(simplex_F[i]) for i in range(n_simplex)])
                simplex_X = [simplex_X[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex_X[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex_X[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = evaluate(x_ref)
                
                if np.sum(f_ref) < np.sum(simplex_F[-2]):
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex_X[-1])
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = evaluate(x_exp)
                    if np.sum(f_exp) < np.sum(f_ref):
                        simplex_X[-1] = x_exp
                        simplex_F[-1] = f_exp
                    else:
                        simplex_X[-1] = x_ref
                        simplex_F[-1] = f_ref
                elif np.sum(f_ref) < np.sum(simplex_F[0]):
                    simplex_X[-1] = x_ref
                    simplex_F[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex_X[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = evaluate(x_con)
                    if np.sum(f_con) < np.sum(simplex_F[-1]):
                        simplex_X[-1] = x_con
                        simplex_F[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n_simplex):
                            simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                            simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                            simplex_F[i] = evaluate(simplex_X[i])
            
            # Return best point from simplex
            best_idx = np.argmin([np.sum(f) for f in simplex_F])
            return simplex_X[best_idx], simplex_F[best_idx]

        # --- Strategy: Reduced initialization + Adaptive DE + Weighted-Sum Local Polish ---
        
        # 1. Initial random sampling (25% of budget = 50 points)
        n_init = max(25, budget // 4)
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        # Convert to arrays for faster processing
        X_arr = np.array(all_X)
        F_arr = np.array(all_F)
        
        # Get initial Pareto front
        fronts = non_dominated_sort(F_arr)
        if not fronts:
            return all_X
            
        pareto_indices = fronts[0]
        pareto_F = F_arr[pareto_indices]
        
        if len(pareto_F) == 0:
            return all_X
        
        # 2. Adaptive DE for the remaining budget
        N_pop = 8  # Smaller population for more generations
        CR = 0.6
        F_base = 0.7
        
        # Initialize DE population from current Pareto front + random
        if len(pareto_F) >= N_pop:
            # Select diverse points from Pareto front using crowding distance
            dists = crowding_distance(F_arr, pareto_indices)
            sorted_pareto = sorted(pareto_indices, key=lambda i: dists.get(i, 0.0), reverse=True)
            pop_F = [F_arr[i] for i in sorted_pareto[:N_pop]]
            pop_X = [X_arr[i] for i in sorted_pareto[:N_pop]]
        else:
            pop_F = list(pareto_F)
            pop_X = list(X_arr[pareto_indices])
            
            n_needed = N_pop - len(pop_F)
            for _ in range(n_needed):
                if used >= budget or time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(dim)
                f = evaluate(x)
                pop_F.append(f)
                pop_X.append(x)
                
            if len(pop_F) < N_pop:
                N_pop = len(pop_F)
                if N_pop < 2:
                    return all_X

        pop_X = np.array(pop_X)
        pop_F = np.array(pop_F)
        
        # Main DE loop
        gen = 0
        while used < budget and time.time() < deadline:
            gen += 1
            # Adaptive F: decrease over generations
            F_factor = F_base * (1.0 - 0.3 * gen / 50.0) * (0.7 + 0.3 * np.random.rand())
            
            offspring_X = []
            offspring_F = []
            
            for i in range(N_pop):
                if used >= budget or time.time() > deadline:
                    break
                
                if N_pop < 4:
                    trial = pop_X[i] + 0.1 * (hi - lo) * np.random.randn(dim)
                    trial = np.clip(trial, lo, hi)
                else:
                    candidates = [j for j in range(N_pop) if j != i]
                    r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                    v = pop_X[r1] + F_factor * (pop_X[r2] - pop_X[r3])
                    
                    trial = np.zeros(dim)
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop_X[i][j]
                    trial = np.clip(trial, lo, hi)
                
                f = evaluate(trial)
                offspring_X.append(trial)
                offspring_F.append(f)
            
            if not offspring_X:
                break
                
            combined_X = np.vstack([pop_X, np.array(offspring_X)])
            combined_F = np.vstack([pop_F, np.array(offspring_F)])
            
            fronts = non_dominated_sort(combined_F)
            if not fronts:
                break
                
            crowd_dists = {}
            for front in fronts:
                dists = crowding_distance(combined_F, front)
                crowd_dists.update(dists)
            
            new_pop_X = []
            new_pop_F = []
            
            for rank, front in enumerate(fronts):
                if len(new_pop_X) + len(front) <= N_pop:
                    for idx in front:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                else:
                    remaining = N_pop - len(new_pop_X)
                    sorted_front = sorted(front, key=lambda i: crowd_dists.get(i, 0.0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                    break
            
            pop_X = np.array(new_pop_X)
            pop_F = np.array(new_pop_F)
            
            # Stop DE if we're close to budget for local polish
            if used > budget - 80:
                break
        
        # 3. Weighted-sum decomposition + L-BFGS-B polish
        if used < budget and time.time() < deadline - 10:
            # Get current Pareto front from all evaluations
            F_all = np.array(all_F)
            fronts_all = non_dominated_sort(F_all)
            
            if fronts_all:
                pareto_idx_all = fronts_all[0]
                pareto_F = F_all[pareto_idx_all]
                
                # Select diverse starting points using crowding distance
                dists_all = crowding_distance(F_all, pareto_idx_all)
                sorted_pareto_all = sorted(pareto_idx_all, key=lambda i: dists_all.get(i, 0.0), reverse=True)
                
                # Use multiple weight vectors for decomposition
                n_weights = min(8, len(sorted_pareto_all))
                # Generate weight vectors: emphasize different trade-offs
                weight_vectors = []
                for w in range(n_weights):
                    if n_weights == 1:
                        weight_vectors.append([0.5, 0.5])
                    else:
                        # Mix of equal weights and biased weights
                        if w == 0:
                            weight_vectors.append([1.0, 0.0])  # Minimize obj 1
                        elif w == 1:
                            weight_vectors.append([0.0, 1.0])  # Minimize obj 2
                        else:
                            # Interpolate between extremes
                            t = (w - 1) / max(1, n_weights - 2)
                            weight_vectors.append([t, 1.0 - t])
                
                # Polish with weighted-sum + L-BFGS-B
                for wi, w_vec in enumerate(weight_vectors):
                    if used >= budget - 5 or time.time() > deadline - 5:
                        break
                    
                    # Pick a diverse starting point for this weight
                    start_idx = sorted_pareto_all[wi % len(sorted_pareto_all)]
                    x_start = all_X[start_idx].copy()
                    
                    # Define objective function for L-BFGS-B
                    def weighted_objective(x):
                        if used >= budget or time.time() > deadline - 5:
                            return 1e10
                        x_clipped = np.clip(x, lo, hi)
                        f = np.array(problem(x_clipped), dtype=float)
                        used_count = [0]
                        # We need to track evaluations properly
                        # But since L-BFGS-B calls this multiple times, we need to be careful
                        # Let's just count it in the evaluate function
                        f_eval = evaluate(x_clipped)
                        return np.dot(w_vec, f_eval)
                    
                    # Use L-BFGS-B with bounds
                    bounds_list = list(zip(lo, hi))
                    try:
                        result = so.minimize(
                            weighted_objective,
                            x_start,
                            method='L-BFGS-B',
                            bounds=bounds_list,
                            options={'maxiter': 30, 'maxfun': 30}
                        )
                    except:
                        pass
        
        return all_X

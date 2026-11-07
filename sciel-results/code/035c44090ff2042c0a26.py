import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a local RNG to decouple from the global seed state if possible,
        # though the evaluator sets np.random.seed. Using default_rng is safer for reproducibility of algorithm logic.
        rng = np.random.default_rng(42)
        
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x), dtype=float)
            except Exception:
                return None
            
            # Check for invalid values
            if not np.all(np.isfinite(f)):
                return None
                
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
            
            # Optimize dominance check using vectorization where possible or simple loops for small N
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            
            for p in range(n):
                for q in range(p + 1, n):
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
                    elif dominates(F[q], F[p]):
                        S[q].append(p)
                        n_dom[p] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            
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
                # Alternate direction to ensure spread
                if rng.random() > 0.5:
                    x_pert[d] += step
                else:
                    x_pert[d] -= step
                x_pert = np.clip(x_pert, lo, hi)
                f_pert = evaluate(x_pert)
                if f_pert is None:
                    return x_start, f_start
                simplex_X.append(x_pert)
                simplex_F.append(f_pert)
            
            n_simplex = dim + 1
            max_iter = min(max_evals - dim, 30) # Reduced iterations for speed
            
            for iteration in range(max_iter):
                if used >= budget or time.time() > deadline:
                    break
                
                # Sort by composite objective (sum of normalized objectives)
                # Simple sum is okay for local polish if scales are somewhat comparable
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
                if f_ref is None:
                    break
                
                if np.sum(f_ref) < np.sum(simplex_F[-2]):
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex_X[-1])
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = evaluate(x_exp)
                    if f_exp is None:
                        simplex_X[-1] = x_ref
                        simplex_F[-1] = f_ref
                        break
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
                    if f_con is None:
                        break
                    if np.sum(f_con) < np.sum(simplex_F[-1]):
                        simplex_X[-1] = x_con
                        simplex_F[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n_simplex):
                            simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                            simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                            simplex_F[i] = evaluate(simplex_X[i])
                            if simplex_F[i] is None:
                                return x_start, f_start
            
            # Return best point from simplex
            best_idx = np.argmin([np.sum(f) for f in simplex_F])
            return simplex_X[best_idx], simplex_F[best_idx]

        # --- Strategy: Reduced initialization + Adaptive DE + Local Polish ---
        
        # 1. Initial random sampling (20% of budget)
        n_init = max(10, budget // 5)
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        if not all_F:
            return all_X
            
        X_arr = np.array(all_X)
        F_arr = np.array(all_F)
        
        # Get initial Pareto front
        fronts = non_dominated_sort(F_arr)
        if not fronts:
            return all_X
            
        pareto_indices = fronts[0]
        
        # 2. Adaptive DE for the remaining budget
        N_pop = 8  # Smaller population for more generations
        
        # Initialize DE population from current Pareto front + random
        if len(pareto_indices) >= N_pop:
            dists = crowding_distance(F_arr, pareto_indices)
            sorted_pareto = sorted(pareto_indices, key=lambda i: dists.get(i, 0.0), reverse=True)
            pop_idx = sorted_pareto[:N_pop]
        else:
            # If front is small, take all and fill with random
            pop_idx = list(pareto_indices)
            while len(pop_idx) < N_pop and used < budget:
                x = lo + (hi - lo) * rng.random(dim)
                f = evaluate(x)
                if f is None: break
                pop_idx.append(used - 1)
            if len(pop_idx) < 2:
                return all_X

        pop_X = [X_arr[i].copy() for i in pop_idx]
        pop_F = [F_arr[i].copy() for i in pop_idx]
        
        F_factor = 0.8
        CR = 0.7
        
        while used < budget and time.time() < deadline:
            # Ensure we have enough for DE
            if len(pop_X) < 3:
                x = lo + (hi - lo) * rng.random(dim)
                f = evaluate(x)
                if f is None: break
                pop_X.append(x)
                pop_F.append(f)
                continue
                
            # Select parents
            i = rng.integers(len(pop_X))
            j = rng.integers(len(pop_X))
            k = rng.integers(len(pop_X))
            if i == j or i == k or j == k:
                continue
                
            x1 = pop_X[i]
            x2 = pop_X[j]
            x3 = pop_X[k]
            
            # DE Mutation
            v = x1 + F_factor * (x2 - x3)
            
            # Crossover
            trial = x1.copy()
            j_rand = rng.integers(dim)
            for d in range(dim):
                if rng.random() < CR or d == j_rand:
                    trial[d] = v[d]
            trial = np.clip(trial, lo, hi)
            
            f_trial = evaluate(trial)
            if f_trial is None:
                break
            
            # Selection: Tournament selection with dominance
            # Find a candidate to replace
            # Simple approach: If trial is non-dominated, add it. 
            # Then prune population.
            
            # Check if trial is dominated by any in pop
            dominated = False
            for f_old in pop_F:
                if dominates(f_old, f_trial):
                    dominated = True
                    break
                    
            if not dominated:
                # Remove points dominated by trial
                new_pop_X = []
                new_pop_F = []
                for x_old, f_old in zip(pop_X, pop_F):
                    if not dominates(f_trial, f_old):
                        new_pop_X.append(x_old)
                        new_pop_F.append(f_old)
                new_pop_X.append(trial)
                new_pop_F.append(f_trial)
                
                # Prune if too large
                if len(new_pop_X) > N_pop:
                    # Compute crowding distance
                    dists = crowding_distance(np.array(new_pop_F), list(range(len(new_pop_F))))
                    # Find index with min CD
                    min_idx = min(dists, key=dists.get)
                    new_pop_X.pop(min_idx)
                    new_pop_F.pop(min_idx)
                    
                pop_X = new_pop_X
                pop_F = new_pop_F
                
        # 3. Local Polish on top Pareto points
        # Update global arrays with any new points from DE
        # (Note: evaluate() already appends to all_X/all_F)
        
        # Re-compute fronts on all evaluated points
        if all_F:
            F_arr_final = np.array(all_F)
            fronts_final = non_dominated_sort(F_arr_final)
            if fronts_final:
                pareto_final = fronts_final[0]
                # Pick top 3 diverse points for polishing
                dists_final = crowding_distance(F_arr_final, pareto_final)
                sorted_pareto_final = sorted(pareto_final, key=lambda i: dists_final.get(i, 0.0), reverse=True)
                polish_candidates = sorted_pareto_final[:3]
                
                for idx in polish_candidates:
                    if used >= budget or time.time() > deadline:
                        break
                    x_polished, f_polished = local_polish(X_arr_final[idx], F_arr_final[idx], max_evals=10)
                    # local_polish already evaluates and appends to all_X/all_F
            
        return all_X

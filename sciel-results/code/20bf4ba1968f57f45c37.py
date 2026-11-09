"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using model-and-optimize approach for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        eval_count = [0]

        def check_time():
            return time.time() < deadline

        def add_point(x, f):
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count[0] += 1

        def evaluate(x):
            f = problem(x)
            add_point(x, f)
            return f

        # --- Phase 1: Structured initial sampling ---
        # Use Latin Hypercube Sampling for diverse initial spread
        n_lhs = min(40, budget // 5)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_points[:, d] = (perm + np.random.random(n_lhs)) / n_lhs
        lhs_points = lo + span * lhs_points

        for x in lhs_points:
            evaluate(x)
            if eval_count[0] >= budget or not check_time():
                break

        # --- Phase 2: NSGA-II-style evolutionary refinement ---
        # Use SBX crossover and polynomial mutation for better exploration/exploitation
        
        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sorting. Returns list of fronts (each front is list of indices)."""
            F_arr = np.array(F_arr)
            n = len(F_arr)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_count[i] += 1
            
            # First front
            front = [i for i in range(n) if n_count[i] == 0]
            if not front:
                return [list(range(n))]
            
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front = next_front
            
            return fronts

        def crowding_distance(F_arr, front_indices):
            """Calculate crowding distance for points in a front."""
            F_arr = np.array(F_arr)
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            for m in range(self.n_obj):
                f_vals = F_arr[front_indices, m]
                f_min = np.min(f_vals)
                f_max = np.max(f_vals)
                if f_max - f_min < 1e-10:
                    dist[:] = np.inf
                    continue
                
                # Sort by objective m
                sorted_idx = np.argsort(f_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (f_vals[sorted_idx[k+1]] - f_vals[sorted_idx[k-1]]) / (f_max - f_min)
            
            return dist

        def sbx_crossover(x1, x2, bounds_arr, eta=20.0):
            """Simulated Binary Crossover."""
            x1 = np.array(x1, dtype=float)
            x2 = np.array(x2, dtype=float)
            lo = bounds_arr[0]
            hi = bounds_arr[1]
            span = hi - lo
            
            u = np.random.random(dim)
            beta = np.ones(dim)
            mask = u <= 0.5
            beta[mask] = (2 * u[mask]) ** (1.0 / (eta + 1))
            beta[~mask] = (2 * (1 - u[~mask])) ** (1.0 / (eta + 1))
            
            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            return c1, c2

        def polynomial_mutation(x, bounds_arr, eta=20.0):
            """Polynomial Mutation."""
            x = np.array(x, dtype=float)
            lo = bounds_arr[0]
            hi = bounds_arr[1]
            span = hi - lo
            
            pm = 1.0 / dim
            x_mut = x.copy()
            
            for i in range(dim):
                if np.random.random() < pm:
                    u = np.random.random()
                    x_i = (x[i] - lo[i]) / span[i] if span[i] > 0 else 0.5
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x_mut[i] = x[i] + delta * span[i]
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            
            return x_mut

        # NSGA-II main loop
        pop_size = 50
        bounds_arr = (lo, hi)
        
        # Initialize population from existing points
        if len(all_X) >= pop_size:
            # Select diverse subset
            F_arr = np.array(all_F)
            fronts = fast_non_dominated_sort(F_arr)
            
            pop_idx = []
            for front in fronts:
                if len(pop_idx) + len(front) <= pop_size:
                    pop_idx.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(pop_idx)
                    cd = crowding_distance(F_arr, front)
                    sorted_front = np.argsort(-cd)
                    pop_idx.extend(front[sorted_front[:remaining]])
                    break
            
            # If still not enough, add random points
            while len(pop_idx) < pop_size:
                idx = np.random.randint(0, len(all_X))
                if idx not in pop_idx:
                    pop_idx.append(idx)
            
            pop_X = [all_X[i].copy() for i in pop_idx[:pop_size]]
            pop_F = [all_F[i].copy() for i in pop_idx[:pop_size]]
        else:
            # Use all existing points and pad with random
            pop_X = [x.copy() for x in all_X]
            pop_F = [f.copy() for f in all_F]
            
            while len(pop_X) < pop_size:
                x = lo + span * np.random.random(dim)
                f = evaluate(x)
                pop_X.append(x.copy())
                pop_F.append(f.copy())
        
        # Main NSGA-II loop
        gen = 0
        while eval_count[0] < budget and check_time():
            gen += 1
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and eval_count[0] < budget and check_time():
                # Tournament selection for two parents
                def tournament_select():
                    idx1 = np.random.randint(0, len(pop_X))
                    idx2 = np.random.randint(0, len(pop_X))
                    
                    # Compare by front rank and crowding distance
                    fronts = fast_non_dominated_sort(np.array(pop_F))
                    rank = np.zeros(len(pop_X), dtype=int)
                    for fi, front in enumerate(fronts):
                        for idx in front:
                            rank[idx] = fi
                    
                    # Get crowding distances
                    cd = np.zeros(len(pop_X))
                    for front in fronts:
                        if len(front) > 2:
                            cd_front = crowding_distance(np.array(pop_F), front)
                            for k, idx in enumerate(front):
                                cd[idx] = cd_front[k]
                    
                    # Tournament
                    if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and cd[idx1] > cd[idx2]):
                        return idx1
                    else:
                        return idx2
                
                p1_idx = tournament_select()
                p2_idx = tournament_select()
                
                # SBX crossover
                c1, c2 = sbx_crossover(pop_X[p1_idx], pop_X[p2_idx], bounds_arr)
                
                # Polynomial mutation
                c1 = polynomial_mutation(c1, bounds_arr)
                c2 = polynomial_mutation(c2, bounds_arr)
                
                # Evaluate
                f1 = evaluate(c1)
                if eval_count[0] < budget and check_time():
                    f2 = evaluate(c2)
                    offspring_X.append(c1.copy())
                    offspring_F.append(f1.copy())
                    offspring_X.append(c2.copy())
                    offspring_F.append(f2.copy())
                else:
                    offspring_X.append(c1.copy())
                    offspring_F.append(f1.copy())
                    break
            
            # Combine parents and offspring
            all_pop_X = pop_X + offspring_X
            all_pop_F = pop_F + offspring_F
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(np.array(all_pop_F))
            
            # Select next population
            new_pop_X = []
            new_pop_F = []
            for front in fronts:
                if len(new_pop_X) + len(front) <= pop_size:
                    for idx in front:
                        new_pop_X.append(all_pop_X[idx].copy())
                        new_pop_F.append(all_pop_F[idx].copy())
                else:
                    remaining = pop_size - len(new_pop_X)
                    cd = crowding_distance(np.array(all_pop_F), front)
                    sorted_front = np.argsort(-cd)
                    for idx in front[sorted_front[:remaining]]:
                        new_pop_X.append(all_pop_X[idx].copy())
                        new_pop_F.append(all_pop_F[idx].copy())
                    break
            
            pop_X = new_pop_X
            pop_F = new_pop_F
            
            # Update global archive with non-dominated points
            if len(pop_F) > 0:
                F_arr = np.array(pop_F)
                non_dom_idx = fast_non_dominated_sort(F_arr)[0]
                for idx in non_dom_idx:
                    x_new = pop_X[idx]
                    f_new = pop_F[idx]
                    
                    # Check if dominated by existing points
                    dominated = False
                    for f_old in all_F:
                        if np.all(np.array(f_old) <= np.array(f_new)) and np.any(np.array(f_old) < np.array(f_new)):
                            dominated = True
                            break
                    
                    if not dominated:
                        add_point(x_new, f_new)

        return all_X
# EVOLVE-BLOCK-END

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

        # --- Phase 2: Differential Evolution with NSGA-II selection ---
        # Use DE/rand/1/bin mutation and binomial crossover
        
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

        def de_mutation_crossover(pop_X, pop_F, target_idx, F=0.8, CR=0.9):
            """Differential Evolution Mutation and Crossover (DE/rand/1/bin)."""
            pop_size = len(pop_X)
            if pop_size < 4:
                # Fallback to random mutation if population is too small
                x_target = np.array(pop_X[target_idx], dtype=float)
                lo = bounds_arr[0]
                hi = bounds_arr[1]
                span = hi - lo
                x_mut = x_target + (np.random.random(dim) - 0.5) * 0.1 * span
                return np.clip(x_mut, lo, hi)

            # Select 3 distinct random individuals
            indices = np.random.choice(pop_size, size=3, replace=False)
            # Ensure none is the target
            while target_idx in indices:
                indices = np.random.choice(pop_size, size=3, replace=False)
            
            x_r1 = np.array(pop_X[indices[0]], dtype=float)
            x_r2 = np.array(pop_X[indices[1]], dtype=float)
            x_r3 = np.array(pop_X[indices[2]], dtype=float)
            x_target = np.array(pop_X[target_idx], dtype=float)
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            v = x_r1 + F * (x_r2 - x_r3)
            
            # Crossover
            lo = bounds_arr[0]
            hi = bounds_arr[1]
            j_rand = np.random.randint(0, dim)
            u = np.random.random(dim)
            c = np.where(u <= CR, v, x_target)
            c[j_rand] = v[j_rand]
            
            # Clip to bounds
            c = np.clip(c, lo, hi)
            
            return c

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
            
            # Pre-compute ranks and crowding distances for selection efficiency
            # Note: We re-compute these every generation in the original code, 
            # but for DE we don't need tournament selection for parents, 
            # we just generate offspring for each parent.
            
            while len(offspring_X) < pop_size and eval_count[0] < budget and check_time():
                # Generate offspring for a random parent
                p_idx = np.random.randint(0, len(pop_X))
                
                # DE mutation and crossover
                c = de_mutation_crossover(pop_X, pop_F, p_idx)
                
                # Evaluate
                f = evaluate(c)
                offspring_X.append(c.copy())
                offspring_F.append(f.copy())
                
                # If we have budget for another, we could do a second one, 
                # but standard DE usually creates one offspring per parent.
                # To fill the offspring population of size pop_size, we just loop.
                # If we want to speed up convergence, we can try to create 
                # offspring for the best parents more often, but random is fine for DE.
            
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

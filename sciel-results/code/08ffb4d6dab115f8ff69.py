import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin (evaluator kills at 330s)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        archive_x = []
        archive_f = []
        evals = 0
        ref_point = np.array([3000.0, 0.06])

        def time_ok():
            return time.time() < deadline

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            af = np.array(archive_f_list)
            leq = np.all(af <= f_new, axis=1)
            lt = np.any(af < f_new, axis=1)
            return np.any(leq & lt)
            
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def add_to_archive(x, f):
            if is_dominated(f, archive_f):
                return False
            
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def halton(dim, n, base_list=None):
            if base_list is None:
                base_list = [2, 3, 5, 7]
            points = []
            for i in range(1, n + 1):
                p = []
                for j in range(dim):
                    f = 1.0
                    r = 0.0
                    i_tmp = i
                    while i_tmp > 0:
                        f /= base_list[j]
                        r += f * (i_tmp % base_list[j])
                        i_tmp //= base_list[j]
                    p.append(r)
                points.append(p)
            return np.array(points)

        # 1. Initialization
        init_size = 20
        if time_ok():
            X_init = lo + range_ * halton(self.dim, init_size)
            for x in X_init:
                if not time_ok() or evals >= self.budget:
                    break
                f = problem(x)
                evals += 1
                add_to_archive(x, f)
        
        # 2. Main Search: NSGA-II style with Crowding Distance
        # We will use a population-based approach to ensure diversity.
        pop_size = 20
        generations = (self.budget - init_size) // pop_size
        if generations < 1:
            generations = 1
            
        # Initialize population with Halton + Random
        # We already have some in archive, but let's create a proper population
        # Actually, for NSGA-II we need a population of candidates.
        # Let's start with the archive points + random points to fill pop_size.
        
        def crowding_distance(f_arr):
            """
            f_arr: (N, M) array of objective values for N points
            Returns: array of crowding distances
            """
            n = len(f_arr)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            m = f_arr.shape[1]
            
            for j in range(m):
                # Sort indices by j-th objective
                sorted_idx = np.argsort(f_arr[:, j])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                min_val = f_arr[sorted_idx[0], j]
                max_val = f_arr[sorted_idx[-1], j]
                span = max_val - min_val
                if span == 0:
                    continue
                
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (f_arr[sorted_idx[k+1], j] - f_arr[sorted_idx[k-1], j]) / span
            
            return cd

        def fast_non_dominated_sort(f_arr):
            """
            f_arr: (N, M) array
            Returns: list of fronts, each front is a list of indices
            """
            n = len(f_arr)
            S = [[] for _ in range(n)] # S[i] = set of solutions dominated by i
            n_dom = np.zeros(n, dtype=int) # n_dom[i] = number of solutions dominated by i
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(f_arr[p] <= f_arr[q]) and np.any(f_arr[p] < f_arr[q]):
                        S[p].append(q)
                    elif np.all(f_arr[q] <= f_arr[p]) and np.any(f_arr[q] < f_arr[p]):
                        n_dom[p] += 1
            
            front1 = [i for i in range(n) if n_dom[i] == 0]
            if not front1:
                return [[]]
            
            fronts.append(front1)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                i += 1
            
            return fronts

        # Current population: start with archive points
        # If archive is small, add random points
        pop_x = list(archive_x)
        pop_f = list(archive_f)
        
        # Fill population with random points if needed
        while len(pop_x) < pop_size and time_ok() and evals < self.budget:
            x = lo + np.random.rand(self.dim) * range_
            f = problem(x)
            evals += 1
            pop_x.append(x)
            pop_f.append(f)
            add_to_archive(x, f)
            
        # If we still don't have enough, duplicate (shouldn't happen with init 20)
        while len(pop_x) < pop_size:
            pop_x.append(np.random.uniform(lo, hi))
            f = problem(pop_x[-1])
            evals += 1
            pop_f.append(f)
            add_to_archive(pop_x[-1], f)

        while evals < self.budget and time_ok():
            # Create offspring
            offspring_x = []
            offspring_f = []
            
            for _ in range(pop_size):
                if evals >= self.budget or not time_ok():
                    break
                
                # Binary Tournament Selection
                i1 = np.random.randint(len(pop_x))
                i2 = np.random.randint(len(pop_x))
                
                # Compare crowding distance within same front?
                # Simplified: just pick the one with lower f_sum or random
                # Proper NSGA-II uses fronts. Let's use a simplified dominance check.
                # If i1 dominates i2, pick i1.
                # Else if i2 dominates i1, pick i2.
                # Else, pick random.
                
                f1 = pop_f[i1]
                f2 = pop_f[i2]
                
                if np.all(f1 <= f2) and np.any(f1 < f2):
                    parent_idx = i1
                elif np.all(f2 <= f1) and np.any(f2 < f1):
                    parent_idx = i2
                else:
                    # Use crowding distance if available, else random
                    parent_idx = np.random.choice([i1, i2])
                
                parent = pop_x[parent_idx]
                
                # SBX Crossover (Simplified to blend)
                if np.random.rand() < 0.9:
                    beta = np.random.rand(self.dim)
                    child1 = (1 - beta) * parent + beta * lo
                    child2 = (1 - (1 - beta)) * parent + (1 - beta) * hi
                    # This is not good. Let's use simple uniform crossover or blend.
                    # Blend:
                    alpha = np.random.rand()
                    child = alpha * parent + (1 - alpha) * (lo + np.random.rand(self.dim) * range_)
                else:
                    child = parent.copy()
                
                # Mutation
                if np.random.rand() < 0.9:
                    sigma = 0.1 * range_
                    child = child + sigma * np.random.normal(size=self.dim)
                
                child = np.clip(child, lo, hi)
                
                f_child = problem(child)
                evals += 1
                
                offspring_x.append(child)
                offspring_f.append(f_child)
                
                # Add to archive immediately to maintain global best
                add_to_archive(child, f_child)
            
            if not offspring_x:
                break
                
            # Combine parent and offspring
            combined_x = pop_x + offspring_x
            combined_f = pop_f + offspring_f
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(np.array(combined_f))
            
            # Select next population
            new_pop_x = []
            new_pop_f = []
            
            for front in fronts:
                if len(new_pop_x) + len(front) <= pop_size:
                    for idx in front:
                        new_pop_x.append(combined_x[idx])
                        new_pop_f.append(combined_f[idx])
                else:
                    # Need to truncate this front
                    # Calculate crowding distance for this front
                    front_f = np.array([combined_f[i] for i in front])
                    cd = crowding_distance(front_f)
                    # Sort by crowding distance descending
                    sorted_front = np.argsort(cd)[::-1]
                    
                    slots = pop_size - len(new_pop_x)
                    for i in range(slots):
                        orig_idx = front[sorted_front[i]]
                        new_pop_x.append(combined_x[orig_idx])
                        new_pop_f.append(combined_f[orig_idx])
                    break
            
            pop_x = new_pop_x
            pop_f = new_pop_f

        # 3. Polish: Nelder-Mead on extreme points
        if time_ok() and evals < self.budget and len(archive_f) > 0:
            archive_f_arr = np.array(archive_f)
            idx_min_f1 = np.argmin(archive_f_arr[:, 0])
            idx_min_f2 = np.argmin(archive_f_arr[:, 1])
            
            # Run NM for min f1
            if evals < self.budget - 10:
                x0 = archive_x[idx_min_f1].copy()
                def obj_f1(x):
                    nonlocal evals
                    if not time_ok() or evals >= self.budget:
                        return 1e10
                    fx = problem(x)
                    evals += 1
                    return fx[0]
                
                try:
                    res = minimize(obj_f1, x0, method='Nelder-Mead', options={'maxiter': 10, 'xatol': 1e-5, 'fatol': 1e-5})
                    if res.success:
                        fx_res = problem(res.x)
                        evals += 1
                        add_to_archive(res.x, fx_res)
                except:
                    pass

            # Run NM for min f2
            if evals < self.budget - 10:
                x0 = archive_x[idx_min_f2].copy()
                def obj_f2(x):
                    nonlocal evals
                    if not time_ok() or evals >= self.budget:
                        return 1e10
                    fx = problem(x)
                    evals += 1
                    return fx[1]
                
                try:
                    res = minimize(obj_f2, x0, method='Nelder-Mead', options={'maxiter': 10, 'xatol': 1e-5, 'fatol': 1e-5})
                    if res.success:
                        fx_res = problem(res.x)
                        evals += 1
                        add_to_archive(res.x, fx_res)
                except:
                    pass

        return archive_x

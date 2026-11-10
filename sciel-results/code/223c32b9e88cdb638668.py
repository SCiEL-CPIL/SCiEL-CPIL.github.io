"""NSGA-II Multi-objective Optimizer for RE21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE21 (4 vars, 2 objectives).
    Strategy:
      1. NSGA-II (Non-dominated Sorting Genetic Algorithm II).
      2. LHS initialization.
      3. SBX Crossover and Polynomial Mutation.
      4. Deadline guard to ensure we finish within wall-clock limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_points = []
        all_f = []
        eval_count = [0]

        def eval_and_store(x):
            nonlocal all_points, all_f
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_points.append(x.copy())
            all_f.append(f.copy())
            eval_count[0] += 1
            return f

        def time_ok():
            return time.time() < deadline and eval_count[0] < self.budget

        def remaining_budget():
            return max(0, self.budget - eval_count[0])

        # NSGA-II Parameters
        pop_size = 40
        n_gen = max(1, (self.budget - pop_size) // pop_size)
        
        # Crossover and Mutation parameters
        cx_prob = 0.9
        mut_prob = 1.0 / self.dim
        eta_c = 20.0
        eta_m = 20.0

        # Helper: LHS Initialization
        def lhs_sample(n, dim, lo, hi, rng):
            samples = np.zeros((n, dim))
            for d in range(dim):
                intervals = np.arange(n) / n
                fractions = intervals + rng.random(n) / n
                samples[:, d] = lo[d] + fractions * (hi[d] - lo[d])
            return samples

        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # Domination sets
            n_dom = np.zeros(n, dtype=int)  # Domination counts
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        
            current_front = [i for i in range(n) if n_dom[i] == 0]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            cd = np.zeros(n)
            if n <= 2:
                return cd
            
            for m in range(self.n_obj):
                F_obj = F[front_indices, m]
                sorted_idx = np.argsort(F_obj)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                range_f = F_obj[sorted_idx[-1]] - F_obj[sorted_idx[0]]
                if range_f == 0:
                    continue
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (F_obj[sorted_idx[k+1]] - F_obj[sorted_idx[k-1]]) / range_f
            return cd

        # Helper: Selection (Tournament)
        def select_tournament(pop_indices, F, fronts, cd):
            rank = np.zeros(len(pop_indices))
            for i, front in enumerate(fronts):
                for idx in front:
                    # Map global index to local index in pop_indices if needed, 
                    # but here we operate on global indices directly for simplicity in this closure
                    pass 
            
            # We need ranks for the specific population passed
            # Let's re-calculate ranks for the given pop_indices subset
            # Actually, it's easier to keep global ranks if we pass global indices.
            # Let's assume pop_indices are global indices into all_points/all_f? 
            # No, NSGA-II usually operates on a specific population.
            # Let's refactor to operate on local arrays for clarity.
            
            # This helper is getting complex. Let's implement a simpler selection:
            # Binary tournament based on (rank, cd)
            # We need rank and cd for the individuals in the population.
            pass

        # Let's implement the main loop with local arrays to avoid index confusion.
        
        # Phase 1: Initialization
        X_pop = lhs_sample(pop_size, self.dim, lo, hi, rng)
        F_pop = np.array([eval_and_store(x) for x in X_pop])
        
        if not time_ok():
            return all_points

        # Main Loop
        while time_ok() and eval_count[0] < self.budget:
            # 1. Create offspring
            X_off = np.zeros_like(X_pop)
            F_off = np.zeros_like(F_pop)
            
            # Calculate ranks and crowding for current population
            fronts = fast_non_dominated_sort(F_pop)
            cd = np.zeros(pop_size)
            for front in fronts:
                cd[front] = crowding_distance(F_pop, front)
            
            # Ranks
            ranks = np.zeros(pop_size)
            for i, front in enumerate(fronts):
                ranks[front] = i
                
            # Selection function
            def select_one():
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                if ranks[i1] < ranks[i2]:
                    return i1
                elif ranks[i2] < ranks[i1]:
                    return i2
                else:
                    if cd[i1] > cd[i2]:
                        return i1
                    elif cd[i2] > cd[i1]:
                        return i2
                    else:
                        return i1 if rng.random() < 0.5 else i2

            for i in range(0, pop_size, 2):
                if rng.random() < cx_prob:
                    p1_idx = select_one()
                    p2_idx = select_one()
                    p1 = X_pop[p1_idx]
                    p2 = X_pop[p2_idx]
                    
                    # SBX Crossover
                    u = rng.random(self.dim)
                    beta = np.zeros(self.dim)
                    mask = u <= 0.5
                    beta[mask] = (2 * u[mask]) ** (1.0 / (eta_c + 1))
                    beta[~mask] = (1 / (2 * (1 - u[~mask]))) ** (1.0 / (eta_c + 1))
                    
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                    
                    # Mutation
                    def mutate(x):
                        mut_mask = rng.random(self.dim) < mut_prob
                        x[mut_mask] = x[mut_mask].copy()
                        for d in np.where(mut_mask)[0]:
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            x[d] = x[d] + delta * (hi[d] - lo[d])
                        return np.clip(x, lo, hi)

                    c1 = mutate(c1)
                    c2 = mutate(c2)
                    
                    X_off[i] = c1
                    X_off[i+1] = c2
                else:
                    X_off[i] = X_pop[rng.integers(pop_size)]
                    X_off[i+1] = X_pop[rng.integers(pop_size)]
            
            # Evaluate offspring
            # We must be careful not to exceed budget
            for i in range(pop_size):
                if not time_ok() or eval_count[0] >= self.budget:
                    break
                F_off[i] = eval_and_store(X_off[i])
            
            # If we broke out of evaluation early, resize offspring
            if eval_count[0] >= self.budget or not time_ok():
                # We might have fewer than pop_size evaluated offspring
                # Just append what we have? No, NSGA-II expects full population.
                # If we ran out of budget, we stop.
                if eval_count[0] >= self.budget:
                    break

            # 2. Combine parents and offspring
            X_comb = np.vstack([X_pop, X_off])
            F_comb = np.vstack([F_pop, F_off])
            n_comb = 2 * pop_size
            
            # 3. Non-dominated sort combined
            fronts_comb = fast_non_dominated_sort(F_comb)
            
            # 4. Select next generation
            X_new = []
            F_new = []
            i = 0
            while len(X_new) < pop_size and i < len(fronts_comb):
                front = fronts_comb[i]
                if len(X_new) + len(front) <= pop_size:
                    # Take whole front
                    X_new.extend(X_comb[front])
                    F_new.extend(F_comb[front])
                else:
                    # Take partial front, sort by crowding distance
                    cd_front = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda idx: cd_front[idx - min(front)], reverse=True)
                    # Note: crowding_distance returns array aligned with front indices order?
                    # My implementation: cd = np.zeros(n); cd[front] = ... 
                    # So cd_front corresponds to the values in the 'front' list order?
                    # No, crowding_distance(F, front_indices) returns an array of size len(front_indices)
                    # where index k corresponds to front_indices[k].
                    
                    # Let's re-verify crowding_distance return.
                    # def crowding_distance(F, front_indices):
                    #    n = len(front_indices)
                    #    cd = np.zeros(n)
                    #    ...
                    #    return cd
                    # Yes, cd[k] is the distance of front_indices[k].
                    
                    # So we sort front_indices by cd values descending
                    sorted_indices = np.argsort(cd_front)[::-1]
                    needed = pop_size - len(X_new)
                    selected = front[sorted_indices[:needed]]
                    X_new.extend(X_comb[selected])
                    F_new.extend(F_comb[selected])
                    break
                i += 1
            
            # Update population
            X_pop = np.array(X_new)
            F_pop = np.array(F_new)
            
            # Check if we should continue
            if not time_ok():
                break

        return all_points
# EVOLVE-BLOCK-END

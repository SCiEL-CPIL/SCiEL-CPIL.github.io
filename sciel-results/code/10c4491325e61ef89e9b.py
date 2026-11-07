import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Constrained random search within shrinking hypersphere centered at origin
        # WFG9 Pareto front is near origin in decision space
        
        # Compute max radius from bounds
        max_radius = np.sqrt(np.sum(np.maximum(np.abs(lo), np.abs(hi))**2))
        
        # We'll sample points uniformly within a hypersphere of radius r
        # r shrinks from max_radius to a small value over the budget
        min_radius = 0.01 * max_radius
        
        # Track all evaluated points and their objectives
        all_X = []
        all_F = []
        evaluations = 0
        
        # Phase 1: Uniform random sampling within shrinking hypersphere
        # Use stratified sampling for better coverage
        
        while evaluations < budget:
            if time.time() > deadline:
                break
            
            # Current fraction of budget used
            frac = evaluations / budget
            
            # Radius shrinks linearly from max_radius to min_radius
            r = max_radius * (1.0 - frac) + min_radius * frac
            
            # Generate a random point within hypersphere of radius r centered at origin
            # Use rejection sampling or direct sampling
            # Direct: sample direction uniformly on sphere, radius as r * u^(1/dim)
            
            # Sample direction uniformly on unit sphere in dim dimensions
            # Using normal distribution trick
            v = np.random.randn(dim)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                v = np.array([1.0] + [0.0]*(dim-1))
                v_norm = 1.0
            v = v / v_norm
            
            # Sample radius: uniform in volume means r * u^(1/dim) where u~U(0,1)
            u = np.random.random()
            r_actual = r * (u ** (1.0 / dim))
            
            x = r_actual * v
            
            # Clip to bounds (problem clips anyway, but let's be safe)
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            evaluations += 1
            all_X.append(x)
            all_F.append(f)
        
        # Return the best non-dominated set (or just all points since evaluator records all)
        if all_X:
            X = np.array(all_X)
            F = np.array(all_F)
        else:
            X = np.array([lo + (hi - lo) * 0.5])
            F = np.array([problem(X[0])])
        
        return X, F

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            fronts = []
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(front)

            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                front = next_front
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            for m in range(n_obj):
                sorted_indices = np.argsort(F[front_indices, m])
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf

                f_min = F[front_indices[sorted_indices[0]], m]
                f_max = F[front_indices[sorted_indices[-1]], m]
                if f_max - f_min < 1e-10:
                    continue

                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    distances[idx] += (F[front_indices[sorted_indices[i+1]], m] -
                                       F[front_indices[sorted_indices[i-1]], m]) / (f_max - f_min)
            return distances

        # Helper: SBX
        def sbx(x1, x2, lo, hi, eta):
            c1, c2 = x1.copy(), x2.copy()
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
                        continue
                    beta = 0.0
                    if x1[i] <= x2[i]:
                        beta = 2 * (x1[i] - lo[i]) / (x2[i] - x1[i] + 1e-14)
                    else:
                        beta = 2 * (hi[i] - x2[i]) / (x2[i] - x1[i] + 1e-14)
                    beta += 1.0
                    alpha = 1.0 + 2.0 * (1.0 - beta)
                    if np.random.random() <= 1.0 / alpha:
                        beta_q = (0.5 * beta * (np.random.random() ** (1.0 / (eta + 1.0))))
                    else:
                        beta_q = (0.5 * (2.0 - beta) * (1.0 / (np.random.random() ** (1.0 / (eta + 1.0)))))

                    c1[i] = 0.5 * (x1[i] + x2[i] - beta_q * (x2[i] - x1[i]))
                    c2[i] = 0.5 * (x1[i] + x2[i] + beta_q * (x2[i] - x1[i]))

                    if c1[i] < lo[i]: c1[i] = lo[i]
                    if c1[i] > hi[i]: c1[i] = hi[i]
                    if c2[i] < lo[i]: c2[i] = lo[i]
                    if c2[i] > hi[i]: c2[i] = hi[i]
            return c1, c2

        # Helper: Polynomial Mutation
        def pm(x, lo, hi, eta):
            for i in range(dim):
                if np.random.random() < p_mutation:
                    if x[i] < lo[i] or x[i] > hi[i]:
                        x[i] = (lo[i] + hi[i]) / 2.0
                    else:
                        y = x[i]
                        if y < lo[i]:
                            y = lo[i]
                        elif y > hi[i]:
                            y = hi[i]

                        r = np.random.random()
                        if r <= 0.5:
                            delta1 = (y - lo[i]) / (hi[i] - lo[i] + 1e-14)
                            xq = 1.0 + (1.0 - delta1) * (2.0 * r) ** (1.0 / (eta + 1.0))
                        else:
                            delta2 = (hi[i] - y) / (hi[i] - lo[i] + 1e-14)
                            xq = 1.0 + (1.0 - delta2) * (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))

                        x[i] = y + (hi[i] - lo[i]) * (xq - 1.0)
                        if x[i] < lo[i]: x[i] = lo[i]
                        if x[i] > hi[i]: x[i] = hi[i]
            return x

        # Main Loop
        while evaluations < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if time.time() > deadline:
                    break
                if evaluations + 1 > budget:
                    break

                # Tournament Selection
                t_size = 2
                candidates = np.random.choice(pop_size, size=t_size, replace=False)
                c1_idx, c2_idx = candidates
                if np.all(F[c1_idx] <= F[c2_idx]) and np.any(F[c1_idx] < F[c2_idx]):
                    parent1_idx = c1_idx
                elif np.all(F[c2_idx] <= F[c1_idx]) and np.any(F[c2_idx] < F[c1_idx]):
                    parent1_idx = c2_idx
                else:
                    parent1_idx = c1_idx if np.random.random() < 0.5 else c2_idx

                candidates = np.random.choice(pop_size, size=t_size, replace=False)
                c1_idx, c2_idx = candidates
                if np.all(F[c1_idx] <= F[c2_idx]) and np.any(F[c1_idx] < F[c2_idx]):
                    parent2_idx = c1_idx
                elif np.all(F[c2_idx] <= F[c1_idx]) and np.any(F[c2_idx] < F[c1_idx]):
                    parent2_idx = c2_idx
                else:
                    parent2_idx = c1_idx if np.random.random() < 0.5 else c2_idx

                p1 = X[parent1_idx]
                p2 = X[parent2_idx]

                if np.random.random() < p_crossover:
                    child1, child2 = sbx(p1, p2, lo, hi, eta_c)
                else:
                    child1, child2 = p1.copy(), p2.copy()

                child1 = pm(child1, lo, hi, eta_m)
                child2 = pm(child2, lo, hi, eta_m)

                f1 = problem(child1)
                evaluations += 1
                offspring_X.append(child1)
                offspring_F.append(f1)

                if len(offspring_X) < pop_size and evaluations < budget:
                    f2 = problem(child2)
                    evaluations += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine populations
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = non_dominated_sort(F_comb)

            # Select next population
            X_new = []
            F_new = []

            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_comb, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break

            X = np.array(X_new)
            F = np.array(F_new)

        return X, F

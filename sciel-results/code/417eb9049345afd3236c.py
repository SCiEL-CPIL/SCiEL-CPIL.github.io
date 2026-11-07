"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer with NSGA-II-style selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size: use most of the budget for initial population
        pop_size = min(30, max(10, self.budget // 10))
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: check if a dominates b (a <= b in all obj, a < b in at least one)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: non-dominated sort
        def non_dominated_sort(F):
            n_pts = len(F)
            fronts = []
            S = [[] for _ in range(n_pts)]  # S[i] = indices dominated by i
            nd = np.zeros(n_pts, dtype=int)  # domination count
            
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        nd[i] += 1
            
            # First front
            front = [i for i in range(n_pts) if nd[i] == 0]
            fronts.append(front)
            
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                front = next_front
            
            return fronts

        # Helper: crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            for k in range(m):
                # Sort by objective k
                sorted_idx = np.argsort(F[indices, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F[indices[sorted_idx[0]], k]
                f_max = F[indices[sorted_idx[-1]], k]
                if f_max - f_min < 1e-10:
                    continue
                
                for i in range(1, len(indices) - 1):
                    idx = sorted_idx[i]
                    dist[idx] += (F[indices[sorted_idx[i + 1]], k] - 
                                  F[indices[sorted_idx[i - 1]], k]) / (f_max - f_min)
            return dist

        # Helper: SBX crossover
        def sbx_crossover(x1, x2, eta=15.0):
            c1, c2 = x1.copy(), x2.copy()
            for i in range(n):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        continue
                    if x1[i] <= x2[i]:
                        beta = 2.0 * (x1[i] - lo[i]) / (x2[i] - x1[i] + 1e-14)
                    else:
                        beta = 2.0 * (hi[i] - x2[i]) / (x2[i] - x1[i] + 1e-14)
                    beta += 1.0
                    alpha = 1.0 + 2.0 * (1.0 - beta)
                    u = np.random.random()
                    if u <= 1.0 / alpha:
                        beta_q = (0.5 * beta * (u ** (1.0 / (eta + 1.0))))
                    else:
                        beta_q = (0.5 * (2.0 - beta) * ((1.0 / (1.0 - u)) ** (1.0 / (eta + 1.0))))
                    
                    c1[i] = 0.5 * (x1[i] + x2[i] - beta_q * (x2[i] - x1[i]))
                    c2[i] = 0.5 * (x1[i] + x2[i] + beta_q * (x2[i] - x1[i]))
                    
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        # Helper: polynomial mutation
        def poly_mutation(x, eta=20.0, p_mut=1.0 / n):
            x = x.copy()
            for i in range(n):
                if np.random.random() < p_mut:
                    y = x[i]
                    u = np.random.random()
                    if u <= 0.5:
                        delta1 = (y - lo[i]) / (hi[i] - lo[i] + 1e-14)
                        xq = 1.0 + (1.0 - delta1) * (2.0 * u) ** (1.0 / (eta + 1.0))
                    else:
                        delta2 = (hi[i] - y) / (hi[i] - lo[i] + 1e-14)
                        xq = 1.0 + (1.0 - delta2) * (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    x[i] = y + (hi[i] - lo[i]) * (xq - 1.0)
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Helper: tournament selection from fronts
        def tournament_select(fronts, F, X, k=2):
            # Pick k random individuals, return the one from the best front
            candidates = np.random.randint(0, len(F), size=k)
            best_front = -1
            best_idx = candidates[0]
            best_cd = -np.inf
            
            # Get front assignments
            front_assign = np.full(len(F), -1)
            for fi, front in enumerate(fronts):
                for idx in front:
                    front_assign[idx] = fi
            
            # Compute crowding distances per front
            cd_map = {}
            for front in fronts:
                if len(front) > 2:
                    cds = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd_map[idx] = cds[i]
                else:
                    for idx in front:
                        cd_map[idx] = np.inf
            
            for idx in candidates:
                fi = front_assign[idx]
                cd = cd_map.get(idx, 0.0)
                if fi < best_front or (fi == best_front and cd > best_cd):
                    best_front = fi
                    best_idx = idx
                    best_cd = cd
            
            return X[best_idx]

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size:
                if used + 2 > self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select two parents
                p1 = tournament_select(fronts, F, X)
                p2 = tournament_select(fronts, F, X)
                
                # Crossover
                if np.random.random() < 0.9:
                    c1, c2 = sbx_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= self.budget:
                    break
                
                if time.time() > deadline:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if not offspring_X:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            combined_fronts = non_dominated_sort(combined_F)
            
            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to use crowding distance to select
                    remaining = pop_size - len(new_X)
                    cds = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(cds)[::-1][:remaining]
                    for idx in sorted_front:
                        new_X.append(combined_X[front[idx]])
                        new_F.append(combined_F[front[idx]])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 20

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        pop_size = self.pop_size

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size

        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [set() for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
            
            current_front = [p for p in range(n) if n_count[p] == 0]
            ranks[current_front] = 0
            front_idx = 0
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                            ranks[q] = front_idx + 1
                current_front = next_front
                front_idx += 1
            return fronts, ranks

        # Helper: Crowding distance
        def crowding_distance(F, front):
            if len(front) <= 2:
                return np.full(len(front), np.inf)
            
            dist = np.zeros(len(front))
            n_obj = F.shape[1]
            for m in range(n_obj):
                f_vals = F[front, m]
                f_min = np.min(f_vals)
                f_max = np.max(f_vals)
                if f_max == f_min:
                    continue
                # Sort indices by objective m
                sorted_idx = np.argsort(f_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, len(front) - 1):
                    idx = sorted_idx[i]
                    prev_idx = sorted_idx[i-1]
                    next_idx = sorted_idx[i+1]
                    dist[idx] += (F[next_idx, m] - F[prev_idx, m]) / (f_max - f_min)
            return dist

        # Helper: Tournament selection
        def tournament_select(pop_F, pop_X, ranks, crowding):
            # Combine rank and crowding into a fitness measure
            # Lower rank is better. Higher crowding is better.
            # We select 2 random candidates and pick the better one.
            # Better: lower rank, or if same rank, higher crowding.
            
            def better(i, j):
                if ranks[i] < ranks[j]:
                    return True
                if ranks[i] > ranks[j]:
                    return False
                return crowding[i] > crowding[j]

            parent_idx = []
            for _ in range(2):
                cand1 = np.random.randint(len(pop_F))
                cand2 = np.random.randint(len(pop_F))
                if better(cand1, cand2):
                    parent_idx.append(cand1)
                else:
                    parent_idx.append(cand2)
            return pop_X[parent_idx[0]], pop_X[parent_idx[1]]

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, rate=0.9, eta=20.0):
            if np.random.random() > rate:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            mask = np.random.random(dim) < 0.5
            for i in range(dim):
                if not mask[i]:
                    continue
                if x1[i] == x2[i]:
                    continue
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                
                c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            
            return c1, c2

        # Helper: Polynomial Mutation
        def poly_mutate(x, rate=None, eta=20.0):
            if rate is None:
                rate = 1.0 / dim
            x_mut = x.copy()
            for i in range(dim):
                if np.random.random() < rate:
                    u = np.random.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    
                    x_mut[i] += delta * (hi[i] - lo[i])
            x_mut = np.clip(x_mut, lo, hi)
            return x_mut

        # Main Evolution Loop
        while evals < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            Q_X = []
            for _ in range(pop_size):
                if evals + 1 > self.budget:
                    break
                p1, p2 = tournament_select(F, X, *non_dominated_sort(F))
                c1, c2 = sbx_crossover(p1, p2)
                c1 = poly_mutate(c1)
                Q_X.append(c1)
                evals += 1
            
            if not Q_X:
                break
                
            Q_X = np.array(Q_X)
            Q_F = np.array([problem(x) for x in Q_X])
            
            # Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Sort
            fronts, ranks = non_dominated_sort(R_F)
            
            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                else:
                    # Need to sort this front by crowding distance
                    front_F = R_F[front]
                    front_indices = front
                    dists = crowding_distance(front_F, np.arange(len(front)))
                    # Sort by dist descending
                    sorted_order = np.argsort(dists)[::-1]
                    needed = pop_size - len(new_X)
                    for i in range(needed):
                        orig_idx = front_indices[sorted_order[i]]
                        new_X.append(R_X[orig_idx])
                        new_F.append(R_F[orig_idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return None

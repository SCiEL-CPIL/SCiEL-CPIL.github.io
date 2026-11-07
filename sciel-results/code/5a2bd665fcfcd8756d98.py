import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size and generations
        # Ensure total evaluations <= budget
        # Init: pop_size. Each gen: pop_size.
        # Total: pop_size * (1 + n_gen) <= budget
        pop_size = 20
        if pop_size > self.budget:
            pop_size = self.budget
        n_gen = (self.budget - pop_size) // pop_size
        
        if n_gen < 1:
            # If budget is very small, just do init
            n_gen = 0
            pop_size = self.budget

        # Initialize population
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            current_front = []
            k = 0
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
                    
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                current_front = next_front
            return rank

        # Helper: Crowding distance
        def crowding_distance(F, rank, indices_in_front):
            if len(indices_in_front) <= 2:
                return np.full(len(indices_in_front), np.inf)
            
            dist = np.zeros(len(F))
            for m in range(self.n_obj):
                f_vals = F[indices_in_front, m]
                min_val = np.min(f_vals)
                max_val = np.max(f_vals)
                if max_val == min_val:
                    continue
                sorted_idx = np.argsort(f_vals)
                dist[indices_in_front[sorted_idx[0]]] = np.inf
                dist[indices_in_front[sorted_idx[-1]]] = np.inf
                for i in range(1, len(sorted_idx) - 1):
                    idx = indices_in_front[sorted_idx[i]]
                    prev_idx = indices_in_front[sorted_idx[i-1]]
                    next_idx = indices_in_front[sorted_idx[i+1]]
                    dist[idx] += (F[next_idx, m] - F[prev_idx, m]) / (max_val - min_val)
            return dist

        # Helper: Tournament selection
        def tournament_select(X, F, rank, cd, k=2):
            idxs = np.random.randint(0, len(X), k)
            best = 0
            for i in range(1, k):
                if rank[idxs[i]] < rank[idxs[best]]:
                    best = i
                elif rank[idxs[i]] == rank[idxs[best]] and cd[idxs[i]] > cd[idxs[best]]:
                    best = i
            return X[idxs[best]].copy()

        # Helper: SBX Crossover
        def sbx_crossover(p1, p2, eta_c=20.0, crossover_prob=0.9):
            child1 = p1.copy()
            child2 = p2.copy()
            for i in range(self.dim):
                if np.random.rand() > crossover_prob:
                    continue
                if abs(p1[i] - p2[i]) < 1e-14:
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta_c + 1))
                else:
                    beta = (2 * (1 - u)) ** (1 / (eta_c + 1))
                c1 = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2 = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, lo, hi, eta_m=20.0, mutation_prob=1.0/self.dim):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() > mutation_prob:
                    continue
                if x_mut[i] == lo[i] or x_mut[i] == hi[i]:
                    continue
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                u = np.random.rand()
                if u < 0.5:
                    xy = 1 - delta1
                    val = 2 * u + (1 - 2 * u) * (xy) ** (eta_m + 1)
                    deltaq = val ** (1 / (eta_m + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (xy) ** (eta_m + 1)
                    deltaq = 1 - val ** (1 / (eta_m + 1))
                x_mut[i] += deltaq * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main loop
        for g in range(n_gen):
            # Create offspring
            Q = []
            FQ = []
            for _ in range(pop_size // 2):
                p1 = tournament_select(X, F, rank, cd)
                p2 = tournament_select(X, F, rank, cd)
                
                # SBX Crossover
                c1, c2 = sbx_crossover(p1, p2)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                Q.append(c1)
                Q.append(c2)
            
            # Ensure we have exactly pop_size offspring
            # If pop_size is odd, this loop creates pop_size-1 or pop_size+1?
            # pop_size=20. Loop runs 10 times. 2 children per iter. Total 20. Correct.
            
            Q = np.array(Q)
            FQ = np.array([problem(x) for x in Q])
            
            # Combine
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            
            # Sort
            rank = non_dominated_sort(R_F)
            
            # Select next generation
            P_next = []
            F_next = []
            remaining = pop_size
            k = 0
            while remaining > 0:
                front_indices = np.where(rank == k)[0]
                if len(front_indices) <= remaining:
                    P_next.extend(R_X[front_indices])
                    F_next.extend(R_F[front_indices])
                    remaining -= len(front_indices)
                    k += 1
                else:
                    # Need to select subset of this front
                    cd = crowding_distance(R_F, rank, front_indices)
                    sorted_front = front_indices[np.argsort(-cd[front_indices])]
                    selected = sorted_front[:remaining]
                    P_next.extend(R_X[selected])
                    F_next.extend(R_F[selected])
                    remaining = 0
                    break
            
            X = np.array(P_next)
            F = np.array(F_next)
            # Recalculate rank and cd for next iteration's selection
            rank = non_dominated_sort(F)
            cd = np.zeros(len(X))
            for r in np.unique(rank):
                front_idx = np.where(rank == r)[0]
                cd[front_idx] = crowding_distance(F, rank, front_idx)
                
        return X

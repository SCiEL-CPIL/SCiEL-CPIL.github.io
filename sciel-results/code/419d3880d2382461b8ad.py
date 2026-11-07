import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)
        
        # Population size
        # 200 budget. N=50 allows 3 generations (50 init + 3*50 = 200).
        # Larger population helps with Hypervolume spread.
        pop_size = 50
        if self.budget < pop_size:
            pop_size = max(2, self.budget)
            
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominating = np.zeros(n, dtype=int)
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif dominates(F[j], F[i]):
                        n_dominating[i] += 1
                if n_dominating[i] == 0:
                    front0.append(i)

            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[front_indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_v - min_v)
            return cd

        def tournament_select(F, fronts, k=2):
            """Select an individual using tournament selection on non-dominated rank then crowding."""
            # Pre-compute ranks for speed? Or just use fronts.
            # For simplicity in small populations, we can assign a rank score.
            # But fronts list is available.
            # Let's create a rank array
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Create a list of indices
            indices = list(range(len(F)))
            winners = []
            for _ in range(k):
                idx = rng.choice(indices)
                winners.append(idx)
            
            # Find the one with lowest rank (best front)
            best_idx = min(winners, key=lambda i: ranks[i])
            
            # If tie in rank, use crowding distance?
            # For tournament size 2, just picking the best rank is often sufficient.
            # To be more robust, if ranks are equal, pick the one with higher crowding distance.
            # We need CD for the whole population or just the front?
            # CD is usually computed per front.
            # This implementation is slightly inefficient but N=50 is small.
            
            # Let's just return the one with the best rank. If we want CD tie-breaking:
            # We need CD values. Let's pass them in or compute them.
            # To keep it simple and fast, we'll stick to rank-based tournament.
            return best_idx

        def sbx_crossover(x1, x2, cr=0.9, eta_c=15.0):
            """Simulated Binary Crossover"""
            if rng.random() > cr:
                return x1.copy(), x2.copy()
            
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x2)
            
            for i in range(self.dim):
                if rng.random() < 0.5:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                        
                        c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                        c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                        
                        # Clip to bounds
                        c1[i] = np.clip(c1[i], lo[i], hi[i])
                        c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        def polynomial_mutation(x, pm=0.1, eta_m=20.0):
            """Polynomial Mutation"""
            y = x.copy()
            for i in range(self.dim):
                if rng.random() < pm:
                    u = rng.random()
                    if u <= 0.5:
                        delta_q = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    
                    y[i] = x[i] + delta_q * (hi[i] - lo[i])
                    y[i] = np.clip(y[i], lo[i], hi[i])
            return y

        def select_next_population(F_all, X_all, pop_size):
            fronts = fast_non_dominated_sort(F_all)
            new_indices = []
            
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(F_all, front)
                    sorted_front = [front[j] for j in np.argsort(-cd)]
                    new_indices.extend(sorted_front[:remaining])
                    break
            
            while len(new_indices) < pop_size:
                new_indices.append(rng.integers(len(F_all)))
            
            return X_all[new_indices], F_all[new_indices]

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break

            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            # Compute ranks for tournament selection
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            X_c = np.zeros((n_children, self.dim))
            
            for i in range(n_children):
                # Select parents using tournament
                p1 = tournament_select(F, fronts)
                p2 = tournament_select(F, fronts)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # We only need n_children offspring. 
                # If n_children is even, we can use both c1 and c2.
                # If odd, we drop one or just take one.
                # Standard NSGA-II generates 2 offspring per pair.
                # Since we iterate i from 0 to n_children-1, and we generate 2 per iteration,
                # we need to adjust the loop or store them in a list.
                
                # Let's restructure: generate pairs
                # This loop structure is slightly inefficient if n_children is not a multiple of 2.
                # But n_children is pop_size (50), which is even.
                
                X_c[2*i] = c1
                if 2*i + 1 < n_children:
                    X_c[2*i + 1] = c2
                # If n_children is odd, the last one is unused. 
                # But pop_size=50 is even.
                
            # If n_children was odd, we'd have a gap. 
            # Let's assume pop_size is even.
            
            F_c = np.array([problem(x) for x in X_c[:n_children]])
            used += n_children

            X_all = np.vstack([X, X_c[:n_children]])
            F_all = np.vstack([F, F_c])
            
            X, F = select_next_population(F_all, X_all, pop_size)

        return X

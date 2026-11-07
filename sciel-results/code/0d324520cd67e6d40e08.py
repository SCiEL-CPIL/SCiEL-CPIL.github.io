"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with non-dominated front tracking."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Initial population
        pop_size = 30
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """
            Returns a list of arrays, each containing indices of points in that front.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # n_d[i] = number of points dominating i
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
            
            for p in range(n):
                if n_d[p] == 0:
                    fronts[0].append(p)
            
            k = 0
            while fronts[k]:
                next_front = []
                for p in fronts[k]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                k += 1
                fronts.append(next_front)
                
            # Remove the last empty front if any
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        def crowding_distance(front_indices, F):
            """
            Calculate crowding distance for points in a given front.
            """
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            dist = np.zeros(len(front_indices))
            F_front = F[front_indices]
            m = self.n_obj
            
            for i in range(m):
                min_val = np.min(F_front[:, i])
                max_val = np.max(F_front[:, i])
                if max_val - min_val < 1e-10:
                    continue
                sorted_idx = np.argsort(F_front[:, i])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for j in range(1, len(front_indices) - 1):
                    dist[sorted_idx[j]] += (F_front[sorted_idx[j+1], i] - F_front[sorted_idx[j-1], i]) / (max_val - min_val)
            
            return dist

        def tournament_select(pop_X, pop_F, pop_ranks, pop_cd, k=2):
            """
            Select k parents using tournament selection.
            """
            winners = []
            for _ in range(k):
                candidates = np.random.randint(0, len(pop_X), size=k)
                best_idx = candidates[0]
                for c in candidates[1:]:
                    # Compare rank first, then crowding distance
                    if pop_ranks[c] < pop_ranks[best_idx]:
                        best_idx = c
                    elif pop_ranks[c] == pop_ranks[best_idx]:
                        if pop_cd[c] > pop_cd[best_idx]:
                            best_idx = c
                winners.append(pop_X[best_idx])
            return winners

        def sbx_crossover(x1, x2, prob=0.9, eta=20.0):
            if np.random.random() > prob:
                return x1.copy(), x2.copy()
            
            x1, x2 = x1.copy(), x2.copy()
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                if np.random.random() < 0.5:
                    # Determine which is larger
                    if x1[i] > x2[i]:
                        val1, val2 = x2[i], x1[i]
                        swap = True
                    else:
                        val1, val2 = x1[i], x2[i]
                        swap = False
                    
                    if np.random.random() < 0.5:
                        beta = (2 * np.random.random()) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - np.random.random()))) ** (1.0 / (eta + 1.0))
                    
                    # Calculate child
                    child1 = 0.5 * (val1 + val2) - 0.5 * (val2 - val1) * beta
                    # Clamp to bounds
                    child1 = np.clip(child1, lo[i], hi[i])
                    
                    # Assign back
                    if swap:
                        x2[i] = child1
                    else:
                        x1[i] = child1
            return x1, x2

        def polynomial_mutation(x, prob=1.0/self.dim, eta=20.0):
            x = x.copy()
            for i in range(self.dim):
                if np.random.random() < prob:
                    if x[i] == lo[i]:
                        delta = 1.0
                    elif x[i] == hi[i]:
                        delta = -1.0
                    else:
                        if np.random.random() < 0.5:
                            delta = (2 * (x[i] - lo[i]) / (hi[i] - lo[i])) ** (1.0 / (eta + 1.0))
                        else:
                            delta = -1.0 * (2 * (hi[i] - x[i]) / (hi[i] - lo[i])) ** (1.0 / (eta + 1.0))
                    x[i] += delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute ranks and crowding distances for current population
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            cd = np.zeros(len(F))
            
            for i, front in enumerate(fronts):
                if not front: continue
                ranks[front] = i
                front_cd = crowding_distance(front, F)
                for j, idx in enumerate(front):
                    cd[idx] = front_cd[j]

            # Select parents
            p1, p2 = tournament_select(X, F, ranks, cd)
            
            # Crossover
            c1, c2 = sbx_crossover(p1, p2)
            
            # Mutation
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)

            # Check budget before evaluation
            if used + 2 > self.budget:
                # Only evaluate one if only 1 left
                f1 = problem(c1)
                used += 1
                # Update population with just c1
                X_new = np.vstack([X, c1])
                F_new = np.vstack([F, f1])
                
                fronts_new = fast_non_dominated_sort(F_new)
                X_next = []
                F_next = []
                for front in fronts_new:
                    if not front: continue
                    if len(X_next) + len(front) <= pop_size:
                        X_next.extend(X_new[front])
                        F_next.extend(F_new[front])
                    else:
                        need = pop_size - len(X_next)
                        front_cd = crowding_distance(front, F_new)
                        sorted_front = np.argsort(front_cd)[::-1]
                        selected_indices = [front[i] for i in sorted_front[:need]]
                        X_next.extend(X_new[selected_indices])
                        F_next.extend(F_new[selected_indices])
                        break
                
                X = np.array(X_next)
                F = np.array(F_next)
                break
            
            # Evaluate
            f1 = problem(c1)
            f2 = problem(c2)
            used += 2
            
            # Update population
            X_new = np.vstack([X, c1, c2])
            F_new = np.vstack([F, f1, f2])
            
            # Sort and select best pop_size
            fronts_new = fast_non_dominated_sort(F_new)
            
            X_next = []
            F_next = []
            for front in fronts_new:
                if not front: continue
                if len(X_next) + len(front) <= pop_size:
                    X_next.extend(X_new[front])
                    F_next.extend(F_new[front])
                else:
                    need = pop_size - len(X_next)
                    front_cd = crowding_distance(front, F_new)
                    sorted_front = np.argsort(front_cd)[::-1]
                    selected_indices = [front[i] for i in sorted_front[:need]]
                    X_next.extend(X_new[selected_indices])
                    F_next.extend(F_new[selected_indices])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)

        return X
# EVOLVE-BLOCK-END

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        pop_size = 20
        
        # 1. Initialize Population
        X = np.random.uniform(lo, hi, (pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations_used = pop_size
        
        if evaluations_used >= budget:
            return X
            
        # 2. NSGA-II Loop
        while evaluations_used < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Create Offspring
            offspring = np.zeros((pop_size, dim))
            for i in range(pop_size):
                # Tournament Selection for Parent 1
                p1 = self._tournament_select(X, F)
                # Tournament Selection for Parent 2
                p2 = self._tournament_select(X, F)
                
                # Crossover
                if np.random.rand() < 0.9:
                    child1, child2 = self._sbx_crossover(p1, p2, lo, hi)
                else:
                    child1, child2 = p1.copy(), p2.copy()
                
                # Mutation
                child1 = self._poly_mutate(child1, lo, hi)
                child2 = self._poly_mutate(child2, lo, hi)
                
                offspring[i*2] = child1
                offspring[i*2+1] = child2
                
            # Evaluate Offspring
            F_offspring = np.array([problem(x) for x in offspring])
            evaluations_used += pop_size
            
            # Combine
            X_combined = np.vstack((X, offspring))
            F_combined = np.vstack((F, F_offspring))
            
            # Selection
            X, F = self._nsga2_select(X_combined, F_combined, pop_size)
            
        return X

    def _tournament_select(self, X, F, k=2):
        idxs = np.random.choice(len(X), k, replace=False)
        # Find non-dominated in tournament
        # Simple: pick the one with best rank or least dominance
        # For simplicity in tournament, we can just pick the one with best sum or use a quick dominance check
        # Let's use a quick check:
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx]

    def _dominates(self, a, b):
        # a dominates b if a is better in all and strictly better in at least one
        return np.all(a <= b) and np.any(a < b)

    def _sbx_crossover(self, p1, p2, lo, hi):
        # Simulated Binary Crossover
        child1 = np.zeros_like(p1)
        child2 = np.zeros_like(p2)
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                child1[i] = p1[i]
                child2[i] = p2[i]
            else:
                if p1[i] == p2[i]:
                    child1[i] = p1[i]
                    child2[i] = p2[i]
                else:
                    u = np.random.rand()
                    beta = 1.0
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / 16.0)
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / 16.0)
                    
                    child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    child1[i] = np.clip(child1[i], lo[i], hi[i])
                    child2[i] = np.clip(child2[i], lo[i], hi[i])
        return child1, child2

    def _poly_mutate(self, x, lo, hi):
        # Polynomial Mutation
        x_new = x.copy()
        for i in range(len(x)):
            if np.random.rand() < 0.9: # Mutation rate
                x_i = x[i]
                if x_i == lo[i] or x_i == hi[i]:
                    continue
                u = np.random.rand()
                delta1 = (x_i - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_i) / (hi[i] - lo[i])
                rand = np.random.rand()
                mut_pow = 20.0
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (mut_pow + 1.0))
                    deltaq = (val ** (1.0 / (mut_pow + 1.0)) - 1.0)
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (mut_pow + 1.0))
                    deltaq = (1.0 - val ** (1.0 / (mut_pow + 1.0)))
                x_new[i] = x_i + deltaq * (hi[i] - lo[i])
                x_new[i] = np.clip(x_new[i], lo[i], hi[i])
        return x_new

    def _nsga2_select(self, X, F, pop_size):
        # Fast Non-Dominated Sorting
        n = len(X)
        if n <= pop_size:
            return X, F
            
        # Calculate dominance counts and dominated sets
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i+1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[j] += 1
        
        # Fronts
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        rank = np.zeros(n, dtype=int)
        
        front_id = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                rank[i] = front_id
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_id += 1
        
        # Select based on fronts
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= pop_size:
                selected_indices.extend(front)
            else:
                # Crowding distance for this front
                # Calculate crowding distance
                cd = np.zeros(len(front))
                for k in range(self.n_obj):
                    # Sort by objective k
                    sorted_idx = np.argsort(F[front, k])
                    cd_sorted = np.zeros(len(front))
                    if len(front) > 2:
                        cd_sorted[0] = np.inf
                        cd_sorted[-1] = np.inf
                        for m in range(1, len(front)-1):
                            f_curr = F[front[sorted_idx[m]], k]
                            f_prev = F[front[sorted_idx[m-1]], k]
                            f_next = F[front[sorted_idx[m+1]], k]
                            if f_next != f_prev:
                                cd_sorted[m] = (f_next - f_prev)
                            else:
                                cd_sorted[m] = 0
                    # Map back to original indices in front
                    for m in range(len(front)):
                        orig_idx = front[sorted_idx[m]]
                        cd[orig_idx] += cd_sorted[m]
                
                # Sort front by crowding distance descending
                front_sorted = sorted(front, key=lambda x: cd[x], reverse=True)
                needed = pop_size - len(selected_indices)
                selected_indices.extend(front_sorted[:needed])
                break
                
        X_selected = X[selected_indices]
        F_selected = F[selected_indices]
        
        # Shuffle to maintain diversity in next generation parents? 
        # NSGA-II doesn't strictly require shuffling parents, but it helps.
        idx = np.random.permutation(len(X_selected))
        return X_selected[idx], F_selected[idx]

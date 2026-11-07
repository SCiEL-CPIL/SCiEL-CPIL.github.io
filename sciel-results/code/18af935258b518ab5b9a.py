import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 20
        # Adjust pop_size if budget is small
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 2)
            
        # 1. Initialization using LHS (Latin Hypercube Sampling)
        # We need pop_size points
        # Simple LHS: divide range into pop_size bins, pick random in each bin for each dim
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            # Shuffle indices for LHS
            indices = np.arange(pop_size)
            np.random.shuffle(indices)
            # For each bin, pick a random value
            for i in range(pop_size):
                low = lo[j] + (hi[j] - lo[j]) * (indices[i] / pop_size)
                high = lo[j] + (hi[j] - lo[j]) * ((indices[i] + 1) / pop_size)
                X[i, j] = np.random.uniform(low, high)
                
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions
        def dominates(a, b):
            # a dominates b if a <= b and a != b
            return np.all(a <= b) and np.any(a < b)
            
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p != q:
                        if dominates(F[p], F[q]):
                            S[p].append(q)
                        elif dominates(F[q], F[p]):
                            n_count[p] += 1
                if n_count[p] == 0:
                    fronts[0].append(p)
                    
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1] # Remove empty last one

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i, m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                if f_max == f_min:
                    continue
                for k in range(1, n - 1):
                    dist[sorted_indices[k]] += (F[sorted_indices[k+1], m] - F[sorted_indices[k-1], m]) / (f_max - f_min)
            return dist

        def sbx_crossover(x1, x2, lo, hi, eta=15):
            x = np.zeros_like(x1)
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x1[j] - x2[j]) < 1e-14:
                        x[j] = x1[j]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        x[j] = np.clip(x[j], lo[j], hi[j])
                else:
                    x[j] = x1[j]
            return x

        def poly_mutation(x, lo, hi, eta=20):
            x = x.copy()
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x[j] = x[j] + delta * (hi[j] - lo[j])
                    x[j] = np.clip(x[j], lo[j], hi[j])
            return x

        # 2. Evolutionary Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Non-dominated sorting
            fronts = non_dominated_sort(F)
            
            # Select parents
            parents = []
            for front in fronts:
                if len(parents) + len(front) > pop_size:
                    break
                parents.extend(front)
            
            # If we didn't fill parents, fill with random or best
            while len(parents) < pop_size:
                parents.append(np.random.randint(0, pop_size))
                
            # Create children
            children_X = []
            children_F = []
            
            # We need to generate (pop_size) children
            # To respect budget, we generate up to remaining budget
            # But NSGA-II usually generates pop_size children per generation
            # If budget is not multiple of pop_size, we stop when budget runs out
            
            n_children = min(pop_size, self.budget - used)
            
            for _ in range(n_children):
                # Binary tournament
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                # Compare fronts
                # Find front index for p1 and p2
                # Simple: just pick randomly from parents list if we maintained it?
                # Better: tournament on the current population F
                
                # Find front rank
                rank1 = 0
                rank2 = 0
                for i, front in enumerate(fronts):
                    if p1_idx in front:
                        rank1 = i
                        break
                    if p2_idx in front:
                        rank2 = i
                        break
                # If tie, use crowding
                if rank1 < rank2:
                    p1 = p1_idx
                elif rank2 < rank1:
                    p1 = p2_idx
                else:
                    # Crowding distance
                    cd = crowding_distance(F, list(range(pop_size)))
                    if cd[p1_idx] > cd[p2_idx]:
                        p1 = p1_idx
                    else:
                        p1 = p2_idx
                
                # Second parent
                p3_idx = np.random.randint(0, pop_size)
                p4_idx = np.random.randint(0, pop_size)
                rank3 = 0
                rank4 = 0
                for i, front in enumerate(fronts):
                    if p3_idx in front:
                        rank3 = i
                        break
                    if p4_idx in front:
                        rank4 = i
                        break
                if rank3 < rank4:
                    p2 = p3_idx
                elif rank4 < rank3:
                    p2 = p4_idx
                else:
                    cd = crowding_distance(F, list(range(pop_size)))
                    if cd[p3_idx] > cd[p4_idx]:
                        p2 = p3_idx
                    else:
                        p2 = p4_idx
                
                # Crossover
                child_x = sbx_crossover(X[p1], X[p2], lo, hi)
                # Mutation
                child_x = poly_mutation(child_x, lo, hi)
                
                child_f = problem(child_x)
                used += 1
                
                children_X.append(child_x)
                children_F.append(child_f)
                
                if used >= self.budget:
                    break
            
            if not children_X:
                break
                
            # Combine
            X = np.vstack([X, np.array(children_X)])
            F = np.vstack([F, np.array(children_F)])
            
            # Selection
            fronts = non_dominated_sort(F)
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X[idx])
                        new_F.append(F[idx])
                else:
                    # Need to select based on crowding
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    # Wait, crowding_distance returns list aligned with input indices order?
                    # My crowding_distance takes indices and returns dist in same order.
                    # So dist[i] corresponds to index i.
                    # We need to sort front by cd descending.
                    pairs = [(i, cd[front.index(i)]) for i in front]
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    needed = pop_size - len(new_X)
                    for i, _ in pairs[:needed]:
                        new_X.append(X[i])
                        new_F.append(F[i])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X

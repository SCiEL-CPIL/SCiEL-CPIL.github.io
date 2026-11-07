import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 20
        
        # 1. Initialization using LHS
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            indices = np.arange(pop_size)
            np.random.shuffle(indices)
            for i in range(pop_size):
                low = lo[j] + (hi[j] - lo[j]) * (indices[i] / pop_size)
                high = lo[j] + (hi[j] - lo[j]) * ((indices[i] + 1) / pop_size)
                X[i, j] = np.random.uniform(low, high)
                
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Dominance check using NumPy
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
            return fronts[:-1]

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
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

        def poly_mutation(x, lo, hi, eta=20, rate=1.0/self.dim):
            x = x.copy()
            for j in range(self.dim):
                if np.random.rand() < rate:
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
            
            # Compute crowding distance for the entire population for selection
            # We only need it for the front that gets truncated, but computing for all is simpler for small N
            # Actually, we can compute it per front as we select
            
            # Create children
            children_X = []
            children_F = []
            
            n_children = min(pop_size, self.budget - used)
            
            for _ in range(n_children):
                # Binary tournament selection
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                
                # Find ranks
                rank1 = 0
                rank2 = 0
                for i, front in enumerate(fronts):
                    if p1_idx in front:
                        rank1 = i
                        break
                    if p2_idx in front:
                        rank2 = i
                        break
                
                if rank1 < rank2:
                    p1 = p1_idx
                elif rank2 < rank1:
                    p1 = p2_idx
                else:
                    # Crowding distance tie-breaker
                    # Calculate crowding distance for the specific front they are in
                    # To avoid recalculating, we can precompute or just use a simple heuristic
                    # For small pop, recalculating is fine
                    front_indices = [p1_idx, p2_idx]
                    # Find the actual front
                    for fr in fronts:
                        if p1_idx in fr and p2_idx in fr:
                            cd = crowding_distance(F, fr)
                            # cd is aligned with fr
                            idx1 = fr.index(p1_idx)
                            idx2 = fr.index(p2_idx)
                            if cd[idx1] > cd[idx2]:
                                p1 = p1_idx
                            else:
                                p1 = p2_idx
                            break
                    else:
                        p1 = p1_idx

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
                    for fr in fronts:
                        if p3_idx in fr and p4_idx in fr:
                            cd = crowding_distance(F, fr)
                            idx3 = fr.index(p3_idx)
                            idx4 = fr.index(p4_idx)
                            if cd[idx3] > cd[idx4]:
                                p2 = p3_idx
                            else:
                                p2 = p4_idx
                            break
                    else:
                        p2 = p3_idx

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
            
            # Selection: Truncate to pop_size
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
                    # cd is aligned with front
                    pairs = list(zip(front, cd))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    needed = pop_size - len(new_X)
                    for i, _ in pairs[:needed]:
                        new_X.append(X[i])
                        new_F.append(F[i])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Optional: Local Search on the best individual
            # Find the non-dominated front
            fronts = non_dominated_sort(F)
            if fronts:
                best_idx = fronts[0][0]
                # Try a few local moves
                for _ in range(2):
                    if used >= self.budget:
                        break
                    # Random perturbation
                    step = 0.05 * (hi - lo)
                    perturbed = X[best_idx] + np.random.uniform(-step, step)
                    perturbed = np.clip(perturbed, lo, hi)
                    pf = problem(perturbed)
                    used += 1
                    # Accept if it dominates the best or is non-dominated
                    if dominates(pf, F[best_idx]):
                        X[best_idx] = perturbed
                        F[best_idx] = pf
                    elif not dominates(F[best_idx], pf):
                        # Check if pf is non-dominated by the current front
                        is_non_dom = True
                        for i in range(len(F)):
                            if i != best_idx and dominates(F[i], pf):
                                is_non_dom = False
                                break
                        if is_non_dom:
                            # Replace a random dominated point or just keep it if we have space?
                            # For simplicity, if it's non-dominated, we might want to add it, 
                            # but we are at pop_size. We can replace the worst crowding distance point.
                            pass # Skipping complex replacement for now to keep it simple

        return X

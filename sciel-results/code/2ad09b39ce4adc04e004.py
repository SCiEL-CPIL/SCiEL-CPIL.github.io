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
        
        # Population size
        N = 20 
        
        # Initialize
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Create offspring
            # Tournament selection
            parents_idx = []
            for _ in range(N):
                # Pick 2 random, choose better based on rank (approx) or just random for now? 
                # Proper NSGA-II needs ranks.
                pass
            
            # This is getting complex to vectorize fully in Python without overhead.
            # Let's use a simpler, efficient heuristic if NSGA-II is too slow.
            # But NSGA-II is robust.
            
            # Let's implement a streamlined NSGA-II.
            
            # 1. Ranks and Crowding of current population
            ranks, crowding = self._fast_nondominated_sort(F)
            
            # 2. Selection
            # We need to select N parents.
            # Simplified: Use ranks.
            # Sort by rank, then crowding.
            sort_idx = np.lexsort((crowding, ranks))
            # But we need to handle the fact that we generate N offspring and keep best 2N.
            
            # Let's generate offspring first.
            # Crossover: SBX
            # Mutation: Polynomial
            
            # To keep it simple and fast, let's do:
            # 1. Select 2 parents via binary tournament on (rank, crowding).
            # 2. SBX to create child.
            # 3. Mutate.
            # 4. Evaluate.
            
            new_X = []
            new_F = []
            
            for _ in range(N):
                # Tournament
                i1, i2 = np.random.choice(N, 2, replace=False)
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and crowding[i1] > crowding[i2]):
                    p1 = X[i1]
                else:
                    p1 = X[i2]
                
                i1, i2 = np.random.choice(N, 2, replace=False)
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and crowding[i1] > crowding[i2]):
                    p2 = X[i1]
                else:
                    p2 = X[i2]
                
                # SBX
                child = self._sbx(p1, p2, lo, hi)
                # Mutation
                child = self._mutate(child, lo, hi)
                
                new_X.append(child)
                fc = problem(child)
                new_F.append(fc)
                used += 1
                
                if used >= self.budget:
                    break
            
            if not new_X: break
            
            new_X = np.array(new_X)
            new_F = np.array(new_F)
            
            # Combine
            X_comb = np.vstack([X, new_X])
            F_comb = np.vstack([F, new_F])
            
            # Sort combined
            ranks_comb, crowding_comb = self._fast_nondominated_sort(F_comb)
            
            # Select best N
            sort_idx = np.lexsort((crowding_comb, ranks_comb))
            X = X_comb[sort_idx[:N]]
            F = F_comb[sort_idx[:N]]
            
        return X

    def _fast_nondominated_sort(self, F):
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        crowding = np.zeros(N)
        
        # Dominance matrix
        # D[i,j] = 1 if i dominates j
        # F[i] <= F[j] and F[i] < F[j]
        # Vectorized:
        # le = F[:, None, :] <= F[None, :, :]
        # lt = F[:, None, :] < F[None, :, :]
        # dom = le & lt.any(axis=2)
        
        le = F[:, None, :] <= F[None, :, :]
        lt = F[:, None, :] < F[None, :, :]
        dom = le & lt.any(axis=2)
        
        # Note: dom[i,i] is False because lt[i,i] is False.
        
        # Front 0
        front_idx = np.where(dom.sum(axis=0) == 0)[0] # Wait, dom[i,j]=1 means i dominates j.
        # If i dominates j, j has an incoming edge.
        # So j is dominated if any i dominates j.
        # dominated_by = dom.T
        # dominated_count = dominated_by.sum(axis=1)
        
        dominated_count = dom.T.sum(axis=1)
        
        current_front = np.where(dominated_count == 0)[0]
        rank_id = 0
        processed = np.zeros(N, dtype=bool)
        
        while len(current_front) > 0:
            for i in current_front:
                ranks[i] = rank_id
                # Update neighbors
                # Neighbors j that are dominated by i
                neighbors = np.where(dom[i, :])[0]
                for j in neighbors:
                    if not processed[j]:
                        dominated_count[j] -= 1
                        if dominated_count[j] == 0:
                            # j is in next front
                            # We can't easily add to current_front during iteration without messing up
                            # Let's collect next_front
                            pass
            # This loop is tricky. Let's do iterative peeling.
            
            # Actually, simpler:
            # 1. Find non-dominated in remaining set.
            # 2. Remove them.
            # 3. Repeat.
            
            pass # Re-implementing below

    def _fast_nondominated_sort_v2(self, F):
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        crowding = np.zeros(N)
        
        le = F[:, None, :] <= F[None, :, :]
        lt = F[:, None, :] < F[None, :, :]
        dom = le & lt.any(axis=2)
        
        dominated_count = dom.T.sum(axis=1)
        
        remaining = np.arange(N)
        rank_id = 0
        
        while len(remaining) > 0:
            # Find non-dominated in remaining
            sub_dom = dom[np.ix_(remaining, remaining)]
            # A point is non-dominated if it doesn't dominate anyone? No.
            # A point is non-dominated if NO ONE dominates it.
            # dominated_in_remaining = sub_dom.T.sum(axis=1) == 0
            
            # Wait, dom[i,j] means i dominates j.
            # So j is dominated if sum(dom[:, j]) > 0.
            # In the submatrix, j is dominated if sum(sub_dom[:, j]) > 0.
            
            dom_count_sub = sub_dom.T.sum(axis=1)
            front_mask = dom_count_sub == 0
            
            if not np.any(front_mask):
                # Should not happen if there's at least 1 point
                # Assign all remaining to current rank?
                front_idx = remaining
            else:
                front_idx = remaining[front_mask]
                
            ranks[front_idx] = rank_id
            
            # Calculate crowding distance for this front
            # Sort by each objective
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F[front_idx, k])
                sorted_front = front_idx[sorted_idx]
                
                # Distance to neighbors
                # End points get infinity
                if len(sorted_front) > 1:
                    crowding[sorted_front[0]] = np.inf
                    crowding[sorted_front[-1]] = np.inf
                    
                    for i in range(1, len(sorted_front) - 1):
                        idx_curr = sorted_front[i]
                        idx_prev = sorted_front[i-1]
                        idx_next = sorted_front[i+1]
                        
                        dist = (F[idx_next, k] - F[idx_prev, k])
                        if dist > 0:
                            crowding[idx_curr] += 1.0 / dist
                        else:
                            crowding[idx_curr] += 0.0 # or inf? usually 0 if same value
                
            # Remove front from remaining
            remaining = np.setdiff1d(remaining, front_idx)
            rank_id += 1
            
        return ranks, crowding

    def _sbx(self, p1, p2, lo, hi):
        eta = 20
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                if u <= 0.5:
                    alpha = (2*u)**(1/(eta+1))
                else:
                    alpha = (2*(1-u))**(1/(eta+1))
                child[i] = 0.5 * ((1+alpha)*p1[i] + (1-alpha)*p2[i])
            else:
                u = np.random.rand()
                if u <= 0.5:
                    alpha = (2*u)**(1/(eta+1))
                else:
                    alpha = (2*(1-u))**(1/(eta+1))
                child[i] = 0.5 * ((1+alpha)*p2[i] + (1-alpha)*p1[i])
        return np.clip(child, lo, hi)

    def _mutate(self, x, lo, hi):
        eta = 20
        for i in range(len(x)):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u <= 0.5:
                    delta = (2*u)**(1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u))**(1/(eta+1))
                x[i] += delta * (hi[i] - lo[i])
        return np.clip(x, lo, hi)

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Population size
        # Reduced to 12 to allow for a polish phase with remaining budget
        self.pop_size = 12
        self.start_time = time.time()
        self.timeout = 300  # Safety margin for 330s limit

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        
        rng = np.random.default_rng(0)
        
        # Initialization
        X = lo + (hi - lo) * rng.random((self.pop_size, d))
        F = np.array([problem(x) for x in X])
        used = self.pop_size
        
        # Main NSGA-II Loop
        while used + self.pop_size <= self.budget:
            # Check time
            if time.time() - self.start_time > self.timeout:
                break
            
            # Calculate ranks and CD for current population X
            ranks, cd = self._calculate_rank_cd(F)
            
            # Create offspring
            Q = np.zeros((self.pop_size, d))
            FQ = np.zeros((self.pop_size, self.n_obj))
            
            for i in range(self.pop_size):
                p1 = self._tournament(ranks, cd, rng)
                p2 = self._tournament(ranks, cd, rng)
                
                x1 = X[p1]
                x2 = X[p2]
                
                # Crossover: BLX-Alpha
                alpha = 0.5
                low = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
                high = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
                child = rng.uniform(low, high, size=d)
                child = np.clip(child, lo, hi)
                
                # Mutation
                # Gaussian mutation
                for k in range(d):
                    if rng.random() < 1.0/d:
                        child[k] += 0.1 * (hi[k] - lo[k]) * rng.standard_normal()
                        child[k] = np.clip(child[k], lo, hi)
                
                Q[i] = child
                FQ[i] = problem(child)
            
            used += self.pop_size
            
            # Combine and select
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            
            X, F = self._nsga2_select(R_X, R_F)

        # Tail: Polish phase
        # Use remaining budget to refine the best solutions
        # First, ensure X and F are aligned and sorted by quality
        if len(X) > 0:
            # Get non-dominated points for polishing targets
            # Simple: Just pick the first few from the sorted list (best fronts)
            ranks, cd = self._calculate_rank_cd(F)
            # Sort indices by rank asc, then cd desc
            sorted_indices = np.lexsort((-cd, ranks))
            # Take top min(remaining, 5) individuals as seeds for local search
            n_seeds = min(used < self.budget and (self.budget - used) or 0, 5)
            # Actually, just loop through remaining budget
            remaining = self.budget - used
            if remaining > 0:
                # Select top individuals for local search
                # We will pick a random individual from the best front for each eval
                best_front_indices = sorted_indices[:min(self.pop_size, 5)]
                
                for _ in range(remaining):
                    if time.time() - self.start_time > self.timeout:
                        break
                    
                    # Pick a seed
                    seed_idx = best_front_indices[rng.integers(len(best_front_indices))]
                    x_seed = X[seed_idx]
                    f_seed = F[seed_idx]
                    
                    # Local search: small perturbation
                    # Sigma decreases as we get closer to end? Or just fixed small
                    sigma = 0.05 * (hi - lo)
                    x_new = x_seed + rng.standard_normal(d) * sigma
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    # Check if f_new is better than f_seed or non-dominated in population
                    # Simple strategy: If f_new dominates f_seed, replace.
                    # Else, if f_new is non-dominated in the whole population, add/replace worst.
                    # To keep it simple and fast:
                    # If f_new dominates f_seed, replace.
                    if np.all(f_new <= f_seed) and np.any(f_new < f_seed):
                        X[seed_idx] = x_new
                        F[seed_idx] = f_new
                    else:
                        # Check if f_new is non-dominated in the current population
                        # If yes, replace the worst dominated point (or just the worst sum)
                        # To avoid O(N^2) check every step, we can just do a simple check against the front
                        # Or just replace the worst point if f_new is better than the worst point
                        worst_idx = np.argmax(np.sum(F, axis=1))
                        if np.all(f_new <= F[worst_idx]) and np.any(f_new < F[worst_idx]):
                            X[worst_idx] = x_new
                            F[worst_idx] = f_new
                        # Else discard (waste eval) - this is suboptimal but safe
                        # Better: Just add to a temporary list and re-select?
                        # Given small remaining, let's just try to insert if non-dominated
                        # Check dominance against all
                        dominated = False
                        for j in range(len(F)):
                            if np.all(F[j] <= f_new) and np.any(F[j] < f_new):
                                dominated = True
                                break
                        if not dominated:
                            # Remove points dominated by f_new
                            new_X = []
                            new_F = []
                            for j in range(len(F)):
                                if not (np.all(f_new <= F[j]) and np.any(f_new < F[j])):
                                    new_X.append(X[j])
                                    new_F.append(F[j])
                            new_X.append(x_new)
                            new_F.append(f_new)
                            # If we have more than pop_size, trim
                            if len(new_X) > self.pop_size:
                                R_X = np.array(new_X)
                                R_F = np.array(new_F)
                                X, F = self._nsga2_select(R_X, R_F)
                            else:
                                X = np.array(new_X)
                                F = np.array(new_F)

        return X

    def _calculate_rank_cd(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        cd = np.zeros(n)
        
        # Non-dominated sorting
        remaining = set(range(n))
        current_rank = 0
        
        while remaining:
            front_indices = []
            for i in list(remaining):
                dominated = False
                for j in remaining:
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front_indices.append(i)
                    remaining.remove(i)
            
            if not front_indices:
                break
                
            for i in front_indices:
                ranks[i] = current_rank
            
            if len(front_indices) > 1:
                for m in range(self.n_obj):
                    order = np.argsort(F[front_indices, m])
                    sorted_indices = [front_indices[k] for k in order]
                    
                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf
                    
                    span = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                    if span > 0:
                        for k in range(1, len(sorted_indices)-1):
                            idx_curr = sorted_indices[k]
                            idx_prev = sorted_indices[k-1]
                            idx_next = sorted_indices[k+1]
                            cd[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / span
            else:
                for i in front_indices:
                    cd[i] = np.inf
            
            current_rank += 1
            
        return ranks, cd

    def _tournament(self, ranks, cd, rng):
        i1 = rng.integers(len(ranks))
        i2 = rng.integers(len(ranks))
        
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            elif cd[i2] > cd[i1]:
                return i2
            else:
                return i1

    def _nsga2_select(self, R_X, R_F):
        ranks, cd = self._calculate_rank_cd(R_F)
        
        # Sort by rank asc, then cd desc
        # np.lexsort keys: last is primary.
        # Primary: ranks (asc)
        # Secondary: -cd (asc) -> cd desc
        sorted_indices = np.lexsort((-cd, ranks))
        
        selected_idx = sorted_indices[:self.pop_size]
        
        return R_X[selected_idx], R_F[selected_idx]

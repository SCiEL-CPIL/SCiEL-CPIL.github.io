import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 10s before 300s limit
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # 20 is a good balance for budget 200
        pop_size = 20
        
        # Initial Population: Latin Hypercube Sampling
        rng = np.random.default_rng(42)
        X = np.zeros((pop_size, dim))
        for j in range(dim):
            idx = np.arange(pop_size)
            rng.shuffle(idx)
            X[idx, j] = (idx + 0.5) / pop_size
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # NSGA-II Loop
        while evals + pop_size <= budget:
            if time.time() > deadline:
                break
                
            # 1. Selection and Crossover
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))
            
            # Pre-compute ranks and distances for selection
            ranks = self.non_dominated_sorting(F)
            distances = self.crowding_distance(F, ranks)
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                    
                # Tournament Selection
                idx1, idx2 = self.tournament_select(ranks, distances, rng)
                p1 = X[idx1]
                p2 = X[idx2]
                
                # Crossover
                child = self.sbx_crossover(p1, p2, lo, hi, rng)
                
                # Mutation
                child = self.poly_mutation(child.reshape(1, -1), lo, hi, rng).flatten()
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                evals += 1
            
            if time.time() > deadline:
                break

            # 2. Combine and Select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            combined_ranks = self.non_dominated_sorting(combined_F)
            combined_distances = self.crowding_distance(combined_F, combined_ranks)
            
            # Sort by rank (asc), then distance (desc)
            # We want the best pop_size
            # Create sort keys
            # Rank is integer, Distance is float.
            # We can use lexsort.
            # lexsort takes keys in reverse order of priority.
            # Priority 1: Rank (lowest is best)
            # Priority 2: Distance (highest is best) -> so we sort by -Distance
            sort_idx = np.lexsort((-combined_distances, combined_ranks))
            
            X = combined_X[sort_idx[:pop_size]]
            F = combined_F[sort_idx[:pop_size]]
            
        return X

    def tournament_select(self, ranks, distances, rng, k=2):
        """Select k individuals, return the best one's index."""
        # For k=2, just pick 2 and compare
        idxs = rng.integers(0, len(ranks), k)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if ranks[idx] < ranks[best_idx]:
                best_idx = idx
            elif ranks[idx] == ranks[best_idx]:
                if distances[idx] > distances[best_idx]:
                    best_idx = idx
        return best_idx

    def sbx_crossover(self, x1, x2, lo, hi, rng, beta=15):
        """SBX Crossover for two individuals."""
        # x1, x2 are 1D arrays of size dim
        dim = len(x1)
        # Crossover rate 0.9
        mask = rng.random(dim) < 0.9
        
        child = x1.copy()
        if not np.any(mask):
            return child
            
        u = rng.random(dim)
        # Calculate betaq
        betaq = np.zeros(dim)
        mask_u = u <= 1.0 / (beta + 1)
        betaq[mask_u] = (2 * u[mask_u]) ** (1.0 / (beta + 1))
        betaq[~mask_u] = (1.0 / (2 * (1.0 - u[~mask_u]))) ** (1.0 / (beta + 1))
        
        # Swap mask
        swap_mask = rng.random(dim) < 0.5
        
        # Compute xq
        xq1 = 0.5 * ((1 + betaq) * x1 + (1 - betaq) * x2)
        xq2 = 0.5 * ((1 + betaq) * x2 + (1 - betaq) * x1)
        
        xq = xq1.copy()
        xq[swap_mask] = xq2[swap_mask]
        
        child[mask] = xq[mask]
        child = np.clip(child, lo, hi)
        return child

    def poly_mutation(self, X, lo, hi, rng, eta=20):
        """Polynomial Mutation for a population (or single individual)."""
        # X is (N, dim)
        N, dim = X.shape
        X = X.copy()
        
        # Mutation rate 1/dim
        mask = rng.random((N, dim)) < 1.0 / dim
        
        for i in range(N):
            if not np.any(mask[i]):
                continue
                
            x = X[i]
            m = mask[i]
            
            u = rng.random(dim)
            
            # Calculate deltaq for all dims
            deltaq = np.zeros(dim)
            
            # Case 1: u < 0.5
            mask_u1 = u < 0.5
            if np.any(mask_u1):
                # deltaq = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                deltaq[mask_u1] = 1 - (2 * (1 - u[mask_u1])) ** (1.0 / (eta + 1))
            
            # Case 2: u >= 0.5
            mask_u2 = u >= 0.5
            if np.any(mask_u2):
                # deltaq = 1 - (2 * u) ** (1.0 / (eta + 1))
                deltaq[mask_u2] = 1 - (2 * u[mask_u2]) ** (1.0 / (eta + 1))
            
            # Apply: x = x + deltaq * (hi - lo)
            # Note: This formula assumes the standard Deb definition where 
            # the sign is handled by the deltaq value being positive or negative 
            # relative to the direction? 
            # Actually, in Deb's book:
            # if u < 0.5: x = x + (hi-lo)*deltaq
            # else: x = x - (hi-lo)*deltaq
            # My deltaq calculation above:
            # u<0.5: 1 - (2(1-u))^(1/(eta+1)). Since u<0.5, 1-u>0.5, 2(1-u)>1. (2(1-u))^... > 1. So 1 - >1 is negative.
            # So x + negative * (hi-lo) -> x decreases.
            # But standard says x increases for u<0.5?
            # Let's check Deb & Jain 2002.
            # if u < 0.5:
            #   deltaq = (1 - (2*(1-u))**(1/(eta+1)))
            #   x = x + deltaq * (hi - lo)
            # Since (2*(1-u)) > 1, deltaq is negative. So x decreases.
            # Wait, usually low u means small change?
            # If u=0, deltaq = 1 - 2^(1/(eta+1)) < 0.
            # So x decreases.
            # If u=0.5, deltaq = 0.
            # If u=1, deltaq = 1 - 2^(1/(eta+1)) < 0.
            # This seems to always decrease or stay same? That's wrong.
            
            # Correct Formula:
            # if u < 0.5:
            #   deltaq = (1 - (2*(1-u))**(1/(eta+1)))
            # else:
            #   deltaq = 1 - (2*u)**(1/(eta+1))
            # 
            # Let's re-evaluate u=0.
            # u=0 < 0.5. deltaq = 1 - 2^(1/(eta+1)). Negative.
            # x = x + negative * (hi-lo). x decreases.
            # 
            # Let's re-evaluate u=1.
            # u=1 >= 0.5. deltaq = 1 - 2^(1/(eta+1)). Negative.
            # x = x - negative * (hi-lo) = x + positive * (hi-lo). x increases.
            # 
            # So u close to 0 -> decrease. u close to 1 -> increase.
            # This is correct. The sign is handled by the subtraction in the else branch.
            
            # So my vectorized code needs to apply the sign correctly.
            # For u < 0.5: x += deltaq * (hi-lo)
            # For u >= 0.5: x -= deltaq * (hi-lo)  <-- WAIT.
            # In the code above, I calculated deltaq for both cases as `1 - ...`.
            # For u >= 0.5, deltaq is negative.
            # If I do x += deltaq * (hi-lo), it decreases.
            # But I want it to increase for u close to 1.
            # So for u >= 0.5, I should do x -= deltaq * (hi-lo) OR define deltaq differently.
            
            # Let's stick to the explicit logic to avoid sign errors.
            
            x_new = x.copy()
            # Apply for u < 0.5
            idx1 = np.where(mask_u1 & m)[0]
            if len(idx1) > 0:
                dq1 = 1 - (2 * (1 - u[idx1])) ** (1.0 / (eta + 1))
                x_new[idx1] += dq1 * (hi[idx1] - lo[idx1])
            
            # Apply for u >= 0.5
            idx2 = np.where(mask_u2 & m)[0]
            if len(idx2) > 0:
                dq2 = 1 - (2 * u[idx2]) ** (1.0 / (eta + 1))
                # dq2 is negative.
                # We want to increase x.
                # x = x - dq2 * (hi - lo)
                x_new[idx2] -= dq2 * (hi[idx2] - lo[idx2])
                
            X[i] = np.clip(x_new, lo, hi)
            
        return X

    def non_dominated_sorting(self, F):
        """Fast Non-Dominated Sorting. Returns ranks array."""
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        # S[i] = set of individuals dominated by i
        # n[i] = number of individuals dominating i
        
        # For simplicity and speed with N<=40, we can use O(N^2) approach
        # Or implement standard fast sorting.
        # Given N is small (40 max), O(N^2) is fine.
        
        # Initialize
        S = [set() for _ in range(N)]
        n = np.zeros(N, dtype=int)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n[i] += 1 # Wait, n[i] is number of individuals dominated by i? 
                    # Standard definition: n_p is number of solutions that dominate p.
                    # So if i dominates j, then j is dominated by i.
                    # n[j] += 1
                    pass
        
        # Let's restart the logic clearly.
        # n[j] = count of i such that i dominates j.
        # S[i] = set of j such that i dominates j.
        
        S = [set() for _ in range(N)]
        n = np.zeros(N, dtype=int)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n[j] += 1
        
        # Front 1
        front = [i for i in range(N) if n[i] == 0]
        rank = 0
        ranks = np.zeros(N, dtype=int)
        
        while front:
            for i in front:
                ranks[i] = rank
            next_front = []
            for i in front:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
            front = next_front
            rank += 1
            
        return ranks

    def crowding_distance(self, F, ranks):
        """Calculate crowding distance for each individual."""
        N = len(F)
        distances = np.zeros(N)
        
        # Group by rank
        unique_ranks = np.unique(ranks)
        
        for r in unique_ranks:
            idxs = np.where(ranks == r)[0]
            if len(idxs) <= 2:
                distances[idxs] = float('inf')
                continue
                
            # Sort by each objective
            for k in range(self.n_obj):
                sorted_idxs = idxs[np.argsort(F[idxs, k])]
                # Assign inf to best and worst
                distances[sorted_idxs[0]] = float('inf')
                distances[sorted_idxs[-1]] = float('inf')
                
                # Calculate range
                f_min = F[sorted_idxs[0], k]
                f_max = F[sorted_idxs[-1], k]
                range_k = f_max - f_min
                if range_k < 1e-10:
                    continue
                    
                for i in range(1, len(sorted_idxs) - 1):
                    idx_curr = sorted_idxs[i]
                    idx_prev = sorted_idxs[i - 1]
                    idx_next = sorted_idxs[i + 1]
                    distances[idx_curr] += (F[idx_next, k] - F[idx_prev, k]) / range_k
                    
        return distances

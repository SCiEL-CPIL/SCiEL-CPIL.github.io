"""Rewritten program using Multi-Objective Differential Evolution (MO-DE) with NSGA-II selection."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Differential Evolution optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        
        # Population size
        # We need to fit generations into the budget.
        # Init: N
        # Each gen: N trials.
        # Total: N * (1 + G) <= budget
        # Let's aim for N=40, G=4 -> 40*5 = 200.
        N = 40
        if N * 5 > budget:
             N = budget // 5
             if N < 10: N = 10
             
        # Initialize Population
        X = lo + (hi - lo) * rng.random((N, n))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # DE Parameters
        F_mut = 0.8  # Mutation scale
        CR = 0.5     # Crossover rate
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(Fs):
            """Fast non-dominated sorting."""
            m = len(Fs)
            S = [[] for _ in range(m)]
            n_dom = [0] * m
            rank = [0] * m
            front = [0]
            
            for p in range(m):
                for q in range(m):
                    if p == q: continue
                    if dominates(Fs[p], Fs[q]):
                        S[p].append(q)
                    elif dominates(Fs[q], Fs[p]):
                        n_dom[p] += 1
                
                if n_dom[p] == 0:
                    rank[p] = 0
                    front[0] += 1
            
            i = 0
            while front[i] > 0:
                next_front = []
                for p in range(m):
                    if rank[p] == i:
                        for q in S[p]:
                            n_dom[q] -= 1
                            if n_dom[q] == 0:
                                rank[q] = i + 1
                                next_front.append(q)
                i += 1
                front.append(len(next_front))
                
            # Create list of fronts
            fronts = [[] for _ in range(i+1)]
            for p in range(m):
                if rank[p] <= i:
                    fronts[rank[p]].append(p)
            return fronts

        def crowding_distance(Fs, indices):
            """Calculate crowding distance for a set of indices in Fs."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            cd = {i: 0.0 for i in indices}
            m_obj = self.n_obj
            
            for k in range(m_obj):
                # Sort indices by k-th objective
                sorted_idx = sorted(indices, key=lambda i: Fs[i][k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                f_min = Fs[sorted_idx[0]][k]
                f_max = Fs[sorted_idx[-1]][k]
                f_range = f_max - f_min
                
                if f_range > 1e-12:
                    for i in range(1, len(sorted_idx) - 1):
                        idx_curr = sorted_idx[i]
                        idx_prev = sorted_idx[i-1]
                        idx_next = sorted_idx[i+1]
                        cd[idx_curr] += (Fs[idx_next][k] - Fs[idx_prev][k]) / f_range
                else:
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] = float('inf')
                        
            return cd

        # Main Loop
        while used < budget:
            # Generate Trial Vectors
            trial_X = np.empty_like(X)
            trial_F = np.empty_like(F)
            
            for i in range(N):
                if used >= budget:
                    break
                
                # Select r1, r2, r3 distinct from i
                r1, r2, r3 = rng.choice(N, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(N, size=3, replace=False)
                    
                # Mutation
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Crossover
                cross_mask = rng.random(n) < CR
                # Ensure at least one gene is copied from v
                if not np.any(cross_mask):
                    j = rng.integers(n)
                    cross_mask[j] = True
                    
                u = np.where(cross_mask, v, X[i])
                u = np.clip(u, lo, hi)
                
                trial_X[i] = u
                trial_F[i] = problem(u)
                used += 1
            
            # Combine Population
            # Parents: 0..N-1, Offspring: N..2N-1
            combined_X = np.vstack([X, trial_X])
            combined_F = np.vstack([F, trial_F])
            
            # Sort and Select
            fronts = non_dominated_sort(combined_F)
            
            new_X = []
            new_F = []
            i = 0
            while i < len(fronts) and len(new_X) < N:
                front = fronts[i]
                if len(new_X) + len(front) <= N:
                    # Take all from this front
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    cd = crowding_distance(combined_F, front)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda idx: cd[idx], reverse=True)
                    needed = N - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
                i += 1
                
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

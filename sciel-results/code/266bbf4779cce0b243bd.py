"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """SPEA2 (Strength Pareto Evolutionary Algorithm 2) for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        N = 20
        if self.budget < N + N:
            N = max(10, self.budget // 2)
        
        # Offspring size
        M = 10
        if self.budget < N + M:
            M = max(5, self.budget - N)
            
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def spea2_select(X_pop, F_pop, N_target):
            n = len(X_pop)
            # 1. Strength calculation
            strength = np.zeros(n)
            # 2. Fitness calculation
            fitness = np.zeros(n)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if dominates(F_pop[i], F_pop[j]):
                        strength[i] += 1
                        fitness[j] += 1 # Note: standard SPEA2 sums strengths of dominators
                    # Actually, fitness[i] = sum(strength[j] for j where j dominates i)
            
            # Correct SPEA2 Fitness:
            # R_i = count of solutions dominated by i
            # F_i = sum of R_j for all j that dominate i
            
            R = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and dominates(F_pop[i], F_pop[j]):
                        R[i] += 1
                        
            F_fit = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and dominates(F_pop[j], F_pop[i]):
                        F_fit[i] += R[j]
                        
            # 3. Selection
            # Identify non-dominated set (F_fit == 0)
            nd_indices = [i for i in range(n) if F_fit[i] == 0]
            
            if len(nd_indices) > N_target:
                # Use crowding distance
                # Calculate crowding distance for nd_indices
                cd = np.zeros(n)
                for k in range(n_obj):
                    f_vals = F_pop[nd_indices, k]
                    sorted_idx = np.argsort(f_vals)
                    if len(nd_indices) > 2:
                        cd[nd_indices[sorted_idx[0]]] = float('inf')
                        cd[nd_indices[sorted_idx[-1]]] = float('inf')
                        f_range = f_vals.max() - f_vals.min()
                        if f_range > 0:
                            for m in range(1, len(nd_indices) - 1):
                                idx = nd_indices[sorted_idx[m]]
                                prev_idx = nd_indices[sorted_idx[m-1]]
                                next_idx = nd_indices[sorted_idx[m+1]]
                                cd[idx] += (F_pop[next_idx, k] - F_pop[prev_idx, k]) / f_range
                # Select N_target with highest crowding distance
                selected_nd = sorted(nd_indices, key=lambda i: cd[i], reverse=True)[:N_target]
                return selected_nd
            else:
                # Keep all non-dominated
                selected = list(nd_indices)
                remaining = N_target - len(selected)
                if remaining > 0:
                    # Select from dominated set based on lowest fitness
                    dom_indices = [i for i in range(n) if i not in selected]
                    dom_indices.sort(key=lambda i: F_fit[i])
                    selected.extend(dom_indices[:remaining])
                return selected

        # Main loop
        while used + M <= self.budget:
            # Generate M offspring
            X_off = np.zeros((M, dim))
            for k in range(M):
                # DE/rand/1
                i1, i2, i3 = rng.integers(0, N, 3)
                F_factor = 0.8
                CR = 0.9
                v = X[i1] + F_factor * (X[i2] - X[i3])
                
                # Crossover
                r = rng.integers(0, dim)
                mask = rng.random(dim) < CR
                mask[r] = True
                u = np.where(mask, v, X[i1])
                
                u = np.clip(u, lo, hi)
                X_off[k] = u
                
            # Evaluate offspring
            F_off = np.array([problem(x) for x in X_off], dtype=float)
            used += M
            
            # Combine
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            # Select
            selected_idx = spea2_select(X_comb, F_comb, N)
            
            X = X_comb[selected_idx]
            F = F_comb[selected_idx]
            
        # If we have remaining budget, use it for more offspring or local search?
        # For simplicity, just stop.
        
        # Extract Pareto front from final population
        front_indices = []
        for i in range(N):
            dominated = False
            for j in range(N):
                if i == j: continue
                if dominates(F[j], F[i]):
                    dominated = True
                    break
            if not dominated:
                front_indices.append(i)
                
        if not front_indices:
            front_indices = list(range(N))
            
        return X[front_indices]
# EVOLVE-BLOCK-END

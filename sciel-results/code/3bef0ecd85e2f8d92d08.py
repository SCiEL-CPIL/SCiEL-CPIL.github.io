"""Optimized program for WFG9 using a hybrid NSGA-II + local search approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Objective Optimizer combining NSGA-II with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.RandomState(42)
        
        # Phase 1: NSGA-II to find a good spread of solutions
        pop_size = 40
        max_gen = 5
        
        # Initialize population
        X = lo + rng.rand(pop_size, self.dim) * (hi - lo)
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # NSGA-II loop
        for gen in range(max_gen):
            if time.time() > deadline - 30:  # Reserve time for local search
                break
            
            # Combine parents and offspring
            # Create offspring using binary tournament selection and crossover
            offspring_X = []
            while len(offspring_X) < pop_size and evals < self.budget - 20:
                # Binary tournament selection
                i1, i2 = rng.choice(pop_size, 2, replace=False)
                if F[i1][0] + F[i1][1] + F[i1][2] < F[i2][0] + F[i2][1] + F[i2][2]:
                    p1 = X[i1]
                else:
                    p1 = X[i2]
                
                i1, i2 = rng.choice(pop_size, 2, replace=False)
                if F[i1][0] + F[i1][1] + F[i1][2] < F[i2][0] + F[i2][1] + F[i2][2]:
                    p2 = X[i1]
                else:
                    p2 = X[i2]
                
                # Simulated binary crossover (SBX)
                child = self._sbx_crossover(p1, p2, lo, hi, rng)
                offspring_X.append(child)
                evals += 1
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += len(offspring_X)
            
            # Combine and sort
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Fast non-dominated sorting
            ranks = self._fast_non_dominated_sort(F_all)
            
            # Crowding distance
            cd = self._crowding_distance(F_all)
            
            # Select next generation
            next_X = []
            next_F = []
            for rank in range(len(ranks)):
                indices = np.where(ranks == rank)[0]
                if len(next_X) + len(indices) <= pop_size:
                    next_X.extend([X_all[i] for i in indices])
                    next_F.extend([F_all[i] for i in indices])
                else:
                    # Sort by crowding distance and take top ones
                    cd_rank = cd[indices]
                    sorted_idx = indices[np.argsort(-cd_rank)]
                    needed = pop_size - len(next_X)
                    for i in sorted_idx[:needed]:
                        next_X.append(X_all[i])
                        next_F.append(F_all[i])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        # Phase 2: Local search refinement around best solutions
        # Sort by sum of objectives to find best solutions
        sums = F.sum(axis=1)
        sorted_idx = np.argsort(sums)
        
        # Take top 10 solutions for local refinement
        top_n = min(10, len(X))
        top_idx = sorted_idx[:top_n]
        
        local_step = 0.1
        for idx in top_idx:
            if evals >= self.budget or time.time() > deadline - 5:
                break
            
            best_x = X[idx].copy()
            best_f = F[idx].copy()
            
            # Coordinate descent
            for coord in range(self.dim):
                if evals >= self.budget or time.time() > deadline - 5:
                    break
                
                for direction in [-1, 1]:
                    step = local_step
                    for _ in range(3):
                        if evals >= self.budget or time.time() > deadline - 5:
                            break
                        
                        trial_x = best_x.copy()
                        trial_x[coord] = np.clip(trial_x[coord] + direction * step, lo[coord], hi[coord])
                        
                        trial_f = problem(trial_x)
                        evals += 1
                        
                        if trial_f.sum() < best_f.sum():
                            best_x = trial_x
                            best_f = trial_f
                            step *= 0.5
                        else:
                            break
            
            X[idx] = best_x
            F[idx] = best_f
        
        return X

    def _sbx_crossover(self, p1, p2, lo, hi, rng):
        """Simulated Binary Crossover."""
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.rand() < 0.5:
                child[i] = p1[i]
            else:
                child[i] = p2[i]
        
        # Apply mutation
        for i in range(len(child)):
            if rng.rand() < 0.2:
                range_i = hi[i] - lo[i]
                child[i] += rng.randn() * range_i * 0.1
                child[i] = np.clip(child[i], lo[i], hi[i])
        
        return child

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)]  # Dominating sets
        n_dominated = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dominated[i] += 1
        
        # First front
        front = []
        for i in range(n):
            if n_dominated[i] == 0:
                front.append(i)
                ranks[i] = 0
        
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
            front = next_front
            k += 1
        
        return ranks

    def _crowding_distance(self, F):
        """Calculate crowding distance."""
        n = len(F)
        cd = np.zeros(n)
        
        if n <= 2:
            cd[:] = float('inf')
            return cd
        
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_idx = np.argsort(F[:, m])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            
            f_min = F[sorted_idx[0], m]
            f_max = F[sorted_idx[-1], m]
            
            if f_max == f_min:
                continue
            
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / (f_max - f_min)
        
        return cd

    def _dominates(self, a, b):
        """Check if a dominates b."""
        return np.all(a <= b) and np.any(a < b)
# EVOLVE-BLOCK-END

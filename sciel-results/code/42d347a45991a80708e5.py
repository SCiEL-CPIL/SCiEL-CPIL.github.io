"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with Crowding Distance-based selection for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size: balance exploration and exploitation within budget
        pop_size = 30
        max_generations = self.budget // pop_size
        
        # Initialize population uniformly
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_non_dominated_sort(fitnesses):
            """Fast non-dominated sorting for multi-objective optimization."""
            n = len(fitnesses)
            if n == 0:
                return [[]]
            
            S = [[] for _ in range(n)]  # S[i] = list of individuals dominated by i
            n_dominates = [0] * n      # n_dominates[i] = number of individuals dominating i
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(fitnesses[i] <= fitnesses[j]) and np.any(fitnesses[i] < fitnesses[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(fitnesses[j] <= fitnesses[i]) and np.any(fitnesses[j] < fitnesses[i]):
                        S[j].append(i)
                        n_dominates[i] += 1
            
            # First front
            front = []
            for i in range(n):
                if n_dominates[i] == 0:
                    rank[i] = 0
                    front.append(i)
            
            current_rank = 0
            fronts = [front]
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_rank += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            return fronts, rank
        
        def crowding_distance(fitnesses, indices):
            """Calculate crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            cd = [0.0] * n
            m = len(fitnesses[indices[0]]) if len(fitnesses) > 0 else 1
            
            for k in range(m):
                sorted_idx = sorted(range(n), key=lambda i: fitnesses[indices[i]][k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                f_min = fitnesses[indices[sorted_idx[0]]][k]
                f_max = fitnesses[indices[sorted_idx[-1]]][k]
                f_range = f_max - f_min
                
                if f_range > 1e-10:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (
                            fitnesses[indices[sorted_idx[i + 1]]][k] - 
                            fitnesses[indices[sorted_idx[i - 1]]][k]
                        ) / f_range
            
            return cd
        
        # Main DE loop
        Fp = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        for gen in range(max_generations):
            if used >= self.budget:
                break
            
            # Sort population by non-dominated fronts and crowding distance
            fronts, rank = fast_non_dominated_sort(F)
            
            # Create trial vectors using DE/rand/1
            new_X = X.copy()
            new_F = F.copy()
            
            for i in range(pop_size):
                if used >= self.budget:
                    break
                
                # Select three distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    if len(set([i, r1, r2, r3])) == 4:
                        break
                
                # DE mutation
                v = X[r1] + Fp * (X[r2] - X[r3])
                
                # Binary crossover
                j_rand = rng.integers(self.dim)
                trial = X[i].copy()
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                used += 1
                
                # Replacement: keep if it dominates or is non-dominated with better crowding
                # Simple replacement: replace if trial is better in at least one objective and not worse in others
                # Or use a simple rule: replace if trial dominates current, or if current dominates trial keep current
                # For simplicity, use tournament-like selection
                dominates_trial = np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)
                dominates_current = np.all(f_trial <= F[i]) and np.any(f_trial < F[i])
                
                if not dominates_trial:
                    # Trial is not dominated by current
                    new_X[i] = trial
                    new_F[i] = f_trial
                else:
                    # Current dominates trial, keep current
                    new_X[i] = X[i]
                    new_F[i] = F[i]
            
            X = new_X
            F = new_F
        
        return X
# EVOLVE-BLOCK-END

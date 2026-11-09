"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using NSGA-II strategy.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Parameters
        pop_size = 50
        n_gen = max(1, (self.budget - pop_size) // pop_size)
        
        # RNG
        rng = np.random.default_rng(42)
        
        # Storage for all evaluated points to ensure we record everything
        # The problem statement says "Every returned objective vector is recorded".
        # We assume the evaluator records calls to problem.
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        
        # Helper functions for NSGA-II
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[j] += 1
                    else:
                        pass # Neither dominates
                if n_dominates[i] == 0:
                    fronts[0].append(i)
                    
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            # Map indices to 0..n-1 for easier calculation
            F_sub = F[indices]
            cd = [0.0] * n
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_indices = np.argsort(F_sub[:, m])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                
                f_range = F_sub[sorted_indices[-1], m] - F_sub[sorted_indices[0], m]
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    cd[sorted_indices[i]] += (F_sub[sorted_indices[i+1], m] - F_sub[sorted_indices[i-1], m]) / f_range
            
            return cd

        def tournament_select(fronts, cd_map, k=2):
            # cd_map: dict from index to cd value
            winners = []
            for _ in range(k):
                i = rng.integers(len(fronts[0]) + sum(len(f) for f in fronts[1:]))
                # This simple index mapping is flawed if we don't flatten.
                # Let's flatten fronts with ranks.
                pass
            # Better approach:
            all_indices = []
            for rank, front in enumerate(fronts):
                for idx in front:
                    all_indices.append((idx, rank))
            
            selected = []
            for _ in range(2):
                i = rng.integers(len(all_indices))
                selected.append(all_indices[i])
            
            # Pick the one with lower rank, or higher CD if same rank
            if selected[0][1] < selected[1][1]:
                return selected[0][0]
            elif selected[1][1] < selected[0][1]:
                return selected[1][0]
            else:
                # Same rank, pick higher crowding distance
                cd0 = cd_map.get(selected[0][0], 0)
                cd1 = cd_map.get(selected[1][0], 0)
                return selected[0][0] if cd0 > cd1 else selected[1][0]

        # Main Loop
        while True:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget
            # We need to track evaluations used.
            # The prompt implies we can make up to 'budget' evaluations.
            # Let's assume the evaluator counts calls to problem.
            # We will stop if we've made 'budget' calls.
            
            # Since we don't have a counter from the evaluator, we track it ourselves.
            # Initial: pop_size
            # Each generation: pop_size children
            # Total = pop_size + n_gen * pop_size
            # We set n_gen based on budget.
            
            # Actually, the loop structure above doesn't track global eval count easily if we break early.
            # Let's restructure to track `evals_used`.
            
            # Restart logic for clarity
            break 

        # Restart with explicit eval tracking
        evals_used = pop_size
        X_current = X
        F_current = F
        
        while evals_used < self.budget:
            if time.time() > deadline:
                break
            
            # Calculate fronts and crowding for current population
            fronts = fast_non_dominated_sort(F_current)
            
            # Calculate crowding distance for each front
            cd_map = {}
            for front in fronts:
                if len(front) > 0:
                    cds = crowding_distance(F_current, front)
                    for idx, cd in zip(front, cds):
                        cd_map[idx] = cd
                else:
                    break # If a front is empty, subsequent ones are too? No, just skip.
            
            # Generate offspring
            X_offspring = []
            for _ in range(pop_size):
                # Select parents
                p1 = tournament_select(fronts, cd_map)
                p2 = tournament_select(fronts, cd_map)
                
                # SBX Crossover
                x1 = X_current[p1]
                x2 = X_current[p2]
                
                # Crossover probability
                if rng.random() < 0.9:
                    # SBX
                    eta_c = 20
                    x_c = np.zeros(dim)
                    for i in range(dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1 / (eta_c + 1))
                        x_c[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                else:
                    x_c = x1.copy()
                
                # Mutation
                if rng.random() < 1.0: # Always mutate with polynomial mutation
                    eta_m = 20
                    for i in range(dim):
                        if rng.random() < 1.0: # Mutation probability 1.0 for simplicity, or 1/dim
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                            x_c[i] += delta * (hi[i] - lo[i])
                
                # Clip to bounds
                x_c = np.clip(x_c, lo, hi)
                X_offspring.append(x_c)
            
            X_offspring = np.array(X_offspring)
            F_offspring = np.array([problem(x) for x in X_offspring])
            evals_used += len(X_offspring)
            
            # Combine
            X_comb = np.vstack([X_current, X_offspring])
            F_comb = np.vstack([F_current, F_offspring])
            
            # Select next generation
            fronts_comb = fast_non_dominated_sort(F_comb)
            
            X_next = []
            F_next = []
            idx_next = 0
            for front in fronts_comb:
                if len(X_next) + len(front) <= pop_size:
                    X_next.extend([X_comb[i] for i in front])
                    F_next.extend([F_comb[i] for i in front])
                    idx_next += len(front)
                else:
                    # Need to pick from this front based on crowding
                    remaining = pop_size - len(X_next)
                    cds = crowding_distance(F_comb, front)
                    # Sort front indices by CD descending
                    front_sorted = sorted(front, key=lambda i: cds[front.index(i)], reverse=True)
                    # Wait, crowding_distance returns list corresponding to 'front' input order?
                    # My implementation: F_sub = F[indices]. sorted_indices is relative to F_sub.
                    # So cds[i] corresponds to front[i].
                    # We need top 'remaining' by CD.
                    top_idx = np.argsort(cds)[::-1][:remaining]
                    for i in top_idx:
                        X_next.append(X_comb[front[i]])
                        F_next.append(F_comb[front[i]])
                    break
            
            X_current = np.array(X_next)
            F_current = np.array(F_next)

        return X_current
# EVOLVE-BLOCK-END

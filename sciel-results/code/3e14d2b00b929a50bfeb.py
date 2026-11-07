"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using NSGA-II style selection with WFG4-aware initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The task says 300s limit. We use 280s to be safe.
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        range_ = hi - lo
        
        # Population size
        # With budget 200, a pop size of 20 allows for 10 full generations (20 evals per gen).
        # This is a good balance for convergence and diversity.
        pop_size = 20
        
        # Initialize population
        # For WFG4, x[0] determines the position on the front.
        # We initialize x[0] uniformly to ensure coverage of the entire front range.
        # Other variables are initialized randomly.
        X = np.zeros((pop_size, self.dim))
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * rng.random(pop_size)
        X[:, 1:] = lo[1:] + range_[1:] * rng.random((pop_size, self.dim - 1))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume calculation
        all_F = list(F)
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            """
            Returns a list of lists, where each inner list contains indices of solutions
            in the same non-dominated front.
            """
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dominated = np.zeros(n, dtype=int)  # n_dominated[i] = count of solutions dominated by i
            rank = np.zeros(n, dtype=int)
            
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
            
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            current_rank = 0
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = current_rank
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                current_front = next_front
                current_rank += 1
                
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of solutions given by indices.
            """
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            F_sub = F[indices]
            cd = np.zeros(n)
            
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                denom = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if denom == 0:
                    continue
                
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / denom
            
            return cd

        # Main Evolutionary Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Generate Offspring
            # We generate pop_size offspring
            # Selection: Tournament based on Rank + Crowding
            # First, we need ranks and crowding distances for the current population
            
            # Optimization: Only compute full sort if population is large or we are in early stages?
            # With N=20, full sort is cheap.
            
            fronts = non_dominated_sort(F)
            ranks = np.zeros(pop_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
            
            # Calculate crowding distance for the first front (usually the most important)
            # For simplicity and speed with small N, we calculate CD for all fronts or just the best?
            # Standard NSGA-II uses CD only for the front that needs truncation.
            # Here, we select parents for offspring generation.
            
            # We will perform tournament selection from the current population
            # Tournament size 2
            
            offspring_X = np.zeros((pop_size, self.dim))
            
            # To ensure diversity in offspring generation, we can bias towards the best front
            # but use crowding distance to pick parents from the same front.
            
            # For simplicity and robustness with small budget:
            # Pick parents using a tournament that prefers lower rank, then higher CD.
            
            # Pre-compute CD for the best front to guide selection?
            # Actually, for parent selection, we can just use the ranks.
            # Let's use a simple tournament: pick 2 random, compare rank. If tie, compare CD.
            
            # We need CD for all solutions to break ties.
            # Calculating CD for all N=20 solutions is fast.
            # However, CD is typically defined per front.
            # Let's assign CD=0 to dominated solutions for selection purposes, or just use Rank.
            # Using Rank alone is often sufficient for small populations to drive convergence.
            # To maintain diversity, we can add a small random perturbation or use CD on the best front.
            
            # Let's compute CD for the best front (fronts[0])
            cd = np.zeros(pop_size)
            if fronts:
                best_front_indices = fronts[0]
                cd_best = crowding_distance(F, best_front_indices)
                for k, idx in enumerate(best_front_indices):
                    cd[idx] = cd_best[k]
            
            # Generate offspring
            for i in range(pop_size):
                # Tournament Selection
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                
                # Compare: lower rank is better. If same rank, higher CD is better.
                if ranks[p1_idx] < ranks[p2_idx]:
                    parent = X[p1_idx]
                elif ranks[p2_idx] < ranks[p1_idx]:
                    parent = X[p2_idx]
                else:
                    if cd[p1_idx] >= cd[p2_idx]:
                        parent = X[p1_idx]
                    else:
                        parent = X[p2_idx]
                
                # Perturbation
                # Adaptive step size: start large, decrease
                progress = used / self.budget
                step_size = 0.1 * (1.0 - progress) + 0.01 * progress
                
                child = parent + step_size * range_ * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
            
            # Evaluate Offspring
            F_offspring = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Record offspring
            all_F.extend(list(F_offspring))
            
            # 2. Selection for Next Generation
            # Combine Parent and Offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, F_offspring])
            
            # Non-dominated Sort
            fronts = non_dominated_sort(F_combined)
            
            # Select best pop_size solutions
            X_new = np.zeros((pop_size, self.dim))
            F_new = np.zeros((pop_size, self.n_obj))
            selected_indices = []
            
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    # Take all from this front
                    selected_indices.extend(front)
                else:
                    # Need to truncate this front based on crowding distance
                    n_needed = pop_size - len(selected_indices)
                    cd_front = crowding_distance(F_combined, front)
                    
                    # Sort by CD descending
                    sorted_order = np.argsort(-cd_front)
                    for k in range(n_needed):
                        selected_indices.append(front[sorted_order[k]])
                    break
            
            X_new = X_combined[selected_indices]
            F_new = F_combined[selected_indices]
            
            X = X_new
            F = F_new
            
            # Check if we have used all budget
            if used >= self.budget:
                break

        return all_F
# EVOLVE-BLOCK-END

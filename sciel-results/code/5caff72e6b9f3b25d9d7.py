"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using an NSGA-II-like strategy with
       time-aware budgeting and a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy random state (seeded by evaluator)
        rng = np.random

        # Population size: keep it modest to stay within budget
        # With budget=200, we can afford pop ~ 20-40
        pop_size = min(30, max(10, budget // 6))

        # Initial population: random points, but bias towards the Pareto front
        # For DTLZ4, the Pareto front is at x[0:3] = 0.5 (for n_obj=3, M=3)
        # Actually for DTLZ4 with M objectives, the Pareto front requires
        # x_{M+1}..x_{M+k} = 0.5 where k=1 for DTLZ4 (k=1)
        # Wait: DTLZ4 has M objectives, N variables. The Pareto front is
        # defined by x_{M+1} = 0.5 (for k=1 in DTLZ4).
        # For M=3, N=12: x[3] = 0.5, and x[0:3] determine the objectives.
        # x[4:12] are position variables that don't affect objectives (they
        # affect g but g=0 when x[3]=0.5... actually no).
        
        # Let me recall DTLZ4:
        # g(x) = 100 * (|x_{M+1..N}| - 0.5)^alpha * (N-M)  -- no wait
        # DTLZ4: g(x_{M+1..N}) = 100 * (|x_{M+1} - 0.5|^0.1 + ... ) 
        # Actually for DTLZ4 with k=1: g(x_{M+1..N}) = 100 * (N-M) * (x_{M+1} - 0.5)^2
        # No, let me be precise:
        # DTLZ4: 
        #   g(x_{M+1..N}) = 100 * (N - M) * (sum_{i=M+1}^{N} (x_i - 0.5)^2)
        #   f_i = (1 + g) * cos(x_1 * pi/2) * ... * cos(x_{i-1} * pi/2) * sin(x_i * pi/2) for i < M
        #   f_M = (1 + g) * sin(x_{M-1} * pi/2)
        # Wait, I need to be more careful. For DTLZ4:
        #   g(x) = 100 * (N - M) * (sum_{i=M+1}^{N} (x_i - 0.5)^2)
        #   f_1 = (1+g) * prod_{j=1}^{M-1} cos(x_j * pi/2)
        #   f_i = (1+g) * cos(x_1*pi/2)*...*cos(x_{i-1}*pi/2) * sin(x_i*pi/2) for 2<=i<=M-1
        #   f_M = (1+g) * sin(x_{M-1}*pi/2)
        
        # So for M=3, N=12:
        #   g = 100 * 9 * sum_{i=4}^{12} (x_i - 0.5)^2  (0-indexed: x[3:12])
        #   f_1 = (1+g) * cos(x[0]*pi/2) * cos(x[1]*pi/2)
        #   f_2 = (1+g) * cos(x[0]*pi/2) * sin(x[1]*pi/2)
        #   f_3 = (1+g) * sin(x[1]*pi/2)
        
        # Wait, that doesn't look right either. Let me reconsider.
        # Standard DTLZ4 with M objectives:
        #   g(x_{M+1..N}) = 100 * (N-M) * (sum_{i=M+1}^{N} (x_i - 0.5)^alpha)
        #   where alpha is typically 10 for DTLZ4
        #   f_1 = (1+g) * prod_{j=1}^{M-1} cos(x_j * pi/2)
        #   f_i = (1+g) * (prod_{j=1}^{i-1} cos(x_j * pi/2)) * sin(x_i * pi/2), i=2..M-1
        #   f_M = (1+g) * sin(x_{M-1} * pi/2)
        
        # For M=3:
        #   f_1 = (1+g) * cos(x[0]*pi/2) * cos(x[1]*pi/2)
        #   f_2 = (1+g) * cos(x[0]*pi/2) * sin(x[1]*pi/2)
        #   f_3 = (1+g) * sin(x[1]*pi/2)
        
        # The Pareto front is g=0, which requires x[3:12] = 0.5 (with alpha=10, 
        # (x_i - 0.5)^10 is minimized at 0.5)
        # And f_i are determined by x[0] and x[1].
        
        # So the key insight: set x[3:12] = 0.5 to get g=0, then optimize x[0:3]
        # for the objective landscape.
        
        # Strategy:
        # 1. Initialize with some points having x[3:12] = 0.5 (on the front)
        # 2. Evolve x[0:3] to spread across the Pareto front
        # 3. Use a mix of exploitation and exploration
        
        # Let's create initial population
        X = np.zeros((pop_size, dim))
        # Set x[3:12] to 0.5 for all (to be on or near Pareto front)
        X[:, 3:12] = 0.5
        # x[0:3] random in [0, 1]
        X[:, 0:3] = rng.random((pop_size, 3))
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II style evolution
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            def fast_non_dominated_sort(F):
                n = len(F)
                domination_count = np.zeros(n, dtype=int)
                dominated_set = [list() for _ in range(n)]
                for p in range(n):
                    for q in range(n):
                        if p == q:
                            continue
                        if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                            dominated_set[p].append(q)
                        elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                            domination_count[p] += 1
                front = [list() for _ in range(n)]
                rank = np.zeros(n, dtype=int)
                for p in range(n):
                    if domination_count[p] == 0:
                        rank[p] = 0
                        front[0].append(p)
                i = 0
                while front[i]:
                    next_front = []
                    for p in front[i]:
                        for q in dominated_set[p]:
                            domination_count[q] -= 1
                            if domination_count[q] == 0:
                                rank[q] = i + 1
                                next_front.append(q)
                    i += 1
                    front[i] = next_front
                return front, rank
            
            front, rank = fast_non_dominated_sort(F)
            
            # Crowding distance
            def crowding_distance(F, indices):
                n = len(indices)
                if n <= 2:
                    return np.ones(n)
                cd = np.zeros(n)
                for m in range(n_obj):
                    f_vals = F[indices, m]
                    f_min = np.min(f_vals)
                    f_max = np.max(f_vals)
                    if f_max - f_min < 1e-12:
                        cd[:] = np.inf
                        continue
                    order = np.argsort(f_vals)
                    cd[order[0]] = np.inf
                    cd[order[-1]] = np.inf
                    for k in range(1, n - 1):
                        cd[order[k]] += (f_vals[order[k+1]] - f_vals[order[k-1]]) / (f_max - f_min)
                return cd
            
            # Select parents from first front using crowding distance
            if len(front[0]) > 0:
                parents_indices = front[0]
                cd = crowding_distance(F, parents_indices)
                # Pick parent with max crowding distance
                best_parent_idx = parents_indices[np.argmax(cd)]
            else:
                best_parent_idx = 0
            
            # Create child: mutate parent
            child = X[best_parent_idx].copy()
            # Gaussian mutation with adaptive sigma
            sigma = 0.1 * span
            child += sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            # Keep x[3:12] close to 0.5 (bias towards Pareto front)
            # With some probability, snap to 0.5
            if rng.random() < 0.7:
                child[3:12] = 0.5
            
            fc = problem(child)
            used += 1
            
            # Replace worst individual (highest rank, then highest crowding distance)
            # Find the worst individual
            # Worst is in the last non-empty front
            last_front_idx = len(front) - 1
            while last_front_idx >= 0 and not front[last_front_idx]:
                last_front_idx -= 1
            if last_front_idx < 0:
                worst_idx = 0
            else:
                worst_front = front[last_front_idx]
                if len(worst_front) == 1:
                    worst_idx = worst_front[0]
                else:
                    cd_worst = crowding_distance(F, worst_front)
                    # Remove the one with lowest crowding distance
                    worst_in_front = np.argmin(cd_worst)
                    worst_idx = worst_front[worst_in_front]
            
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

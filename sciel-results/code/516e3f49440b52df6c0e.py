"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Track all evaluated points
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            eval_count += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # NSGA-II Parameters
        # Population size N. With budget 200, we can afford roughly 10-20 generations.
        # Let's try N=20, which allows ~10 generations (20 init + 10*20 offspring = 220? No, budget is 200).
        # Init: 20.
        # Gen 1: 20 offspring. Total 40.
        # Gen 2: 20 offspring. Total 60.
        # ...
        # Gen 9: 20 offspring. Total 200.
        # So 9 generations with N=20 fits exactly.
        N = 20
        n_generations = (self.budget - N) // N
        
        # 1. Initialize Population
        P = np.random.uniform(lo, hi, size=(N, dim))
        F = np.array([evaluate(x) for x in P])
        
        # 2. Evolutionary Loop
        for gen in range(n_generations):
            if eval_count >= self.budget or time.time() > deadline:
                break

            # Create Offspring Q
            Q = np.empty_like(P)
            F_Q = np.empty_like(F)
            
            for i in range(N):
                if eval_count >= self.budget:
                    break
                    
                # Selection: Binary Tournament
                idx1, idx2 = np.random.choice(N, 2, replace=False)
                # Compare fitness using non-domination rank and crowding distance
                # We need to maintain a combined set for selection? 
                # Standard NSGA-II selects from P (current non-dominated set).
                # Since P is already the non-dominated set from previous step (or subset), 
                # we just pick from P.
                
                # For simplicity and speed, we can just pick randomly from P 
                # or implement a quick tournament. 
                # Given N=20, a full tournament is cheap.
                # However, implementing full NSGA-II selection (rank + crowding) inside the loop
                # is complex. A simpler "Elitist" approach:
                # Keep the best N from {P, Q}.
                
                # Let's use a simple crossover and mutation
                # SBX (Simulated Binary Crossover) or Uniform Crossover
                # Let's use Uniform Crossover for simplicity and robustness
                parent1, parent2 = P[idx1], P[idx2]
                
                # Uniform Crossover
                mask = np.random.random(dim) < 0.5
                child = np.where(mask, parent1, parent2)
                
                # Polynomial Mutation / Gaussian Mutation
                # Simple Gaussian mutation with adaptive sigma
                sigma = 0.1 * span
                child = child + np.random.normal(0, sigma, dim)
                child = np.clip(child, lo, hi)
                
                Q[i] = child
                F_Q[i] = evaluate(child)

            # Combine P and Q
            R_X = np.vstack([P, Q])
            R_F = np.vstack([F, F_Q])
            
            # Non-dominated Sorting on R
            # Returns list of fronts, where fronts[0] is the best non-dominated set
            fronts = self._fast_non_dominated_sort(R_F)
            
            # Select next population P_new
            P_new = []
            F_new = []
            
            for front in fronts:
                # If adding this entire front exceeds N, we need to select from it
                if len(P_new) + len(front) <= N:
                    P_new.extend(front)
                    F_new.extend(front)
                else:
                    # Need to select (N - len(P_new)) individuals from this front
                    # based on Crowding Distance
                    needed = N - len(P_new)
                    # Calculate crowding distance for this front
                    cd = self._crowding_distance(R_F[front])
                    # Sort by crowding distance descending
                    sorted_idx = np.argsort(-cd)
                    selected_idx = front[sorted_idx[:needed]]
                    P_new.extend(selected_idx)
                    F_new.extend(selected_idx)
                    break
                    
            P = R_X[P_new]
            F = R_F[F_new]

        return all_X

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns a list of arrays, where each array contains indices of individuals in that front.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices of individuals dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of individuals dominating i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        current_front = np.where(n_dom == 0)[0]
        if len(current_front) > 0:
            fronts.append(current_front)
        
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if len(next_front) > 0:
                fronts.append(np.array(next_front))
            current_front = next_front
            
        return fronts

    def _crowding_distance(self, F_front):
        """
        Calculate crowding distance for a set of non-dominated points.
        F_front: array of shape (k, n_obj)
        Returns: array of shape (k,)
        """
        k = len(F_front)
        if k <= 2:
            return np.full(k, np.inf)
            
        dist = np.zeros(k)
        for m in range(F_front.shape[1]):
            # Sort by objective m
            sorted_idx = np.argsort(F_front[:, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            
            f_min = F_front[sorted_idx[0], m]
            f_max = F_front[sorted_idx[-1], m]
            f_range = f_max - f_min
            if f_range == 0:
                continue
                
            for i in range(1, k - 1):
                idx = sorted_idx[i]
                dist[idx] += (F_front[sorted_idx[i + 1], m] - F_front[sorted_idx[i - 1], m]) / f_range
                
        return dist
# EVOLVE-BLOCK-END

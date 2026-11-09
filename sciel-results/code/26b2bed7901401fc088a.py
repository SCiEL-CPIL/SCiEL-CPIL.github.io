"""Multi-objective optimizer for WFG9: structured sweep + local refinement."""
import numpy as np
import time


class Optimizer:
    """Structured sweep along primary decision variables + Gaussian refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def now():
            return time.time()

        # State for tracking all evaluated points
        all_X = []  # list of x vectors
        all_F = []  # list of F vectors

        budget_used = 0

        def eval_point(x):
            nonlocal budget_used
            if budget_used >= budget or now() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            budget_used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # ---- NSGA-II Implementation ----
        
        # Population size: small enough to allow many generations, large enough for diversity
        # 200 budget / 20 pop = 10 generations roughly (if we evaluate pop each gen)
        # But we evaluate offspring + parents? 
        # Standard NSGA-II: Pop P (N), Mates (N), Offspring Q (N). 
        # Eval P once. Then loop: create Q (N evals), combine P+Q (2N), select N.
        # Total evals = N (initial) + G * N (offspring).
        # If N=20, G=9 -> 20 + 180 = 200. Perfect.
        
        N = 20
        
        # Initialize population
        rng = np.random.default_rng(42)
        P = np.random.uniform(lo, hi, (N, dim))
        
        # Evaluate initial population
        for i in range(N):
            f = eval_point(P[i])
            if f is None:
                break
        
        if budget_used == 0:
            return np.empty((0, dim))
            
        # Calculate fitness for initial population
        # We need to maintain a list of (x, f) for the current population
        # But we also need to track *all* points for the final score.
        # The `all_X` and `all_F` lists handle the global tracking.
        # For NSGA-II selection, we only care about the current P and the new Q.
        
        # Helper for Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            """
            F: numpy array of shape (n, m)
            Returns: list of fronts, where each front is a list of indices
            """
            n = len(F)
            if n == 0:
                return []
            
            S = [set() for _ in range(n)]
            n_dom = [0] * n
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
            
            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            if not front:
                return []
            fronts.append(front)
            
            while front:
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                front = next_front
            
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for points in front_indices.
            F: numpy array of shape (n, m)
            front_indices: list of indices
            Returns: dict mapping index to distance
            """
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            distances = {i: 0.0 for i in front_indices}
            m = F.shape[1]
            
            for k in range(m):
                # Sort by objective k
                sorted_indices = sorted(front_indices, key=lambda i: F[i, k])
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                
                if f_max - f_min < 1e-10:
                    continue
                    
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                for i in range(1, len(sorted_indices) - 1):
                    idx_prev = sorted_indices[i-1]
                    idx_curr = sorted_indices[i]
                    idx_next = sorted_indices[i+1]
                    distances[idx_curr] += (F[idx_next, k] - F[idx_prev, k]) / (f_max - f_min)
            
            return distances

        def select_parents(pop, fitnesses):
            """
            Selection using tournament selection based on non-dominated rank and crowding distance.
            pop: list of x vectors
            fitnesses: list of f vectors
            Returns: two parent indices
            """
            n = len(pop)
            if n == 0:
                return -1, -1
            
            F = np.array(fitnesses)
            fronts = fast_non_dominated_sort(F)
            
            # Assign ranks
            ranks = [0] * n
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Calculate crowding distance for each front
            crowding = {i: 0.0 for i in range(n)}
            for front in fronts:
                dists = crowding_distance(F, front)
                for idx in front:
                    crowding[idx] = dists[idx]
            
            def compare(i, j):
                if ranks[i] != ranks[j]:
                    return ranks[i] < ranks[j] # Lower rank is better
                else:
                    return crowding[i] > crowding[j] # Higher crowding is better
            
            def tournament():
                idx1 = rng.integers(n)
                idx2 = rng.integers(n)
                if compare(idx1, idx2):
                    return idx1
                else:
                    return idx2
            
            p1 = tournament()
            p2 = tournament()
            return p1, p2

        def crossover(x1, x2):
            """SBX Crossover"""
            eta = 20.0
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            
            for i in range(dim):
                if rng.random() < 0.5:
                    # Swap
                    x1_i, x2_i = x2[i], x1[i]
                    # Swap children roles for this gene? No, standard SBX produces two children.
                    # Let's just generate children based on x1, x2
                    pass
                
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            
            return child1, child2

        def mutation(x):
            """Polynomial Mutation"""
            eta_m = 20.0
            for i in range(dim):
                if rng.random() < 0.5:
                    span_i = hi[i] - lo[i]
                    if span_i < 1e-10:
                        continue
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    x[i] += delta_q * span_i
            
            return np.clip(x, lo, hi)

        # Main Loop
        current_pop_X = P.copy()
        current_pop_F = [f for f in all_F] # Initial N points
        
        while budget_used < budget and now() < deadline:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(N):
                if budget_used >= budget or now() > deadline:
                    break
                
                p1_idx, p2_idx = select_parents(current_pop_X, current_pop_F)
                
                if p1_idx == -1:
                    break
                    
                x1 = current_pop_X[p1_idx]
                x2 = current_pop_X[p2_idx]
                
                c1, c2 = crossover(x1, x2)
                c1 = mutation(c1)
                c2 = mutation(c2)
                
                f1 = eval_point(c1)
                if f1 is None:
                    break
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if budget_used >= budget or now() > deadline:
                    break
                
                f2 = eval_point(c2)
                if f2 is None:
                    break
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if not offspring_X:
                break
            
            # Combine parents and offspring
            combined_X = np.vstack([current_pop_X, np.array(offspring_X)])
            combined_F = np.array(current_pop_F + offspring_F)
            
            # Sort combined population
            F = combined_F
            fronts = fast_non_dominated_sort(F)
            
            # Select next population
            next_pop_indices = []
            for front in fronts:
                if len(next_pop_indices) + len(front) <= N:
                    next_pop_indices.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    needed = N - len(next_pop_indices)
                    dists = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: dists[i], reverse=True)
                    next_pop_indices.extend(sorted_front[:needed])
                    break
            
            if not next_pop_indices:
                break
                
            current_pop_X = combined_X[next_pop_indices]
            current_pop_F = [combined_F[i] for i in next_pop_indices]
            
            # Check if we ran out of budget or time
            if budget_used >= budget or now() > deadline:
                break

        # Return all evaluated points
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, dim))

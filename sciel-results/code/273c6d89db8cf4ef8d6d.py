"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II Multi-objective Optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # NSGA-II parameters
        # Population size: 40 is a good balance for 4D, 3Obj
        pop_size = 40
        # Crossover and mutation rates
        crossover_rate = 0.9
        mutation_rate = 1.0 / dim
        # Polynomial mutation exponent
        eta_m = 20
        
        # 1. Initialization: Latin Hypercube Sampling for diversity
        # Use a portion of the budget for initial population if possible, 
        # but usually we just generate random points within bounds.
        # However, to ensure we use evaluations wisely, we evaluate the initial pop.
        
        X_pop = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F_pop = np.zeros((pop_size, n_obj))
        eval_count = 0
        
        for i in range(pop_size):
            if time.time() > deadline or eval_count >= self.budget:
                break
            try:
                F_pop[i] = problem(X_pop[i])
                eval_count += 1
            except Exception:
                F_pop[i] = np.full(n_obj, 1e6)
        
        # If we ran out of budget during initialization, return what we have
        if eval_count >= self.budget:
            return None
            
        # Helper functions for NSGA-II

        def fast_non_dominated_sort(F):
            """
            Returns a list of fronts. Each front is a list of indices.
            F: (N, n_obj) array
            """
            N = len(F)
            S = [[] for _ in range(N)]  # S[i] = indices dominated by i
            n = [0] * N  # n[i] = number of solutions dominated by i
            rank = [0] * N  # rank[i] = non-dominated front rank
            
            for p in range(N):
                for q in range(N):
                    if p == q: continue
                    # Check if p dominates q
                    # p dominates q if p <= q in all objectives and p < q in at least one
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
                        # q dominates p, so p is in S[q] (handled in next loop or symmetric check)
                        # Actually, let's just count dominators for p
                        pass 
                    # Wait, standard implementation:
                    # if p dominates q: S[p].append(q), n[q] += 1
                    # if q dominates p: S[q].append(p), n[p] += 1
                    # Let's rewrite for clarity and correctness
                    
            # Re-implementation for correctness
            S = [[] for _ in range(N)]
            n = [0] * N
            rank = [0] * N
            
            for p in range(N):
                for q in range(N):
                    if p == q: continue
                    # Does p dominate q?
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n[q] += 1
                    # Does q dominate p?
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].append(p)
                        n[p] += 1
                        # Note: The 'elif' above is slightly inefficient but correct logic 
                        # if we assume no mutual domination. 
                        # Actually, we should check both independently.
                        
            # Corrected loop:
            S = [[] for _ in range(N)]
            n = [0] * N
            rank = [0] * N
            
            for p in range(N):
                for q in range(N):
                    if p == q: continue
                    # p dominates q?
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n[q] += 1
                    # q dominates p?
                    if np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].append(p)
                        n[p] += 1

            front = 0
            current_front = [i for i in range(N) if n[i] == 0]
            for i in current_front:
                rank[i] = front
            
            while current_front:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                            rank[q] = front + 1
                front += 1
                current_front = next_front
            return rank

        def crowding_distance(F, rank):
            """
            Calculate crowding distance for each point.
            F: (N, n_obj)
            rank: list of ranks for each point
            Returns array of distances
            """
            N = len(F)
            dist = np.zeros(N)
            
            # Group by rank
            ranks = set(rank)
            for r in ranks:
                indices = [i for i in range(N) if rank[i] == r]
                if len(indices) <= 2:
                    # Edge of front or small front
                    for i in indices:
                        dist[i] = float('inf')
                    continue
                
                # Sort by each objective and assign distances
                for k in range(n_obj):
                    # Sort indices by objective k
                    sorted_indices = sorted(indices, key=lambda i: F[i, k])
                    dist[sorted_indices[0]] = float('inf')
                    dist[sorted_indices[-1]] = float('inf')
                    
                    f_min = F[sorted_indices[0], k]
                    f_max = F[sorted_indices[-1], k]
                    if f_max - f_min == 0:
                        continue
                        
                    for i in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[i]
                        idx_prev = sorted_indices[i-1]
                        idx_next = sorted_indices[i+1]
                        dist[idx] += (F[idx_next, k] - F[idx_prev, k]) / (f_max - f_min)
                        
            return dist

        def select_parents(F, rank, dist):
            """
            Tournament selection.
            Returns two parents.
            """
            N = len(F)
            
            def tournament(k=2):
                candidates = np.random.choice(N, size=k, replace=False)
                best = candidates[0]
                for c in candidates[1:]:
                    # Compare rank first
                    if rank[c] < rank[best]:
                        best = c
                    elif rank[c] == rank[best]:
                        # Then compare crowding distance (higher is better)
                        if dist[c] > dist[best]:
                            best = c
                return best

            p1 = tournament()
            p2 = tournament()
            return p1, p2

        def crossover(x1, x2):
            """
            Simulated Binary Crossover (SBX).
            """
            if np.random.rand() > crossover_rate:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for j in range(dim):
                if np.random.rand() < 0.5:
                    if x1[j] == x2[j]:
                        continue
                    if x1[j] > x2[j]:
                        x1[j], x2[j] = x2[j], x1[j]
                        
                    beta = 1.0
                    if np.random.rand() < 0.5:
                        beta = (2 * (x1[j] - lo[j]) / (hi[j] - lo[j])) ** (1.0 / (eta_m + 1.0))
                    else:
                        beta = (2 * (hi[j] - x2[j]) / (hi[j] - lo[j])) ** (1.0 / (eta_m + 1.0))
                        
                    beta_q = 0.5 * (1 + beta)
                    beta_p = 0.5 * (1 - beta)
                    
                    c1[j] = beta_q * x1[j] + beta_p * x2[j]
                    c2[j] = beta_p * x1[j] + beta_q * x2[j]
                    
                    # Clip to bounds
                    c1[j] = np.clip(c1[j], lo[j], hi[j])
                    c2[j] = np.clip(c2[j], lo[j], hi[j])
                    
            return c1, c2

        def mutate(x):
            """
            Polynomial Mutation.
            """
            y = x.copy()
            for j in range(dim):
                if np.random.rand() < mutation_rate:
                    y_j = y[j]
                    delta1 = (y_j - lo[j]) / (hi[j] - lo[j])
                    delta2 = (hi[j] - y_j) / (hi[j] - lo[j])
                    u = np.random.rand()
                    if u < 0.5:
                        xy = (2 * u + (1 - 2 * u) * (delta1 ** (eta_m + 1))) ** (1.0 / (eta_m + 1.0))
                        deltaq = (xy - 1) * (hi[j] - lo[j])
                    else:
                        xy = (2 * (1 - u) + (2 * (u - 0.5)) * (delta2 ** (eta_m + 1))) ** (1.0 / (eta_m + 1.0))
                        deltaq = (xy - 1) * (hi[j] - lo[j])
                        
                    y[j] = y_j + deltaq
                    y[j] = np.clip(y[j], lo[j], hi[j])
            return y

        # Main NSGA-II Loop
        # We have `pop_size` individuals.
        # Each generation produces `pop_size` offspring.
        # Total evaluations = pop_size (initial) + generations * pop_size
        # Budget remaining = self.budget - eval_count
        # Max generations = remaining // pop_size
        
        max_generations = (self.budget - eval_count) // pop_size
        if max_generations < 1:
             return None

        for gen in range(max_generations):
            if time.time() > deadline or eval_count >= self.budget:
                break
                
            # Calculate ranks and crowding distances for current population
            ranks = fast_non_dominated_sort(F_pop)
            dists = crowding_distance(F_pop, ranks)
            
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                if time.time() > deadline or eval_count >= self.budget:
                    # If we run out, we stop generating offspring
                    break
                    
                p1, p2 = select_parents(F_pop, ranks, dists)
                c1, c2 = crossover(X_pop[p1], X_pop[p2])
                c1 = mutate(c1)
                
                # We only take one child per pair to save budget? 
                # Standard NSGA-II takes both. But we have a hard budget.
                # If we take both, we use 2 evals per offspring slot? No, 1 eval per individual.
                # We need pop_size individuals. So we need pop_size evaluations.
                # So we generate pop_size individuals.
                
                offspring_X[i] = c1
                try:
                    offspring_F[i] = problem(c1)
                    eval_count += 1
                except Exception:
                    offspring_F[i] = np.full(n_obj, 1e6)
                    eval_count += 1 # Count failed evals? Usually yes to prevent infinite loops, but let's assume problem is robust.
                    
            # Combine parents and offspring
            # P_t: (pop_size, dim), F_t: (pop_size, n_obj)
            # Q_t: (pop_size, dim), G_t: (pop_size, n_obj)
            # R_t: (2*pop_size, dim), H_t: (2*pop_size, n_obj)
            
            X_R = np.vstack([X_pop, offspring_X])
            F_R = np.vstack([F_pop, offspring_F])
            
            # Non-dominated sort on R
            ranks_R = fast_non_dominated_sort(F_R)
            dists_R = crowding_distance(F_R, ranks_R)
            
            # Sort R by rank (ascending) and then by crowding distance (descending)
            # We want to select the best pop_size individuals
            # Create a list of indices sorted by (rank, -dist)
            indices_R = np.arange(len(X_R))
            # Sort by rank, then by dist descending
            # Python sort is stable, so we can sort by dist desc first, then rank asc?
            # Or just create a key.
            
            # For efficiency, let's just select front by front.
            P_next = []
            i = 0
            # Get unique ranks sorted
            unique_ranks = sorted(set(ranks_R))
            
            for r in unique_ranks:
                # Get indices with this rank
                indices_in_front = [idx for idx in range(len(X_R)) if ranks_R[idx] == r]
                
                if len(P_next) + len(indices_in_front) <= pop_size:
                    # All fit
                    P_next.extend(indices_in_front)
                else:
                    # Need to select only some from this front
                    slots_left = pop_size - len(P_next)
                    # Sort this front by crowding distance descending
                    indices_in_front.sort(key=lambda idx: dists_R[idx], reverse=True)
                    P_next.extend(indices_in_front[:slots_left])
                    break
                    
            # Extract next population
            if len(P_next) == pop_size:
                X_pop = X_R[P_next]
                F_pop = F_R[P_next]
            else:
                # Should not happen, but safety
                break
                
        return None
# EVOLVE-BLOCK-END

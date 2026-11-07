"""NSGA-II based multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Parameters
        pop_size = 40
        # Ensure we don't exceed budget. 
        # Initial evals: pop_size.
        # Each generation: pop_size offspring + pop_size parents = 2*pop_size candidates, select pop_size.
        # We need to be careful with the exact budget.
        
        # Let's calculate how many full generations we can do.
        # Initial: pop_size evaluations.
        # Remaining: budget - pop_size.
        # Each generation consumes pop_size evaluations (for offspring).
        # Total evaluations = pop_size + n_gens * pop_size.
        # We want pop_size + n_gens * pop_size <= budget.
        # n_gens = (budget - pop_size) // pop_size.
        
        # If budget is small, we might not even be able to do one full generation.
        # For DTLZ1 with budget 200, pop_size 40 is reasonable.
        # 200 - 40 = 160. 160 // 40 = 4 generations.
        # Total evals = 40 + 4*40 = 200. Perfect.
        
        if budget < pop_size:
            pop_size = budget

        n_gens = (budget - pop_size) // pop_size

        # Initialize population
        P = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in P])
        used = pop_size

        # Helper functions for NSGA-II

        def fast_non_dominated_sort(F):
            """
            Returns ranks and crowding distances for a set of points F.
            F: (N, n_obj) array.
            Returns:
                ranks: (N,) array of rank indices (0 is best).
                crowding: (N,) array of crowding distances.
            """
            N = len(F)
            if N == 0:
                return np.array([]), np.array([])
            
            S = [[] for _ in range(N)] # S[i] = indices j dominated by i
            n = [0] * N # n[i] = number of points dominating i
            
            # Calculate dominance relationships
            for i in range(N):
                for j in range(i + 1, N):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n[i] += 1
            
            ranks = np.zeros(N, dtype=int)
            rank = 0
            current_front = [i for i in range(N) if n[i] == 0]
            
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = rank
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            next_front.append(j)
                current_front = next_front
                rank += 1
            
            # Calculate crowding distance for each front
            crowding = np.zeros(N)
            # Group indices by rank
            fronts = {}
            for i in range(N):
                r = ranks[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
            
            for r, indices in fronts.items():
                if len(indices) <= 2:
                    for i in indices:
                        crowding[i] = float('inf')
                    continue
                
                for m in range(n_obj):
                    # Sort indices by m-th objective
                    sorted_indices = sorted(indices, key=lambda i: F[i][m])
                    crowding[sorted_indices[0]] = float('inf')
                    crowding[sorted_indices[-1]] = float('inf')
                    
                    f_min = F[sorted_indices[0]][m]
                    f_max = F[sorted_indices[-1]][m]
                    
                    if f_max - f_min == 0:
                        continue
                    
                    for k in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[k]
                        prev_idx = sorted_indices[k-1]
                        next_idx = sorted_indices[k+1]
                        crowding[idx] += (F[next_idx][m] - F[prev_idx][m]) / (f_max - f_min)
            
            return ranks, crowding

        def tournament_select(P, F, ranks, crowding, k=2):
            """
            Select one individual using binary tournament.
            """
            i1 = np.random.randint(len(P))
            i2 = np.random.randint(len(P))
            
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                if crowding[i1] > crowding[i2]:
                    return i1
                else:
                    return i2

        def sbx_crossover(x1, x2, eta=20.0):
            """
            Simulated Binary Crossover.
            Returns two children.
            """
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x2)
            
            for i in range(dim):
                if np.random.random() < 0.5:
                    # Standard SBX
                    if x1[i] == x2[i]:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
                    else:
                        if x1[i] < x2[i]:
                            low, high = x1[i], x2[i]
                        else:
                            low, high = x2[i], x1[i]
                        
                        delta = high - low
                        if delta < 1e-10:
                            c1[i] = x1[i]
                            c2[i] = x2[i]
                            continue
                        
                        # Calculate beta
                        r = np.random.random()
                        if r <= 0.5:
                            beta = (2 * r) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - r))) ** (1.0 / (eta + 1.0))
                        
                        c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                        c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
                        # Clip to bounds
                        c1[i] = np.clip(c1[i], lo[i], hi[i])
                        c2[i] = np.clip(c2[i], lo[i], hi[i])
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            
            return c1, c2

        def polynomial_mutation(x, eta=20.0):
            """
            Polynomial Mutation.
            Returns mutated x.
            """
            x_mut = x.copy()
            for i in range(dim):
                if np.random.random() < 0.1: # Mutation rate 1/dim approx, but fixed 0.1 is often better for small dim
                    if x[i] == lo[i] or x[i] == hi[i]:
                        # If at boundary, push inwards
                        if x[i] == lo[i]:
                            delta = (hi[i] - lo[i])
                        else:
                            delta = (hi[i] - lo[i])
                        
                        r = np.random.random()
                        if r <= 0.5:
                            beta = (2 * r) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - r))) ** (1.0 / (eta + 1.0))
                        
                        if x[i] == lo[i]:
                            x_mut[i] = x[i] + beta * delta
                        else:
                            x_mut[i] = x[i] - beta * delta
                    else:
                        # Standard polynomial mutation
                        r = np.random.random()
                        if r <= 0.5:
                            beta = (2 * r) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - r))) ** (1.0 / (eta + 1.0))
                        
                        # delta is relative to range
                        delta = (hi[i] - lo[i])
                        
                        # Ensure we stay in bounds
                        # x_new = x +/- delta * beta
                        # If x + delta*beta > hi, we need to adjust beta or clip
                        # Standard approach: calculate max allowed delta based on position
                        if r <= 0.5:
                            # Add
                            max_delta = (hi[i] - x[i]) / delta
                            if beta > max_delta:
                                beta = max_delta
                            x_mut[i] = x[i] + beta * delta
                        else:
                            # Subtract
                            max_delta = (x[i] - lo[i]) / delta
                            if beta > max_delta:
                                beta = max_delta
                            x_mut[i] = x[i] - beta * delta
            
            x_mut = np.clip(x_mut, lo, hi)
            return x_mut

        # Main NSGA-II Loop
        for gen in range(n_gens):
            # Create offspring
            Q = np.zeros((pop_size, dim))
            for i in range(pop_size):
                p1_idx = tournament_select(P, F, ranks, crowding)
                p2_idx = tournament_select(P, F, ranks, crowding)
                
                c1, c2 = sbx_crossover(P[p1_idx], P[p2_idx])
                
                # We only need pop_size offspring. 
                # We can use both children if we pair up, or just one.
                # Standard NSGA-II creates N offspring from N parents.
                # Here we create 2 children per pair, but we only take one? 
                # Or we can fill Q with c1 and c2 from pop_size/2 pairs?
                # Let's just create pop_size individuals.
                # If pop_size is even, we can do pop_size/2 pairs and take both.
                # If odd, we might have to handle differently.
                # Simpler: For each i, pick 2 parents, do crossover, take 1 child (randomly or always c1).
                # Actually, standard SBX produces 2 children. 
                # Let's just take c1 for simplicity and efficiency, or randomly pick.
                # Let's randomly pick one of the two children to maintain diversity.
                if np.random.random() < 0.5:
                    Q[i] = c1
                else:
                    Q[i] = c2
                
                # Mutation
                Q[i] = polynomial_mutation(Q[i])
            
            # Evaluate offspring
            F_Q = np.array([problem(x) for x in Q])
            used += pop_size
            
            # Combine P and Q
            R = np.vstack((P, Q))
            F_R = np.vstack((F, F_Q))
            
            # Sort combined population
            ranks_R, crowding_R = fast_non_dominated_sort(F_R)
            
            # Select next generation P
            # Sort indices by rank, then by crowding distance (descending)
            sort_idx = np.lexsort((crowding_R, ranks_R))
            # We want highest crowding first, so we reverse the crowding sort?
            # np.lexsort uses last key as primary. 
            # So keys: (crowding, rank). 
            # We want rank ascending, crowding descending.
            # So we should sort by (rank, -crowding) or similar.
            # Let's do it manually to be safe.
            
            # Create list of (rank, -crowding, index)
            sort_keys = [(ranks_R[i], -crowding_R[i], i) for i in range(len(R))]
            sort_keys.sort()
            
            selected_indices = [k[2] for k in sort_keys[:pop_size]]
            
            P = R[selected_indices]
            F = F_R[selected_indices]
            
            # Update ranks and crowding for next generation's tournament
            ranks, crowding = fast_non_dominated_sort(F)

        return P
# EVOLVE-BLOCK-END

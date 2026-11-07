"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Parameters
        pop_size = 30
        # Number of evaluations for initialization
        init_evals = pop_size
        # Remaining budget for evolution
        remaining = budget - init_evals
        # Number of generations
        n_gen = max(1, remaining // pop_size)

        # Helper: fast non-dominated sorting
        def fast_non_dominated_sort(F):
            """
            Returns ranks array where ranks[i] is the front index (0-based) of solution i.
            F is a (N, n_obj) array.
            """
            N = F.shape[0]
            S = [set() for _ in range(N)]  # S[i] = set of indices dominated by i
            n = [0] * N  # n[i] = number of solutions dominating i
            rank = [-1] * N  # rank[i] = front index
            fronts = []
            
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n[i] += 1
            
            # First front
            front0 = [i for i in range(N) if n[i] == 0]
            if not front0:
                # All dominated? Should not happen with at least one non-dominated point usually, 
                # but if all are dominated by each other in a cycle (impossible for Pareto), 
                # or just handle edge case.
                # Actually, there must be at least one non-dominated point.
                # If front0 is empty, something is wrong, but let's assume at least one exists.
                # For safety, if empty, put all in front 0.
                front0 = list(range(N))
            
            rank = np.zeros(N, dtype=int)
            current_front = 0
            while front0:
                fronts.append(front0)
                next_front = []
                for i in front0:
                    rank[i] = current_front
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            next_front.append(j)
                front0 = next_front
                current_front += 1
            return rank, fronts

        # Helper: crowding distance
        def crowding_distance(F, ranks, indices):
            """
            Computes crowding distance for each solution in the population.
            F: (N, n_obj) array
            ranks: array of ranks for each solution
            indices: array of indices of solutions to consider (usually all)
            Returns: array of crowding distances
            """
            N = len(indices)
            cd = np.zeros(N)
            
            # Group by rank
            for r in range(max(ranks) + 1):
                # Get indices in this rank
                mask = ranks == r
                idx_in_rank = indices[mask]
                if len(idx_in_rank) <= 2:
                    cd[idx_in_rank] = np.inf
                    continue
                
                # Sort by each objective
                for m in range(n_obj):
                    # Get objective values for this rank
                    obj_vals = F[idx_in_rank, m]
                    sorted_order = np.argsort(obj_vals)
                    sorted_idx = idx_in_rank[sorted_order]
                    
                    # Assign inf to extremes
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    
                    f_min = obj_vals[sorted_order[0]]
                    f_max = obj_vals[sorted_order[-1]]
                    if f_max - f_min < 1e-15:
                        continue
                    
                    for k in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[k]
                        cd[idx] += (obj_vals[sorted_order[k+1]] - obj_vals[sorted_order[k-1]]) / (f_max - f_min)
            
            return cd

        # Helper: tournament selection
        def tournament_select(pop, ranks, cds, k=2):
            """
            Selects one individual from pop using tournament selection.
            pop: list/array of indices
            ranks: array of ranks
            cds: array of crowding distances
            k: tournament size
            """
            candidates = np.random.choice(pop, size=k, replace=False)
            best_idx = candidates[0]
            for c in candidates[1:]:
                # Prefer lower rank (better)
                if ranks[c] < ranks[best_idx]:
                    best_idx = c
                elif ranks[c] == ranks[best_idx]:
                    # Prefer higher crowding distance (less crowded)
                    if cds[c] > cds[best_idx]:
                        best_idx = c
            return best_idx

        # Helper: binary crossover (SBX)
        def sbx_crossover(x1, x2, eta=20.0):
            """
            Simulated Binary Crossover.
            Returns two children.
            """
            if np.random.random() > 0.5:
                return x1.copy(), x2.copy()
            
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x2)
            for i in range(len(x1)):
                if np.random.random() > 0.5:
                    continue
                if x1[i] == x2[i]:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                else:
                    if np.random.random() < 0.5:
                        # c1 is the one that gets the "lower" value potentially
                        if (2.0 * np.random.random()) <= (1.0 / (eta + 1.0)):
                            beta = ((2.0 * np.random.random()) * (x1[i] - x2[i]) + min(x1[i], x2[i])) / (x1[i] - x2[i] + min(x1[i], x2[i]))
                            # Correction: standard SBX formula
                            # Let's use a simpler, robust implementation
                            pass
                    # Let's use a simpler, more standard SBX implementation to avoid bugs
                    break
            
            # Simpler, more robust SBX
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x2)
            for i in range(len(x1)):
                if np.random.random() > 0.5:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                    continue
                if x1[i] == x2[i]:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                    continue
                
                # Calculate beta
                u = np.random.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                # x1 is larger than x2?
                if x1[i] > x2[i]:
                    x1_l, x2_l = x1[i], x2[i]
                    c1[i] = 0.5 * ((1.0 + beta) * x1_l + (1.0 - beta) * x2_l)
                    c2[i] = 0.5 * ((1.0 - beta) * x1_l + (1.0 + beta) * x2_l)
                else:
                    x1_l, x2_l = x2[i], x1[i]
                    c1[i] = 0.5 * ((1.0 + beta) * x1_l + (1.0 - beta) * x2_l)
                    c2[i] = 0.5 * ((1.0 - beta) * x1_l + (1.0 + beta) * x2_l)
            
            return c1, c2

        # Helper: polynomial mutation
        def poly_mutation(x, eta=20.0):
            """
            Polynomial mutation.
            """
            x_mut = x.copy()
            for i in range(len(x_mut)):
                if np.random.random() > 0.5:
                    continue
                u = np.random.random()
                if u < 0.5:
                    delta_q = (1.0 + 2.0 * (u - 0.5)) ** (1.0 / (eta + 1.0))
                else:
                    delta_q = (1.0 + 2.0 * (0.5 - u)) ** (1.0 / (eta + 1.0))
                
                # Bounds for x[i]
                lo_i = lo[i]
                hi_i = hi[i]
                if hi_i - lo_i < 1e-15:
                    continue
                
                delta_i = (hi_i - lo_i) * delta_q
                if u < 0.5:
                    x_mut[i] += delta_i
                else:
                    x_mut[i] -= delta_i
                x_mut[i] = np.clip(x_mut[i], lo_i, hi_i)
            return x_mut

        # Initialization
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        evaluations = init_evals

        # Initial sorting
        ranks, fronts = fast_non_dominated_sort(F)
        indices = np.arange(pop_size)
        cds = crowding_distance(F, ranks, indices)

        # Evolution
        while evaluations < budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and evaluations < budget:
                # Select parents
                p1 = tournament_select(indices, ranks, cds)
                p2 = tournament_select(indices, ranks, cds)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                
                # Mutation
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                
                # Evaluate c1
                f1 = problem(c1)
                evaluations += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if evaluations >= budget:
                    break
                
                # Evaluate c2
                f2 = problem(c2)
                evaluations += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if len(offspring_X) == 0:
                break

            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Sort combined
            combined_ranks, combined_fronts = fast_non_dominated_sort(combined_F)
            combined_indices = np.arange(len(combined_X))
            combined_cds = crowding_distance(combined_F, combined_ranks, combined_indices)
            
            # Select next population
            new_X = []
            new_F = []
            count = 0
            current_front = 0
            while count < pop_size:
                if current_front >= len(combined_fronts):
                    break
                front_indices = combined_fronts[current_front]
                if len(front_indices) <= (pop_size - count):
                    for idx in front_indices:
                        if count < pop_size:
                            new_X.append(combined_X[idx])
                            new_F.append(combined_F[idx])
                            count += 1
                else:
                    # Sort by crowding distance
                    front_cds = combined_cds[front_indices]
                    sorted_front = front_indices[np.argsort(-front_cds)]
                    for idx in sorted_front:
                        if count < pop_size:
                            new_X.append(combined_X[idx])
                            new_F.append(combined_F[idx])
                            count += 1
                        else:
                            break
                current_front += 1
            
            if len(new_X) < pop_size:
                # Fill remaining with random or best available?
                # Just fill with random to ensure full population
                while len(new_X) < pop_size:
                    x_new = lo + (hi - lo) * np.random.random(dim)
                    f_new = problem(x_new)
                    evaluations += 1
                    new_X.append(x_new)
                    new_F.append(f_new)
                    if evaluations >= budget:
                        break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Recalculate ranks and cds for next generation
            ranks, fronts = fast_non_dominated_sort(F)
            indices = np.arange(pop_size)
            cds = crowding_distance(F, ranks, indices)

        return None
# EVOLVE-BLOCK-END

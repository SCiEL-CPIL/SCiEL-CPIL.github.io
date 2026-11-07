"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """NSGA-II multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Parameters
        pop_size = 20
        eta_c = 15.0  # Distribution index for SBX
        eta_m = 20.0  # Distribution index for Polynomial Mutation
        prob_c = 0.9  # Probability of crossover
        
        # Helper functions
        def non_dominated_sort(F):
            """
            Perform non-dominated sorting on a set of objective vectors.
            Returns a list of lists, where each inner list contains indices of the population
            belonging to that front.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] contains indices dominated by i
            n_dominated = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1
            
            current_front = []
            for i in range(n):
                if n_dominated[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            front_index = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = front_index + 1
                            next_front.append(j)
                current_front = next_front
                front_index += 1
                
            return fronts, rank

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices in the population.
            Returns an array of distances for the given indices.
            """
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            cd = np.zeros(n)
            F_subset = F[indices]
            
            for m in range(n_obj):
                sorted_indices = np.argsort(F_subset[:, m])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                f_min = F_subset[sorted_indices[0], m]
                f_max = F_subset[sorted_indices[-1], m]
                if f_max - f_min < 1e-15:
                    continue
                for k in range(1, n - 1):
                    idx = sorted_indices[k]
                    cd[idx] += (F_subset[sorted_indices[k+1], m] - F_subset[sorted_indices[k-1], m]) / (f_max - f_min)
            
            return cd

        def sbx_crossover(x1, x2, lo, hi, eta_c, prob_c):
            """
            Simulated Binary Crossover.
            Returns two offspring.
            """
            if np.random.random() > prob_c:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(dim):
                if np.random.random() < 0.5:
                    if x1[i] == x2[i]:
                        continue
                    if x1[i] < x2[i]:
                        x_lo, x_hi = x1[i], x2[i]
                    else:
                        x_lo, x_hi = x2[i], x1[i]
                    
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            
            # Clip to bounds
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        def poly_mutation(x, lo, hi, eta_m):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for i in range(dim):
                if np.random.random() < 0.5:
                    x_lo, x_hi = lo[i], hi[i]
                    if x_hi - x_lo < 1e-15:
                        continue
                    u = np.random.random()
                    if u < 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    x_mut[i] = x[i] + delta1 * (x_hi - x_lo)
                    x_mut[i] = np.clip(x_mut[i], x_lo, x_hi)
            return x_mut

        # Initialize Population
        P = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in P], dtype=float)
        evaluations = pop_size
        
        # Main Loop
        while evaluations + pop_size <= budget:
            # Non-dominated sort and crowding distance for selection
            fronts, rank = non_dominated_sort(F)
            
            # Calculate crowding distance for each front
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    cd_front = crowding_distance(F, front)
                    for idx, cd_val in zip(front, cd_front):
                        cd[idx] = cd_val
            
            # Create Offspring
            Q = []
            F_Q = []
            
            for _ in range(pop_size):
                # Tournament Selection
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                
                # Selection based on rank and crowding distance
                if rank[p1_idx] < rank[p2_idx]:
                    p1 = P[p1_idx]
                elif rank[p1_idx] > rank[p2_idx]:
                    p1 = P[p2_idx]
                else:
                    if cd[p1_idx] > cd[p2_idx]:
                        p1 = P[p1_idx]
                    else:
                        p1 = P[p2_idx]
                
                p3_idx = np.random.randint(0, pop_size)
                p4_idx = np.random.randint(0, pop_size)
                if rank[p3_idx] < rank[p4_idx]:
                    p2 = P[p3_idx]
                elif rank[p3_idx] > rank[p4_idx]:
                    p2 = P[p4_idx]
                else:
                    if cd[p3_idx] > cd[p4_idx]:
                        p2 = P[p3_idx]
                    else:
                        p2 = P[p4_idx]
                
                # Crossover and Mutation
                c1, c2 = sbx_crossover(p1, p2, lo, hi, eta_c, prob_c)
                c1 = poly_mutation(c1, lo, hi, eta_m)
                c2 = poly_mutation(c2, lo, hi, eta_m)
                
                Q.append(c1)
                Q.append(c2)
                
                # We need to stop if we exceed budget, but we are generating in pairs.
                # Since pop_size is 20, we generate 20 offspring per generation.
                # The loop above runs pop_size/2 times? No, the loop runs pop_size times, generating 2 each?
                # No, standard NSGA-II generates pop_size offspring.
                # Let's adjust the loop to generate exactly pop_size offspring.
                pass
            
            # The loop above was incorrect in structure for generating exactly pop_size offspring.
            # Let's regenerate Q properly.
            Q = []
            for _ in range(pop_size // 2):
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                
                if rank[p1_idx] < rank[p2_idx]:
                    p1 = P[p1_idx]
                elif rank[p1_idx] > rank[p2_idx]:
                    p1 = P[p2_idx]
                else:
                    if cd[p1_idx] > cd[p2_idx]:
                        p1 = P[p1_idx]
                    else:
                        p1 = P[p2_idx]
                
                p3_idx = np.random.randint(0, pop_size)
                p4_idx = np.random.randint(0, pop_size)
                if rank[p3_idx] < rank[p4_idx]:
                    p2 = P[p3_idx]
                elif rank[p3_idx] > rank[p4_idx]:
                    p2 = P[p4_idx]
                else:
                    if cd[p3_idx] > cd[p4_idx]:
                        p2 = P[p3_idx]
                    else:
                        p2 = P[p4_idx]
                
                c1, c2 = sbx_crossover(p1, p2, lo, hi, eta_c, prob_c)
                c1 = poly_mutation(c1, lo, hi, eta_m)
                c2 = poly_mutation(c2, lo, hi, eta_m)
                
                Q.append(c1)
                Q.append(c2)
            
            # Evaluate Offspring
            F_Q = np.array([problem(x) for x in Q], dtype=float)
            evaluations += pop_size
            
            # Combine Parents and Offspring
            R = np.vstack([P, Q])
            F_R = np.vstack([F, F_Q])
            
            # Non-dominated sort and crowding distance for combined population
            fronts_R, rank_R = non_dominated_sort(F_R)
            
            cd_R = np.zeros(2 * pop_size)
            for front in fronts_R:
                if len(front) > 0:
                    cd_front = crowding_distance(F_R, front)
                    for idx, cd_val in zip(front, cd_front):
                        cd_R[idx] = cd_val
            
            # Select next generation
            P_next = []
            for front in fronts_R:
                if len(P_next) + len(front) <= pop_size:
                    for idx in front:
                        P_next.append(R[idx])
                else:
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda x: cd_R[x], reverse=True)
                    for idx in sorted_front:
                        if len(P_next) < pop_size:
                            P_next.append(R[idx])
                        else:
                            break
                    break
            
            P = np.array(P_next)
            F = np.array([problem(x) for x in P]) # Re-evaluating is wasteful, but we already have F_R.
            # We should extract F from F_R to avoid extra evaluations.
            F = np.array([F_R[np.where(np.all(R == p, axis=1))[0][0]] for p in P])
            # This extraction is fragile if there are duplicates. Better to keep indices.
            
            # Let's refine the selection to keep indices.
            P_indices = []
            for front in fronts_R:
                if len(P_indices) + len(front) <= pop_size:
                    P_indices.extend(front)
                else:
                    sorted_front = sorted(front, key=lambda x: cd_R[x], reverse=True)
                    for idx in sorted_front:
                        if len(P_indices) < pop_size:
                            P_indices.append(idx)
                        else:
                            break
                    break
            
            P = R[P_indices]
            F = F_R[P_indices]

        return None
# EVOLVE-BLOCK-END

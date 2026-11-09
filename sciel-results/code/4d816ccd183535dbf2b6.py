# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a fixed seed for reproducibility within the run, 
        # but the problem statement says evaluator seeds np.random.
        # We should use np.random directly to respect the evaluator's seeding.
        
        pop_size = 50
        
        # Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        eval_count = pop_size
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Set of solutions dominated by i
            n_dom = np.zeros(n, dtype=int)  # Number of solutions dominated by i
            rank = np.zeros(n, dtype=int)
            
            fronts = []
            front0 = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        
            if n_dom[i] == 0:
                rank[i] = 0
                front0.append(i)
                
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
                
            return fronts, rank

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices in F.
            Returns a dictionary of index -> distance.
            """
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = len(indices)
            dist = np.zeros(m)
            f = F[indices]
            
            for j in range(f.shape[1]):
                f_min = np.min(f[:, j])
                f_max = np.max(f[:, j])
                if f_max == f_min:
                    continue
                # Sort by objective j
                sorted_idx = np.argsort(f[:, j])
                # End points get infinite distance
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                for k in range(1, m - 1):
                    i_prev = sorted_idx[k-1]
                    i_next = sorted_idx[k+1]
                    dist[sorted_idx[k]] += (f[i_next, j] - f[i_prev, j]) / (f_max - f_min)
                    
            # Map back to original indices
            cd_dict = {}
            for i, orig_idx in enumerate(indices):
                cd_dict[orig_idx] = dist[i]
            return cd_dict

        def tournament_select(F, ranks, cd_dict, n_sel):
            """
            Select n_sel indices using binary tournament selection based on rank and crowding distance.
            """
            selected = []
            for _ in range(n_sel):
                i1 = np.random.randint(len(F))
                i2 = np.random.randint(len(F))
                
                if ranks[i1] < ranks[i2]:
                    selected.append(i1)
                elif ranks[i2] < ranks[i1]:
                    selected.append(i2)
                else:
                    # Same rank, use crowding distance
                    if cd_dict[i1] > cd_dict[i2]:
                        selected.append(i1)
                    else:
                        selected.append(i2)
            return selected

        # Main loop
        while eval_count < budget:
            if time.time() > deadline:
                break
            
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            n_offspring = pop_size
            for _ in range(n_offspring):
                if eval_count >= budget:
                    break
                    
                # Select parents
                # We need ranks and cd for current population
                # To avoid recalculating every single offspring generation (expensive), 
                # we calculate once per generation.
                
                # Actually, standard NSGA-II creates a whole offspring population, combines, sorts, and selects.
                # So we break this inner loop logic. Let's restructure to generation-based.
                pass
                
            # Restructure: Generation-based NSGA-II
            
            # 1. Selection and Crossover/Mutation to create offspring
            fronts, ranks = fast_non_dominated_sort(F)
            # Calculate crowding distance for the first front (or all fronts? Usually just the one that might be crowded)
            # For simplicity and correctness in selection, we calculate CD for all fronts or just use rank first.
            # Standard NSGA-II: Sort by rank, then CD within rank.
            
            # Let's compute CD for each front to allow proper selection
            cd_dicts = []
            for front in fronts:
                cd_dicts.append(crowding_distance(F, front))
            
            # Function to get CD for an index
            def get_cd(idx):
                r = ranks[idx]
                return cd_dicts[r].get(idx, 0.0)
            
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < n_offspring and eval_count < budget:
                # Select 2 parents
                p1 = tournament_select(F, ranks, {i: get_cd(i) for i in range(len(F))}, 1)[0]
                p2 = tournament_select(F, ranks, {i: get_cd(i) for i in range(len(F))}, 1)[0]
                
                x1 = X[p1]
                x2 = X[p2]
                
                # SBX Crossover
                child1 = np.empty(dim)
                child2 = np.empty(dim)
                for j in range(dim):
                    if np.random.random() < 0.5:
                        if abs(x1[j] - x2[j]) < 1e-14:
                            child1[j] = x1[j]
                            child2[j] = x2[j]
                        else:
                            u = np.random.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (15 + 1)) # eta_cr = 15
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (15 + 1))
                            
                            child1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                            child2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                    else:
                        child1[j] = x1[j]
                        child2[j] = x2[j]
                
                # Polynomial Mutation
                def poly_mutate(x):
                    x_mut = x.copy()
                    for j in range(dim):
                        if np.random.random() < 1.0 / dim:
                            u = np.random.random()
                            if u <= 0.5:
                                delta_q = (2 * u) ** (1.0 / (20 + 1)) - 1 # eta_m = 20
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (20 + 1))
                            
                            x_mut[j] += delta_q * (hi[j] - lo[j])
                    return np.clip(x_mut, lo, hi)
                
                child1 = poly_mutate(child1)
                child2 = poly_mutate(child2)
                
                # Evaluate
                f1 = problem(child1)
                eval_count += 1
                offspring_X.append(child1)
                offspring_F.append(f1)
                
                if eval_count >= budget:
                    break
                    
                if len(offspring_X) < n_offspring:
                    f2 = problem(child2)
                    eval_count += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)
            
            if not offspring_X:
                break
                
            # Combine parents and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Selection
            fronts_c, ranks_c = fast_non_dominated_sort(F_combined)
            
            new_X = []
            new_F = []
            for front in fronts_c:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend([X_combined[i] for i in front])
                    new_F.extend([F_combined[i] for i in front])
                else:
                    # Need to select some from this front
                    remaining = pop_size - len(new_X)
                    cd_front = crowding_distance(F_combined, front)
                    # Sort by CD descending
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        new_X.append(X_combined[i])
                        new_F.append(F_combined[i])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
            # If we didn't fill the population (should not happen if offspring >= pop_size)
            # But with budget constraints, we might have fewer offspring.
            # In that case, we keep the current population size as is.
            # The loop condition `while eval_count < budget` handles termination.
            
        # Return all evaluated points? 
        # The task says: "Every returned objective vector is recorded... return value of __call__ is ignored."
        # Wait, "Every returned objective vector is recorded". 
        # Does this mean the evaluator records every call to `problem(x)`?
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This implies the evaluator wraps `problem` or intercepts calls?
        # No, usually in these tasks, the `Optimizer` is expected to return the set of non-dominated points or all points?
        # Let's re-read carefully: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This phrasing is tricky. If the return value is ignored, how are the vectors recorded?
        # It must mean that the `problem` function provided by the evaluator records every call internally.
        # So I just need to make sure I use the `problem` function for all evaluations.
        # The `return` statement of `__call__` is indeed ignored for scoring, but might be used for debugging or other purposes? 
        # "The return value of __call__ is ignored."
        # So I can return anything, but I should probably return the non-dominated set for sanity.
        
        # However, the scoring is based on "all recorded vectors". 
        # If the evaluator records every `problem(x)` call, then my internal bookkeeping of `all_F` in the previous code was unnecessary for the score, 
        # but the `problem` calls are what matter.
        
        # I will return the final non-dominated front from my population as a good practice.
        fronts_final, _ = fast_non_dominated_sort(F)
        if fronts_final:
            best_front = fronts_final[0]
            return F[best_front]
        return F

# EVOLVE-BLOCK-END

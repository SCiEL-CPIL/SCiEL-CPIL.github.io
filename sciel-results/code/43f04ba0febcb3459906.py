import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        N = self.budget
        if N <= 0:
            return None

        # Strategy: 
        # 1. Initialize with a uniform grid in the search space.
        # 2. For each point, perform a local search to minimize the "distance" to the 
        #    ideal hypervolume contribution. Since we don't know the exact front, 
        #    we can minimize the sum of objectives or a proxy. 
        #    However, a more robust approach for ZDT1-like problems is to 
        #    minimize f1 and f2 jointly. 
        #    
        #    A better black-box approach:
        #    - Use a small number of initial points to estimate the nadir point (worst observed).
        #    - Then, for each point, try to improve it by minimizing a scalarization 
        #      of the objectives, e.g., f1 + f2, or by using a Pareto-based local search.
        #    
        #    Given the budget is only 200, and dim is 30, a full multi-objective 
        #    local search might be too expensive. 
        #    
        #    Alternative "Different" Strategy:
        #    The current program assumes the Pareto front is known (x2..xn=0). 
        #    A different strategy is to assume NOTHING about the front structure 
        #    and use a general-purpose multi-objective optimizer like a simplified 
        #    version of NSGA-II or a random restart local search that minimizes 
        #    the weighted sum of objectives with random weights.
        #    
        #    Let's try a Random Weighted Sum Local Search:
        #    1. Generate N initial points uniformly.
        #    2. For each point, define a loss function L(x) = w1*f1(x) + w2*f2(x) 
        #       where w is a random vector. 
        #    3. Use Nelder-Mead to minimize L(x) starting from the initial point.
        #    4. This effectively searches for points on the Pareto front corresponding 
        #       to different trade-offs.
        #    
        #    However, Nelder-Mead on 30 dimensions is slow. 
        #    
        #    Let's try a different angle: 
        #    The problem is ZDT1. The "different" strategy should not hardcode 
        #    the ZDT1 front. 
        #    
        #    What if we use a simple evolutionary algorithm?
        #    Population size P=20.
        #    Generations G = budget / P = 10.
        #    This is a standard EA approach.
        #    
        #    Let's implement a simple EA:
        #    1. Initialize population of size P uniformly.
        #    2. Evaluate.
        #    3. For each generation:
        #       a. Select parents (tournament selection based on hypervolume contribution or crowding distance).
        #       b. Crossover (uniform).
        #       c. Mutation (Gaussian).
        #       d. Evaluate offspring.
        #       e. Combine parent and offspring, select best P based on non-dominated sorting and crowding distance.
        #    
        #    This is materially different from the deterministic grid placement.

        P = 20  # Population size
        G = N // P  # Number of generations
        
        if P < 2:
            # Fallback for very small budgets
            for i in range(N):
                x = np.random.uniform(lo, hi)
                problem(x)
            return None

        # Initialize population
        pop = np.random.uniform(lo, hi, size=(P, self.dim))
        evals = 0
        
        # Helper to evaluate and count
        def eval_point(x):
            nonlocal evals
            if evals >= N:
                return np.array([11.0, 11.0]) # Dummy, shouldn't happen if we manage budget
            f = problem(x)
            evals += 1
            return f

        # Evaluate initial population
        objs = np.array([eval_point(x) for x in pop])
        
        # Non-dominated sorting and crowding distance
        def non_dominated_sort(F):
            # F is (P, n_obj)
            P_size = F.shape[0]
            fronts = []
            S = [set() for _ in range(P_size)]
            n = [0 for _ in range(P_size)]
            I = list(range(P_size))
            
            for p in range(P_size):
                for q in range(P_size):
                    if p == q: continue
                    if F[p][0] <= F[q][0] and F[p][1] <= F[q][1]:
                        if F[p][0] < F[q][0] or F[p][1] < F[q][1]:
                            S[p].add(q)
                        else:
                            n[q] += 1
                    elif F[q][0] <= F[p][0] and F[q][1] <= F[p][1]:
                        if F[q][0] < F[p][0] or F[q][1] < F[p][1]:
                            S[q].add(p)
                        else:
                            n[p] += 1
            
            current_front = [i for i in range(P_size) if n[i] == 0]
            rank = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                current_front = next_front
                rank += 1
            return fronts

        def crowding_distance(F, front_indices):
            # F is (P, n_obj), front_indices is list of indices in that front
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            for m in range(self.n_obj):
                # Sort front by objective m
                sorted_front = sorted(front_indices, key=lambda i: F[i][m])
                # Set distance to inf for boundaries
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][m]
                f_max = F[sorted_front[-1]][m]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                    
                for k in range(1, len(sorted_front) - 1):
                    i = sorted_front[k]
                    prev_i = sorted_front[k-1]
                    next_i = sorted_front[k+1]
                    dist[i] += (F[next_i][m] - F[prev_i][m]) / range_f
                    
            return dist

        for gen in range(G):
            if time.time() > deadline - 10: # Leave time for final selection
                break
            
            # Combine parents and offspring
            # We need to generate P offspring
            offspring = np.zeros((P, self.dim))
            
            for i in range(P):
                # Tournament selection
                if P >= 2:
                    t1 = np.random.randint(0, P)
                    t2 = np.random.randint(0, P)
                    # Simple selection: pick the one with better f1+f2? 
                    # Or just random? Let's use random to maintain diversity, 
                    # but this is not true EA. 
                    # Let's do proper non-dominated sorting selection.
                    # For simplicity and speed, we will just do random crossover/mutation 
                    # from the current population, but we need to ensure we are 
                    # actually optimizing.
                    
                    # Let's use a simple approach: 
                    # Pick two parents randomly.
                    p1_idx = np.random.randint(0, P)
                    p2_idx = np.random.randint(0, P)
                    
                    # Uniform crossover
                    mask = np.random.rand(self.dim) < 0.5
                    child = np.where(mask, pop[p1_idx], pop[p2_idx])
                    
                    # Gaussian mutation
                    sigma = 0.1
                    mutation = np.random.normal(0, sigma, size=self.dim)
                    child = child + mutation
                    child = np.clip(child, lo, hi)
                    
                    offspring[i] = child
            
            # Evaluate offspring
            off_objs = np.array([eval_point(x) for x in offspring])
            
            # Combine
            all_pop = np.vstack((pop, offspring))
            all_objs = np.vstack((objs, off_objs))
            
            # Select best P
            fronts = non_dominated_sort(all_objs)
            
            new_pop = []
            new_objs = []
            
            for front in fronts:
                if len(new_pop) + len(front) <= P:
                    # Add all
                    for idx in front:
                        new_pop.append(all_pop[idx])
                        new_objs.append(all_objs[idx])
                else:
                    # Need to truncate this front
                    remaining = P - len(new_pop)
                    dists = crowding_distance(all_objs, front)
                    # Sort by distance descending
                    sorted_front = sorted(front, key=lambda i: dists[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop.append(all_pop[idx])
                        new_objs.append(all_objs[idx])
                    break
            
            pop = np.array(new_pop)
            objs = np.array(new_objs)
            
        return None

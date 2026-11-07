"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using NSGA-II for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # NSGA-II parameters
        pop_size = 40
        n_gen = (budget - pop_size) // pop_size
        if n_gen < 1:
            n_gen = 1
            pop_size = budget // 2
            
        # Initialize population with LHS
        lhs = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            offsets = np.random.rand(pop_size)
            lhs[:, d] = (perm + offsets) / pop_size
            
        x_pop = lo + lhs * (hi - lo)
        
        # Evaluate initial population
        F = np.array([problem(x_pop[i]) for i in range(pop_size)])
        eval_count = pop_size
        
        # Helper: non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            front = [0]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # p dominates q?
                    if all(F[p][k] <= F[q][k] for k in range(n_obj)) and any(F[p][k] < F[q][k] for k in range(n_obj)):
                        S[p].append(q)
                    # q dominates p?
                    elif all(F[q][k] <= F[p][k] for k in range(n_obj)) and any(F[q][k] < F[p][k] for k in range(n_obj)):
                        n_count[p] += 1
                        
            if n_count[p] == 0:
                rank[p] = 0
                front[0] += 1
                
            i = 0
            while front[i] > 0:
                next_front = []
                i += 1
                for p in S[i-1]:
                    n_count[p] -= 1
                    if n_count[p] == 0:
                        rank[p] = i
                        next_front.append(p)
                if next_front:
                    front.append(len(next_front))
                else:
                    break
                    
            return rank, front

        # Helper: crowding distance
        def crowding_distance(F, rank, front_idx):
            n = len(F)
            cd = [0.0] * n
            members = [i for i in range(n) if rank[i] == front_idx]
            n_members = len(members)
            if n_members <= 2:
                for i in members:
                    cd[i] = float('inf')
                return cd
                
            for k in range(n_obj):
                sorted_members = sorted(members, key=lambda i: F[i][k])
                cd[sorted_members[0]] = float('inf')
                cd[sorted_members[-1]] = float('inf')
                f_min = F[sorted_members[0]][k]
                f_max = F[sorted_members[-1]][k]
                if f_max == f_min:
                    continue
                for j in range(1, n_members - 1):
                    cd[sorted_members[j]] += (F[sorted_members[j+1]][k] - F[sorted_members[j-1]][k]) / (f_max - f_min)
                    
            return cd

        # Helper: SBX crossover
        def sbx_crossover(x1, x2, lo, hi, eta=15.0):
            x1 = x1.copy()
            x2 = x2.copy()
            for d in range(len(x1)):
                if np.random.rand() < 0.5:
                    if abs(x1[d] - x2[d]) < 1e-14:
                        continue
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = ((1.0 + 2.0 * (x1[d] - x2[d]) / (hi[d] - lo[d])) ** (eta + 1))
                        c1 = 0.5 * (x1[d] + x2[d]) - 0.5 * (x1[d] - x2[d]) * beta
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = ((1.0 + 2.0 * (x2[d] - x1[d]) / (hi[d] - lo[d])) ** (eta + 1))
                        c1 = 0.5 * (x1[d] + x2[d]) + 0.5 * (x1[d] - x2[d]) * beta
                    c1 = np.clip(c1, lo[d], hi[d])
                    x1[d] = c1
                    x2[d] = lo[d] + hi[d] - c1
            return x1, x2

        # Helper: polynomial mutation
        def polynomial_mutation(x, lo, hi, eta=20.0, prob=None):
            if prob is None:
                prob = 1.0 / len(x)
            x = x.copy()
            for d in range(len(x)):
                if np.random.rand() < prob:
                    u = np.random.rand()
                    if u < 0.5:
                        delta_q = (2.0 * u) ** (1.0 / (eta + 1)) - 1.0
                    else:
                        delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1))
                    x[d] += delta_q * (hi[d] - lo[d])
                    x[d] = np.clip(x[d], lo[d], hi[d])
            return x

        # Main loop
        for gen in range(n_gen):
            # Non-dominated sort
            rank, front = non_dominated_sort(F)
            
            # Create offspring
            offspring_x = []
            offspring_F = []
            
            while len(offspring_x) < pop_size:
                if eval_count + 2 > budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select parents using tournament selection
                def tournament_select(pop_x, pop_F, rank, size=2):
                    indices = np.random.randint(0, len(pop_x), size=size)
                    best = indices[0]
                    for idx in indices[1:]:
                        if rank[idx] < rank[best]:
                            best = idx
                    return best
                    
                p1 = tournament_select(x_pop, F, rank)
                p2 = tournament_select(x_pop, F, rank)
                
                # Crossover
                c1, c2 = sbx_crossover(x_pop[p1], x_pop[p2], lo, hi)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                # Evaluate
                f1 = problem(c1)
                eval_count += 1
                offspring_x.append(c1)
                offspring_F.append(f1)
                
                if len(offspring_x) < pop_size and eval_count < budget:
                    f2 = problem(c2)
                    eval_count += 1
                    offspring_x.append(c2)
                    offspring_F.append(f2)
                    
            if not offspring_x:
                break
                
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            combined_x = np.vstack([x_pop, np.array(offspring_x)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined
            rank_comb, front_comb = non_dominated_sort(combined_F)
            
            # Select next generation
            next_x = []
            next_F = []
            i = 0
            while len(next_x) < pop_size and i < len(front_comb):
                front_members = [j for j in range(len(combined_F)) if rank_comb[j] == i]
                if len(next_x) + len(front_members) <= pop_size:
                    for j in front_members:
                        next_x.append(combined_x[j])
                        next_F.append(combined_F[j])
                else:
                    # Need to select from this front using crowding distance
                    cd = crowding_distance(combined_F, rank_comb, i)
                    remaining = pop_size - len(next_x)
                    front_sorted = sorted(front_members, key=lambda j: cd[j], reverse=True)
                    for j in front_sorted[:remaining]:
                        next_x.append(combined_x[j])
                        next_F.append(combined_F[j])
                    break
                i += 1
                
            x_pop = np.array(next_x)
            F = np.array(next_F)
            
            if time.time() > deadline:
                break
                
        return x_pop
# EVOLVE-BLOCK-END

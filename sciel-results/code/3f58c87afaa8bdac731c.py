import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish in < 300s.
        # However, with only 200 evaluations of a simple numpy function, 
        # the computation time is negligible (milliseconds). 
        # The risk is only if `problem` is very slow, but the contract says it clips and returns.
        # We will still guard against infinite loops or extreme slowness.
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        evals = 0
        population = []
        fitnesses = []
        
        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                return None
            evals += 1
            fx = np.asarray(problem(x), dtype=float)
            if not np.all(np.isfinite(fx)):
                return None
            return fx

        def non_dominated_sort(pop, fits):
            """
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(pop)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            fronts = []
            
            for i in range(n):
                S[i] = []
                n_count[i] = 0
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(fits[i] <= fits[j]) and np.any(fits[i] < fits[j]):
                        S[i].append(j)
                    elif np.all(fits[j] <= fits[i]) and np.any(fits[j] < fits[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
            
            k = 0
            while fronts[k] is not None and k < n:
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts

        def crowding_distance(front_indices, fits):
            """
            Calculate crowding distance for individuals in a front.
            """
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            for m in range(n_obj):
                # Sort indices by objective m
                sorted_idx = sorted(front_indices, key=lambda i: fits[i][m])
                min_val = fits[sorted_idx[0]][m]
                max_val = fits[sorted_idx[-1]][m]
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                if max_val - min_val < 1e-10:
                    continue
                    
                for k in range(1, n - 1):
                    i_prev = sorted_idx[k-1]
                    i_next = sorted_idx[k+1]
                    i_curr = sorted_idx[k]
                    dist[i_curr] += (fits[i_next][m] - fits[i_prev][m]) / (max_val - min_val)
            
            # Map back to original indices
            cd = {}
            for i, idx in enumerate(front_indices):
                cd[idx] = dist[i]
            return cd

        def de_mutation_crossover(pop, fits, F, CR):
            """
            Generate offspring using DE/rand/1/bin.
            """
            n = len(pop)
            offspring = []
            
            for i in range(n):
                # Select 3 distinct random individuals
                # We need to ensure they are distinct from i if possible, 
                # but DE usually allows any distinct triple from the population.
                while True:
                    a, b, c = np.random.choice(n, 3, replace=False)
                    if a != b and b != c and a != c:
                        break
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Boundary handling: Clip to [0, 1]
                v = np.clip(v, 0.0, 1.0)
                
                # Crossover
                u = np.zeros_like(pop[i])
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() <= CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                offspring.append(u)
            
            return offspring

        # Initialize population
        # Use a mix of uniform random and some structured points to ensure coverage
        init_size = 40
        # Ensure we don't exceed budget
        if init_size > budget:
            init_size = budget
            
        # Generate initial population
        for _ in range(init_size):
            x = np.random.rand(dim)
            # Map to bounds
            x = lo + x * (hi - lo)
            fx = evaluate(x)
            if fx is not None:
                population.append(x)
                fitnesses.append(fx)
        
        if not population:
            return np.array([])

        F = 0.8
        CR = 0.5
        pop_size = len(population)
        
        # Evolution Loop
        while evals < budget:
            # Check time
            if time.time() - t0 > 280:
                break
                
            # Calculate ranks
            fronts = non_dominated_sort(population, fitnesses)
            
            # Generate offspring
            # We generate as many offspring as we can afford
            # But typically we generate one offspring per parent in NSGA-II
            # Let's generate `pop_size` offspring, but check budget
            num_offspring = min(pop_size, budget - evals)
            if num_offspring <= 0:
                break
            
            # We need to select parents. In standard DE, we mutate each individual.
            # We will generate `num_offspring` children.
            children = []
            child_fits = []
            
            for _ in range(num_offspring):
                # Pick a random parent to mutate (DE usually mutates each, 
                # but here we sample to fit budget)
                # Actually, it's better to iterate through the current population 
                # and mutate each one, but if budget is tight, we might not be able to.
                # With budget 200 and pop 40, we can do ~4 full generations.
                # So we should mutate ALL current population members.
                pass
            
            # Let's reset logic: Generate offspring for the entire current population
            offspring = de_mutation_crossover(population, fitnesses, F, CR)
            
            # Evaluate offspring
            for x in offspring:
                if evals >= budget:
                    break
                fx = evaluate(x)
                if fx is not None:
                    children.append(x)
                    child_fits.append(fx)
            
            if not children:
                break
                
            # Combine parents and offspring
            combined_pop = population + children
            combined_fit = fitnesses + child_fits
            
            # Selection: NSGA-II style
            # 1. Non-dominated sort
            fronts = non_dominated_sort(combined_pop, combined_fit)
            
            new_pop = []
            new_fit = []
            
            for front in fronts:
                if len(new_pop) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_fit.append(combined_fit[idx])
                else:
                    # Need to truncate
                    # Calculate crowding distance for this front
                    cd = crowding_distance(front, combined_fit)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    slots_needed = pop_size - len(new_pop)
                    for idx in sorted_front[:slots_needed]:
                        new_pop.append(combined_pop[idx])
                        new_fit.append(combined_fit[idx])
                    break
            
            if not new_pop:
                break
                
            population = new_pop
            fitnesses = new_fit
            
            # If we didn't use all offspring because of budget, we stop anyway
            # The loop condition `evals < budget` will handle termination

        # Return the best points found
        # The problem states: "Every returned objective vector is recorded... return value of __call__ is ignored"
        # Wait, the contract says: "Every returned objective vector is recorded... return value of __call__ is ignored."
        # This implies I should NOT return the population from __call__ in a way that gets recorded?
        # "The return value of __call__ is ignored."
        # So I just need to make sure the `problem` calls are the ones recording.
        # My code calls `problem(x)` inside `evaluate`. This is correct.
        # The return value of `__call__` is ignored by the evaluator.
        
        return np.array(population)

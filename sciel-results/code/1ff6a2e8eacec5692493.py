import random
import math
import time

def construct():
    def score(S):
        present = [False] * 513
        for x in S:
            present[x] = True
            
        diffs = set()
        list_S = list(S)
        n = len(list_S)
        
        for i in range(n):
            a = list_S[i]
            for j in range(i+1, n):
                c = list_S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if present[b]:
                        d = c - a
                        diffs.add(d)
        return len(diffs)

    def random_set():
        return set(random.sample(range(1, 513), 32))

    def repair(S):
        # Ensure S has exactly 32 distinct integers in 1..512
        S = set(S)
        # Remove invalid
        S = {x for x in S if 1 <= x <= 512}
        
        # If too many, remove random
        if len(S) > 32:
            items = list(S)
            random.shuffle(items)
            S = set(items[:32])
            
        # If too few, add random
        while len(S) < 32:
            x = random.randint(1, 512)
            S.add(x)
            
        return S

    def crossover(p1, p2):
        # Merge and repair
        child = p1 | p2
        return repair(child)

    def mutate(S, rate=0.1):
        S = set(S)
        n_mut = int(32 * rate)
        for _ in range(n_mut):
            if random.random() < 0.5:
                # Replace one
                if S:
                    x = random.choice(list(S))
                    S.remove(x)
                    S.add(random.randint(1, 512))
            else:
                # Swap two
                items = list(S)
                if len(items) >= 2:
                    i, j = random.sample(range(len(items)), 2)
                    x, y = items[i], items[j]
                    # Replace x with new, y with new? Or just swap values? 
                    # Swapping values in a set is not a move. 
                    # Move: Remove x, add u. Remove y, add v.
                    u = random.randint(1, 512)
                    v = random.randint(1, 512)
                    if u not in S and v not in S and u != v:
                        S.remove(x)
                        S.remove(y)
                        S.add(u)
                        S.add(v)
        return repair(S)

    def tournament_select(pop, fitnesses, k=3):
        # Pick k random, return best
        indices = random.sample(range(len(pop)), k)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return pop[best_idx]

    start_time = time.time()
    time_limit = 110
    
    pop_size = 20
    population = [random_set() for _ in range(pop_size)]
    fitnesses = [score(ind) for ind in population]
    
    best_individual = max(population, key=lambda x: fitnesses[population.index(x)])
    best_score = max(fitnesses)
    
    generation = 0
    while time.time() - start_time < time_limit:
        # Selection, Crossover, Mutation
        new_population = []
        
        # Elitism: Keep top 5
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        for i in sorted_indices[:5]:
            new_population.append(population[i])
            
        # Fill rest
        while len(new_population) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            child = mutate(child)
            
            new_population.append(child)
            
        # Evaluate new population
        fitnesses = [score(ind) for ind in new_population]
        
        # Update best
        max_fit = max(fitnesses)
        if max_fit > best_score:
            best_score = max_fit
            best_individual = new_population[fitnesses.index(max_fit)]
            
        population = new_population
        generation += 1
        
    return list(best_individual)

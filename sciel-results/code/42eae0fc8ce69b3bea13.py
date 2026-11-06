"""New program: Genetic Algorithm with repair for 4-AP free sets."""
import random

def construct():
    n = 256
    random.seed(42)
    
    # Precompute 4-APs and 3-APs
    all_4aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            all_4aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    all_3aps = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            all_3aps.append((a - 1, a + d - 1, a + 2 * d - 1))
            
    # Precompute degrees for repair heuristic
    deg_4ap = [0] * n
    for ap in all_4aps:
        for x in ap:
            deg_4ap[x] += 1
            
    def count_3aps(S):
        S_set = set(S)
        count = 0
        for a, b, c in all_3aps:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
        
    def repair(S):
        S_set = set(S)
        changed = True
        while changed:
            changed = False
            for a, b, c, d in all_4aps:
                if a in S_set and b in S_set and c in S_set and d in S_set:
                    candidates = [a, b, c, d]
                    candidates.sort(key=lambda x: deg_4ap[x])
                    S_set.remove(candidates[0])
                    changed = True
                    break
        return S_set

    # Initialize population
    pop_size = 40
    population = []
    fitnesses = []
    
    for _ in range(pop_size):
        S = set(random.sample(range(n), int(n*0.25)))
        S = repair(S)
        if len(S) > 0:
            population.append(S)
            fitnesses.append(count_3aps(S))
            
    if not population:
        return [1, 2, 4, 8]
        
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_S = population[best_idx]
    best_score = fitnesses[best_idx]
    
    # GA Loop
    num_generations = 60
    mutation_rate = 0.15
    
    for gen in range(num_generations):
        new_population = []
        new_fitnesses = []
        
        # Elitism
        new_population.append(best_S)
        new_fitnesses.append(best_score)
        
        while len(new_population) < pop_size:
            # Tournament Selection
            i1 = random.randint(0, len(population)-1)
            i2 = random.randint(0, len(population)-1)
            if fitnesses[i1] > fitnesses[i2]:
                p1 = population[i1]
            else:
                p1 = population[i2]
                
            i1 = random.randint(0, len(population)-1)
            i2 = random.randint(0, len(population)-1)
            if fitnesses[i1] > fitnesses[i2]:
                p2 = population[i1]
            else:
                p2 = population[i2]
                
            # Crossover
            if random.random() < 0.8:
                offspring = set()
                for i in range(n):
                    if i in p1:
                        if random.random() < 0.5:
                            offspring.add(i)
                    elif i in p2:
                        if random.random() < 0.5:
                            offspring.add(i)
                    else:
                        if random.random() < 0.02:
                            offspring.add(i)
            else:
                offspring = set(p1)
                
            # Mutation
            num_mutations = int(n * mutation_rate)
            for _ in range(num_mutations):
                x = random.randint(0, n-1)
                if x in offspring:
                    if random.random() < 0.5:
                        offspring.remove(x)
                else:
                    if random.random() < 0.5:
                        offspring.add(x)
                        
            # Repair
            offspring = repair(offspring)
            
            if len(offspring) > 0:
                new_population.append(offspring)
                new_fitnesses.append(count_3aps(offspring))
                
        population = new_population
        fitnesses = new_fitnesses
        
        current_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[current_best_idx] > best_score:
            best_score = fitnesses[current_best_idx]
            best_S = population[current_best_idx]
            
    return sorted([x+1 for x in best_S])

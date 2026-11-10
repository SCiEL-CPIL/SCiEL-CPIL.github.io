import random
import copy

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def poly_square(coeffs):
    """Compute exact square of polynomial given as coeff list (ascending degree)."""
    n = len(coeffs)
    if n == 0:
        return []
    result = [0] * (2*n - 1)
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for k, (i, ci) in enumerate(nz):
        for l in range(k, len(nz)):
            j, cj = nz[l]
            val = ci * cj
            if i == j:
                result[i+j] += val
            else:
                result[i+j] += 2 * val
    return result

def valid_candidate(coeffs):
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    if count_nonzero(coeffs) != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    # Genetic Algorithm Approach
    random.seed(42)
    
    def get_fitness(individual):
        # individual is a list of (exp, coeff)
        # Convert to coeff list
        coeffs = [0]*41
        valid = True
        for e, c in individual:
            if e < 0 or e >= 41:
                valid = False
                break
            if c < -64 or c > 64 or c == 0:
                valid = False
                break
            if coeffs[e] != 0: # Collision
                valid = False
                break
            coeffs[e] = c
        if not valid:
            return float('inf')
        # Ensure length constraint: last coeff non-zero
        # The list 'coeffs' is length 41.
        # We need to trim it? No, valid_candidate checks len<=41 and last non-zero.
        # But if we have a term at index 40, len is 41. If max exp is 30, len is 31.
        # valid_candidate requires len(coeffs) <= 41 and coeffs[-1] != 0.
        # My coeffs list is always length 41.
        # I need to create the trimmed list for valid_candidate and poly_square.
        
        # Find max exp
        max_exp = max(e for e, c in individual)
        trimmed = coeffs[:max_exp+1]
        if trimmed[-1] == 0:
            return float('inf')
        if len(trimmed) > 41:
            return float('inf')
        if count_nonzero(trimmed) != 12:
            return float('inf')
        
        sq = poly_square(trimmed)
        return count_nonzero(sq)

    def init_population(size):
        pop = []
        for _ in range(size):
            # Random exponents
            exps = random.sample(range(41), 12)
            random.shuffle(exps) # Not needed if we sample, but ensures order doesn't matter
            coeffs = [random.choice([-1, 1, -2, 2, -3, 3, -5, 5, -10, 10, -20, 20, -50, 50]) for _ in range(12)]
            ind = list(zip(exps, coeffs))
            pop.append(ind)
        return pop

    def crossover(p1, p2):
        # Uniform crossover
        child = []
        for i in range(12):
            if random.random() < 0.5:
                child.append(p1[i])
            else:
                child.append(p2[i])
        # Ensure unique exponents? 
        # Crossover might produce duplicate exponents.
        # If duplicate, we need to fix it.
        exps = [e for e, c in child]
        if len(set(exps)) < 12:
            # Repair: replace duplicates with random new exps
            seen = set()
            for i, (e, c) in enumerate(child):
                if e in seen:
                    new_e = random.randint(0, 40)
                    while new_e in seen or (new_e, c) in child: # Avoid collision
                         # Simple repair: just pick random
                         new_e = random.randint(0, 40)
                    child[i] = (new_e, c)
                seen.add(child[i][0])
        return child

    def mutate(ind):
        child = ind[:]
        # 1. Change a coefficient
        if random.random() < 0.5:
            i = random.randint(0, 11)
            e, c = child[i]
            new_c = random.randint(-64, 64)
            if new_c == 0: new_c = 1
            child[i] = (e, new_c)
        else:
            # 2. Change an exponent
            i = random.randint(0, 11)
            e, c = child[i]
            new_e = random.randint(0, 40)
            # Check collision
            other_exps = [child[j][0] for j in range(12) if j != i]
            if new_e in other_exps:
                 # Pick a new one that doesn't collide
                 while new_e in other_exps:
                     new_e = random.randint(0, 40)
            child[i] = (new_e, c)
        return child

    # GA Parameters
    pop_size = 100
    generations = 500
    elite_count = 5
    crossover_rate = 0.8
    mutation_rate = 0.3
    
    population = init_population(pop_size)
    best_individual = None
    best_fitness = float('inf')
    
    # Evaluate initial population
    for ind in population:
        f = get_fitness(ind)
        if f < best_fitness:
            best_fitness = f
            best_individual = ind
    
    for gen in range(generations):
        # Sort by fitness
        population.sort(key=lambda x: get_fitness(x))
        
        # Elitism
        new_pop = [population[i] for i in range(elite_count)]
        
        # Generate offspring
        while len(new_pop) < pop_size:
            p1 = random.choice(population[:pop_size//2])
            p2 = random.choice(population[:pop_size//2])
            
            if random.random() < crossover_rate:
                child = crossover(p1, p2)
            else:
                child = p1[:]
            
            if random.random() < mutation_rate:
                child = mutate(child)
            
            new_pop.append(child)
        
        population = new_pop
        
        # Update best
        for ind in population:
            f = get_fitness(ind)
            if f < best_fitness:
                best_fitness = f
                best_individual = ind

    # Convert best_individual to coeff list
    coeffs = [0]*41
    for e, c in best_individual:
        coeffs[e] = c
    
    # Trim
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
        
    if valid_candidate(coeffs):
        return coeffs
    else:
        # Fallback to a known good candidate if GA fails
        return [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]

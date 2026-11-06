import random
import time

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct d > 0 such that a, a+d, a+2d are in the set.
    Uses genetic algorithm with bitset-based scoring for speed.
    """
    
    def score(S):
        """
        Calculate the number of distinct differences d > 0 
        such that a, a+d, a+2d are in S for some a.
        Uses bitset operations for speed.
        """
        bitset = 0
        for v in S:
            bitset |= (1 << (v - 1))
        
        count = 0
        for d in range(1, 256):
            mask = bitset & (bitset >> d) & (bitset >> (2 * d))
            if mask != 0:
                count += 1
        return count

    def make_set():
        """Create a random set of 32 distinct integers in 1..512."""
        return random.sample(range(1, 513), 32)
    
    def crossover(p1, p2):
        """Uniform crossover."""
        child = []
        for i in range(32):
            if random.random() < 0.5:
                child.append(p1[i])
            else:
                child.append(p2[i])
        # Ensure distinctness: if duplicates, replace with random
        seen = set()
        for i in range(32):
            if child[i] in seen:
                while True:
                    child[i] = random.randint(1, 512)
                    if child[i] not in seen:
                        break
            seen.add(child[i])
        return child
    
    def mutate(ind):
        """Mutation: replace 1-3 elements with random values."""
        ind = ind[:]
        num_mut = random.randint(1, 3)
        for _ in range(num_mut):
            idx = random.randint(0, 31)
            while True:
                new_val = random.randint(1, 512)
                if new_val not in ind:
                    break
            ind[idx] = new_val
        return ind
    
    # Initial population
    pop_size = 150
    random.seed(123)
    
    # Create diverse initial population
    population = []
    # Structured starts
    population.append(list(range(1, 33)))
    population.append(list(range(1, 17)) + list(range(257, 273)))
    population.append(list(range(1, 17)) + list(range(129, 145)))
    population.append(list(range(1, 17)) + list(range(193, 209)))
    population.append(list(range(1, 17)) + list(range(321, 337)))
    
    # Random starts
    while len(population) < pop_size:
        population.append(make_set())
    
    # Evaluate initial population
    scores = [score(ind) for ind in population]
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_set = population[best_idx][:]
    best_score = scores[best_idx]
    
    start_time = time.time()
    time_limit = 108.0  # Leave 12s for safety
    
    # Genetic algorithm
    while time.time() - start_time < time_limit:
        # Create new generation
        new_population = []
        
        # Elitism: keep top 20%
        sorted_indices = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        elite_count = pop_size // 5
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]][:])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            t1 = random.randint(0, pop_size - 1)
            t2 = random.randint(0, pop_size - 1)
            if scores[t1] >= scores[t2]:
                p1 = population[t1]
            else:
                p1 = population[t2]
            
            t1 = random.randint(0, pop_size - 1)
            t2 = random.randint(0, pop_size - 1)
            if scores[t1] >= scores[t2]:
                p2 = population[t1]
            else:
                p2 = population[t2]
            
            if random.random() < 0.7:
                child = crossover(p1, p2)
            else:
                child = p1[:]
            
            child = mutate(child)
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        scores = [score(ind) for ind in population]
        
        # Update best
        for i in range(pop_size):
            if scores[i] > best_score:
                best_score = scores[i]
                best_set = population[i][:]
        
        # Occasional restart: if stuck, perturb best and reinsert
        if random.random() < 0.05:
            perturbed = best_set[:]
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, 31)
                while True:
                    new_val = random.randint(1, 512)
                    if new_val not in perturbed:
                        break
                perturbed[idx] = new_val
            # Replace worst individual
            worst_idx = min(range(pop_size), key=lambda i: scores[i])
            population[worst_idx] = perturbed
            scores[worst_idx] = score(perturbed)
            if scores[worst_idx] > best_score:
                best_score = scores[worst_idx]
                best_set = perturbed[:]
    
    return best_set

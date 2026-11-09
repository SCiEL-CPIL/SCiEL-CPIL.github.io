"""Genetic algorithm with bitmask evaluation for max 3-AP differences."""
import time
import random
import math

def _score_bitmask(mask):
    """Count distinct d>0 such that a, a+d, a+2d all in set represented by bitmask (1-indexed)."""
    diffs = set()
    # Extract set elements for pair iteration
    elements = [i + 1 for i in range(512) if (mask >> i) & 1]
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            c = elements[j]
            if (a + c) % 2 == 0:
                b = (a + c) // 2
                if b > a and ((mask >> (b - 1)) & 1):
                    diffs.add((c - a) // 2)
    return len(diffs)

def _score_set(S):
    """Count distinct d>0 such that a, a+d, a+2d all in S."""
    S = set(S)
    diffs = set()
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            a = lst[i]
            c = lst[j]
            if (a + c) % 2 == 0:
                b = (a + c) // 2
                if b in S and b > a:
                    diffs.add((c - a) // 2)
    return len(diffs)

def _set_to_mask(S):
    """Convert set of 1..512 values to bitmask."""
    mask = 0
    for v in S:
        if 1 <= v <= 512:
            mask |= (1 << (v - 1))
    return mask

def _mask_to_list(mask):
    """Convert bitmask to list of values."""
    return [i + 1 for i in range(512) if (mask >> i) & 1]

def _crossover(parent1, parent2):
    """Uniform crossover: for each position, randomly pick from parent1 or parent2."""
    child = []
    used = set()
    for i in range(32):
        if random.random() < 0.5:
            val = parent1[i]
        else:
            val = parent2[i]
        if val in used:
            # Try to fix by picking from the other parent
            alt = parent2[i] if val == parent1[i] else parent1[i]
            if alt not in used:
                val = alt
            else:
                # Find a random unused value
                while True:
                    val = random.randint(1, 512)
                    if val not in used:
                        break
        used.add(val)
        child.append(val)
    return child

def _mutate(individual, mutation_rate=0.15):
    """Replace some elements with random values."""
    individual = list(individual)
    for i in range(32):
        if random.random() < mutation_rate:
            individual[i] = random.randint(1, 512)
    # Fix duplicates
    seen = set()
    for i in range(32):
        if individual[i] in seen:
            while individual[i] in seen:
                individual[i] = random.randint(1, 512)
        seen.add(individual[i])
    return individual

def _make_structured_init(seed=42):
    """Create diverse structured initial solutions."""
    rng = random.Random(seed)
    inits = []
    
    # Grid-based: 4x8 with varying spacing
    for M in [32, 48, 64, 80, 100]:
        S = set()
        for i in range(4):
            for j in range(8):
                val = i * M + j + 1
                if 1 <= val <= 512:
                    S.add(val)
        if len(S) == 32:
            inits.append(list(S))
    
    # Arithmetic progression blocks
    for start in [1, 2, 3, 5]:
        S = set()
        # Two APs of length 16
        for i in range(16):
            S.add(start + i * 2)
        for i in range(16):
            S.add(start + 32 + i * 2)
        if len(S) == 32:
            inits.append(list(S))
    
    # Random structured
    for _ in range(5):
        S = rng.sample(range(1, 513), 32)
        inits.append(S)
    
    return inits

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    rng = random.Random(123)
    
    # Initialize population with diverse structured solutions
    pop_size = 16
    initial_solutions = _make_structured_init(123)
    
    population = []
    scores = []
    
    # Add structured inits
    for sol in initial_solutions[:pop_size]:
        population.append(list(sol))
        scores.append(_score_set(set(sol)))
    
    # Fill remaining with random
    while len(population) < pop_size:
        pop = rng.sample(range(1, 513), 32)
        population.append(pop)
        scores.append(_score_set(set(pop)))
    
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_individual = list(population[best_idx])
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    elite_size = 4
    tournament_size = 3
    generations = 0
    
    while time.time() < deadline:
        generations += 1
        
        # Sort by score
        sorted_indices = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        
        # Track best
        if scores[sorted_indices[0]] > best_score:
            best_score = scores[sorted_indices[0]]
            best_individual = list(population[sorted_indices[0]])
        
        # Create new population
        new_population = []
        new_scores = []
        
        # Elitism: keep top individuals
        for i in range(elite_size):
            idx = sorted_indices[i]
            new_population.append(list(population[idx]))
            new_scores.append(scores[idx])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = sorted_indices[rng.randrange(tournament_size)]
            p2_idx = sorted_indices[rng.randrange(tournament_size)]
            
            # Crossover
            if rng.random() < 0.8:
                child = _crossover(population[p1_idx], population[p2_idx])
            else:
                child = list(population[p1_idx])
            
            # Mutation
            child = _mutate(child, mutation_rate=0.1)
            
            # Evaluate
            child_score = _score_set(set(child))
            new_population.append(child)
            new_scores.append(child_score)
        
        population = new_population
        scores = new_scores
        
        # Periodically inject random individuals for diversity
        if generations % 10 == 0 and time.time() < deadline - 30:
            for _ in range(2):
                rand_pop = rng.sample(range(1, 513), 32)
                rand_score = _score_set(set(rand_pop))
                # Replace worst individual if random is better
                worst_idx = min(range(pop_size), key=lambda i: scores[i])
                if rand_score > scores[worst_idx]:
                    population[worst_idx] = rand_pop
                    scores[worst_idx] = rand_score
    
    # Final local search on best individual
    if time.time() < deadline - 10:
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            cur_set = set(best_individual)
            for i in range(32):
                old_val = best_individual[i]
                for new_val in range(1, 513):
                    if new_val == old_val or new_val in cur_set:
                        continue
                    best_individual[i] = new_val
                    new_score = _score_set(set(best_individual))
                    if new_score > best_score:
                        best_score = new_score
                        changed = True
                        cur_set = set(best_individual)
                    else:
                        best_individual[i] = old_val
    
    return best_individual

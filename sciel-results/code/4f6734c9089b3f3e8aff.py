"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.fft import fft, ifft

# EVOLVE-BLOCK-START

def _score_set_fft(s):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d all in s.
    Uses FFT-based approach: for each d, check if there exists a with a, a+d, a+2d in s.
    
    Key insight: a 3-AP with difference d exists iff there's a pair (x, y) in s 
    with y - x = 2d and (x+y)//2 in s.
    
    For speed, we precompute the indicator array and use vectorized numpy ops.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    
    max_val = max(s_set)
    n = len(s_set)
    lst = sorted(s_set)
    
    # Build indicator array
    arr = np.zeros(max_val + 1, dtype=np.uint8)
    for x in s_set:
        arr[x] = 1
    
    # For each pair (i, j) with i < j, if (j-i) is even, check if mid is in set
    # This is O(n^2) but with n=32, that's only 496 pairs - very fast
    diffs_found = set()
    for i in range(n):
        for j in range(i + 1, n):
            diff = lst[j] - lst[i]
            if diff % 2 == 0:
                mid = (lst[i] + lst[j]) // 2
                if arr[mid]:
                    diffs_found.add(diff // 2)
    
    return len(diffs_found)


def _score_set_bitset(s):
    """
    Fast scoring using Python integer bitsets.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    
    max_val = max(s_set)
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    diffs_found = set()
    # d ranges from 1 to max_val//2
    # For each d, check if (bitset & (bitset >> d) & (bitset >> (2*d))) != 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs_found.add(d)
    
    return len(diffs_found)


def _score_set_fast(s):
    """
    Hybrid fast scoring: use the pair-based method which is O(n^2) with n=32.
    This is extremely fast for our small set size.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    
    max_val = max(s_set)
    arr = np.zeros(max_val + 1, dtype=bool)
    for x in s_set:
        arr[x] = True
    
    lst = sorted(s_set)
    n = len(lst)
    diffs_found = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = lst[j] - lst[i]
            if diff % 2 == 0:
                mid = (lst[i] + lst[j]) // 2
                if arr[mid]:
                    diffs_found.add(diff // 2)
    
    return len(diffs_found)


def _ga_search(deadline):
    """
    Genetic Algorithm with FFT-speed scoring to explore the search space thoroughly.
    """
    random.seed(42)
    
    pop_size = 200
    elite_count = 10
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    # Initialize population with diverse starting points
    population = []
    
    # Add some structured initial solutions
    # 1. Dense interval
    population.append(list(range(1, 33)))
    # 2. Another interval
    population.append(list(range(10, 42)))
    # 3. Geometric-like
    geo = [int(1 + i * 15) for i in range(32)]
    geo = [min(x, 512) for x in geo]
    geo = list(set(geo))
    while len(geo) < 32:
        geo.append(random.randint(1, 512))
    population.append(list(set(geo))[:32])
    
    # 4. Random sets
    for _ in range(pop_size - 3):
        s = set()
        while len(s) < 32:
            s.add(random.randint(1, 512))
        population.append(list(s))
    
    # Evaluate initial population
    scores = [_score_set_fast(ind) for ind in population]
    
    best_idx = np.argmax(scores)
    best_set = list(population[best_idx])
    best_score = scores[best_idx]
    
    generation = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness
        sorted_pairs = sorted(zip(scores, population), key=lambda x: -x[0])
        scores = [p[0] for p in sorted_pairs]
        population = [p[1] for p in sorted_pairs]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_set = list(population[0])
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stagnant for too long, increase diversity
        if no_improve_count > 50:
            # Inject random immigrants
            for _ in range(20):
                s = set()
                while len(s) < 32:
                    s.add(random.randint(1, 512))
                population[-1] = list(s)
                population = population[:-1]
                population.append(list(s))
            no_improve_count = 0
            # Re-evaluate modified population
            for i in range(len(population)):
                population[i] = list(set(population[i]))
                while len(population[i]) < 32:
                    population[i].append(random.randint(1, 512))
                population[i] = list(set(population[i]))[:32]
            scores = [_score_set_fast(ind) for ind in population]
            continue
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(elite_count):
            new_population.append(list(population[i]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Select parents (tournament selection)
            p1 = _tournament_select(population, scores, 5)
            p2 = _tournament_select(population, scores, 5)
            
            if random.random() < crossover_rate:
                child = _crossover(p1, p2)
            else:
                child = list(p1)
            
            # Mutation
            child = _mutate(child, mutation_rate)
            
            # Ensure validity
            child = _ensure_valid(child)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
        
        # Evaluate new population
        for i in range(len(population)):
            population[i] = list(set(population[i]))
            while len(population[i]) < 32:
                population[i].append(random.randint(1, 512))
            population[i] = list(set(population[i]))[:32]
        
        scores = [_score_set_fast(ind) for ind in population]
    
    return best_set, best_score


def _tournament_select(population, scores, k):
    """Tournament selection."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: scores[i])
    return population[best_idx]


def _crossover(p1, p2):
    """Uniform crossover for sets."""
    s1 = set(p1)
    s2 = set(p2)
    child = set()
    for x in s1:
        if x in s2:
            child.add(x)
        else:
            if random.random() < 0.5:
                child.add(x)
    for x in s2:
        if x not in s1:
            if random.random() < 0.5:
                child.add(x)
    # Ensure exactly 32 elements
    while len(child) < 32:
        child.add(random.randint(1, 512))
    if len(child) > 32:
        child = set(random.sample(list(child), 32))
    return list(child)


def _mutate(individual, rate):
    """Mutation: randomly replace some elements."""
    s = set(individual)
    n_mutate = max(1, int(len(s) * rate))
    for _ in range(n_mutate):
        to_remove = random.choice(list(s))
        s.remove(to_remove)
        s.add(random.randint(1, 512))
    # Ensure validity
    s = set(s)
    while len(s) < 32:
        s.add(random.randint(1, 512))
    if len(s) > 32:
        s = set(random.sample(list(s), 32))
    return list(s)


def _ensure_valid(individual):
    """Ensure exactly 32 distinct integers in 1..512."""
    s = set(individual)
    s = {x for x in s if 1 <= x <= 512}
    while len(s) < 32:
        s.add(random.randint(1, 512))
    if len(s) > 32:
        s = set(random.sample(list(s), 32))
    return list(s)


def _local_search_polish(s, deadline):
    """
    Final local search polish on the best solution found by GA.
    Try all single-element swaps to see if we can improve.
    """
    s = set(s)
    current_score = _score_set_fast(s)
    best_set = list(s)
    best_score = current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s_list = sorted(s)
        
        for i in range(len(s_list)):
            if time.time() > deadline:
                break
            removed = s_list[i]
            for new_elem in range(1, 513):
                if new_elem in s:
                    continue
                
                new_set = s - {removed} | {new_elem}
                new_score = _score_set_fast(new_set)
                
                if new_score > current_score:
                    s = new_set
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_set = list(s)
                    break
            if improved:
                break
    
    return best_set, best_score


def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Phase 1: Genetic Algorithm for broad exploration
    ga_set, ga_score = _ga_search(deadline - 20.0)  # Leave 20s for polish
    
    # Phase 2: Local search polish
    final_set, final_score = _local_search_polish(ga_set, deadline)
    
    return final_set

# EVOLVE-BLOCK-END

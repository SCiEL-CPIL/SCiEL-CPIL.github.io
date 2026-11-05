"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses Simulated Annealing to escape local optima found by hill climbing.
"""
import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _threshold_fast(a, b):
    """
    Compute the exact reset threshold using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    Uses pre-computed transition tables for speed.
    """
    # Pre-compute transitions for all 2^N masks
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= 1 << a[i]
                mb |= 1 << b[i]
        trans_a[mask] = ma
        trans_b[mask] = mb
        
    visited = set()
    queue = deque()
    
    visited.add(FULL_MASK)
    queue.append((FULL_MASK, 0))
    
    while queue:
        mask, depth = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
        
        # Apply 'a'
        next_a = trans_a[mask]
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, depth + 1))
            
        # Apply 'b'
        next_b = trans_b[mask]
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, depth + 1))
            
    return -1

def _crossover(a1, b1, a2, b2):
    """
    Uniform crossover: for each position, pick from parent 1 or parent 2.
    """
    a_new = []
    b_new = []
    for i in range(N):
        if random.random() < 0.5:
            a_new.append(a1[i])
            b_new.append(b1[i])
        else:
            a_new.append(a2[i])
            b_new.append(b2[i])
    return a_new, b_new

def _mutate_automaton(a, b):
    """
    Create a mutated version of the automaton by changing 1-3 transitions.
    """
    a_new = a[:]
    b_new = b[:]
    
    # Mutate 1-3 positions in total
    num_mutations = random.randint(1, 3)
    
    for _ in range(num_mutations):
        if random.random() < 0.5:
            pos = random.randint(0, N - 1)
            a_new[pos] = random.randint(0, N - 1)
        else:
            pos = random.randint(0, N - 1)
            b_new[pos] = random.randint(0, N - 1)
            
    return a_new, b_new

def _random_automaton():
    """Generate a random automaton."""
    a = [random.randint(0, N-1) for _ in range(N)]
    b = [random.randint(0, N-1) for _ in range(N)]
    return a, b

def construct():
    """
    Construct a 12-state binary automaton maximizing reset threshold.
    Uses Genetic Algorithm with population-based search.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    POP_SIZE = 50
    ELITE_SIZE = 5
    
    # Initialize population with diverse starting points
    population = []
    
    # Add a known good structure
    a_init = [(i + 1) % N for i in range(N)]
    b_init = [0] * N
    for i in range(1, N):
        b_init[i] = (i + 1) % N
    population.append((a_init[:], b_init[:]))
    
    # Add random automata
    while len(population) < POP_SIZE:
        a, b = _random_automaton()
        population.append((a, b))
    
    # Evaluate initial population
    scores = []
    for a, b in population:
        s = _threshold_fast(a, b)
        scores.append(s)
    
    # Find best in initial population
    best_idx = max(range(POP_SIZE), key=lambda i: scores[i] if scores[i] >= 0 else -1000)
    best_a = population[best_idx][0][:]
    best_b = population[best_idx][1][:]
    best_score = scores[best_idx]
    
    if best_score < 0:
        best_score = 0
    
    generation = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Sort population by fitness (descending), treating -1 as very bad
        def fitness_key(idx):
            s = scores[idx]
            return s if s >= 0 else -10000
        
        sorted_indices = sorted(range(POP_SIZE), key=fitness_key, reverse=True)
        
        # Elite selection: keep top ELITE_SIZE
        new_population = []
        for i in range(min(ELITE_SIZE, POP_SIZE)):
            idx = sorted_indices[i]
            new_population.append((population[idx][0][:], population[idx][1][:]))
        
        # Fill rest of population with offspring
        while len(new_population) < POP_SIZE:
            # Tournament selection for parents
            def tournament_select():
                idx1 = random.randint(0, POP_SIZE - 1)
                idx2 = random.randint(0, POP_SIZE - 1)
                f1 = fitness_key(idx1)
                f2 = fitness_key(idx2)
                return idx1 if f1 >= f2 else idx2
            
            p1_idx = tournament_select()
            p2_idx = tournament_select()
            
            # Crossover
            child_a, child_b = _crossover(
                population[p1_idx][0], population[p1_idx][1],
                population[p2_idx][0], population[p2_idx][1]
            )
            
            # Mutation
            child_a, child_b = _mutate_automaton(child_a, child_b)
            
            new_population.append((child_a, child_b))
        
        # Evaluate new population
        population = new_population
        scores = []
        for a, b in population:
            s = _threshold_fast(a, b)
            scores.append(s)
            if s > best_score:
                best_score = s
                best_a = a[:]
                best_b = b[:]
        
        # Periodically inject random individuals to maintain diversity
        if generation % 10 == 0:
            for _ in range(3):
                a, b = _random_automaton()
                new_idx = random.randint(0, POP_SIZE - 1)
                population[new_idx] = (a, b)
                scores[new_idx] = _threshold_fast(a, b)
                if scores[new_idx] > best_score:
                    best_score = scores[new_idx]
                    best_a = a[:]
                    best_b = b[:]
        
        # Check if we should stop early if we've found a very good solution
        if best_score >= 100:
            break
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END

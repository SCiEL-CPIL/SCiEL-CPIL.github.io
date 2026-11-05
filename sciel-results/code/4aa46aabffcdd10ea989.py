import time
import random
import sys
from collections import deque

N = 12
FULL_STATE = (1 << N) - 1
DEADLINE = 50.0  # Safety margin under 60s kill

def calculate_threshold(a_trans, b_trans):
    """
    Returns the shortest reset word length for the automaton defined by a_trans and b_trans,
    or -1 if not synchronizing.
    """
    a_bits = [1 << a_trans[i] for i in range(N)]
    b_bits = [1 << b_trans[i] for i in range(N)]
    
    # Precompute transition tables for all possible subsets
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        if mask == 0:
            continue
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= a_bits[idx]
            res_b |= b_bits[idx]
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b
        
    # BFS
    visited = [False] * (1 << N)
    visited[FULL_STATE] = True
    queue = deque()
    queue.append((FULL_STATE, 0))
    
    while queue:
        state, dist = queue.popleft()
        if state.bit_count() == 1:
            return dist
            
        next_a = trans_a[state]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
            
        next_b = trans_b[state]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    
    def get_score(a, b):
        return calculate_threshold(a, b)

    # Initialization: Start with a mix of random and structured (Pinsker-like)
    def random_automaton():
        return [random.randint(0, N-1) for _ in range(N)]

    # Pinsker-like initialization
    def pinsker_automaton():
        a = [(i + 1) % N for i in range(N)]
        # b: 0->0, i->i-1 for i>0 is not always best, let's try a few variants
        b1 = [0] + [i-1 for i in range(1, N)]
        b2 = [(i - 1) % N for i in range(N)]
        b3 = [0] * N
        return a, b1, b2, b3

    population = []
    
    # Add some structured starts
    a_p, b_p1, b_p2, b_p3 = pinsker_automaton()
    population.append([a_p[:], b_p1])
    population.append([a_p[:], b_p2])
    population.append([a_p[:], b_p3])
    
    # Add random starts
    while len(population) < 30:
        a = random_automaton()
        b = random_automaton()
        population.append([a, b])
    
    # Evaluate initial population
    scores = []
    for ind in population:
        scores.append(get_score(ind[0], ind[1]))
    
    best_score = max(scores)
    best_idx = scores.index(best_score)
    best_a = population[best_idx][0][:]
    best_b = population[best_idx][1][:]
    
    # Genetic Algorithm Loop
    pop_size = len(population)
    
    while time.time() - start_time < DEADLINE:
        # Sort population by score (descending)
        # We need to re-evaluate if we mutated, but we keep scores from previous gen for selection
        # Actually, it's better to keep (a, b, score) tuples
        
        # Sort current population
        sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        
        # Selection: Keep top 50%
        elite_count = pop_size // 2
        new_population = []
        
        # Copy elites (deep copy)
        for i in range(elite_count):
            a, b = sorted_pop[i][0]
            new_population.append([a[:], b[:]])
            
        # Generate offspring to fill the rest
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = random.randint(0, elite_count - 1)
            p2_idx = random.randint(0, elite_count - 1)
            
            # If they are the same, pick another
            if p1_idx == p2_idx:
                p2_idx = (p2_idx + 1) % elite_count
                
            parent1 = sorted_pop[p1_idx][0]
            parent2 = sorted_pop[p2_idx][0]
            
            # Crossover
            child_a = parent1[0][:]
            child_b = parent1[1][:]
            
            # Uniform crossover for a
            for i in range(N):
                if random.random() < 0.5:
                    child_a[i] = parent2[0][i]
            # Uniform crossover for b
            for i in range(N):
                if random.random() < 0.5:
                    child_b[i] = parent2[1][i]
            
            # Mutation
            if random.random() < 0.5:
                i = random.randint(0, N-1)
                child_a[i] = random.randint(0, N-1)
            if random.random() < 0.5:
                i = random.randint(0, N-1)
                child_b[i] = random.randint(0, N-1)
                
            new_population.append([child_a, child_b])
            
        population = new_population
        scores = []
        
        # Evaluate new population
        for ind in population:
            s = get_score(ind[0], ind[1])
            scores.append(s)
            if s > best_score:
                best_score = s
                best_idx = len(scores) - 1
                best_a = ind[0][:]
                best_b = ind[1][:]
                
        # Early termination if we hit the theoretical max?
        if best_score == (N-1)**2:
            break
            
    if best_a is None:
        # Fallback
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        
    return [best_a, best_b]

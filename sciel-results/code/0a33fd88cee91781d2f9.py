import time
import random
import sys

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns -1 if not synchronizing.
    """
    # Precompute transition of subsets
    # subset is an integer bitmask
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # Optimization: Precompute bit mapping for all possible masks? 
    # N=12, so 4096 masks. 4096 * 12 operations is ~50k ops. Very fast.
    for mask in range(1, 1 << N):
        ma = 0
        mb = 0
        # Iterate through bits
        # A faster way might be to iterate through set bits, but N is small.
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << trans_a[i])
                mb |= (1 << trans_b[i])
        next_a[mask] = ma
        next_b[mask] = mb
    
    # BFS on subset automaton
    # Start state is FULL_MASK (all states)
    # Target state is any singleton mask (1 << i)
    
    visited = [False] * (1 << N)
    queue = [(FULL_MASK, 0)]
    visited[FULL_MASK] = True
    
    while queue:
        mask, depth = queue.pop(0)
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
            
        # Expand
        ma = next_a[mask]
        mb = next_b[mask]
        
        if not visited[ma]:
            visited[ma] = True
            queue.append((ma, depth + 1))
            
        if not visited[mb]:
            visited[mb] = True
            queue.append((mb, depth + 1))
            
    return -1

def evaluate_individual(ind):
    """
    ind is a list of 24 ints (12 for a, 12 for b).
    Returns reset length or -1.
    """
    trans_a = ind[:N]
    trans_b = ind[N:]
    return compute_threshold(trans_a, trans_b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Initial population
    pop_size = 100
    population = []
    
    # Seed with some known structures? 
    # 1. The seed program's automaton
    seed_a = [(i + 1) % N for i in range(N)]
    seed_b = [0] * N
    population.append(seed_a + seed_b)
    
    # 2. Random initialization
    while len(population) < pop_size:
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        population.append(a + b)
        
    best_score = -1
    best_ind = None
    
    # Evaluate initial population
    for ind in population:
        score = evaluate_individual(ind)
        if score > best_score:
            best_score = score
            best_ind = ind[:]
            
    # Evolutionary loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        # Sort population by score (descending)
        # We need scores for sorting. Let's store (score, ind)
        scored_pop = []
        for ind in population:
            # Optimization: If we already know the score, reuse? 
            # For simplicity, re-evaluate. N=12 BFS is very fast (<1ms).
            # 100 individuals * 1ms = 0.1s per generation. 
            # 85s / 0.1s = ~850 generations. Plenty.
            score = evaluate_individual(ind)
            scored_pop.append((score, ind))
            
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_ind = scored_pop[0][1][:]
            
        # Selection: Keep top 20% (elite)
        elite_count = max(2, pop_size // 5)
        elites = [ind for _, ind in scored_pop[:elite_count]]
        
        # Generate offspring
        new_population = []
        
        # Add elites
        for ind in elites:
            new_population.append(ind[:])
            
        # Fill rest with mutated/crossovered individuals
        while len(new_population) < pop_size:
            # Crossover
            if random.random() < 0.7:
                # Tournament selection for parents
                p1 = random.choice(elites)
                p2 = random.choice(elites)
                
                # Uniform crossover
                child = []
                for i in range(2 * N):
                    if random.random() < 0.5:
                        child.append(p1[i])
                    else:
                        child.append(p2[i])
            else:
                # Clone from elite
                p1 = random.choice(elites)
                child = p1[:]
                
            # Mutation
            # Probability of mutating each gene
            p_mutate = 0.05
            for i in range(2 * N):
                if random.random() < p_mutate:
                    # Mutate to a random state
                    child[i] = random.randint(0, N - 1)
                    
            new_population.append(child)
            
        population = new_population
        
    # Return the best found
    # Ensure it is synchronizing (score > 0)
    if best_score > 0:
        return [best_ind[:N], best_ind[N:]]
    else:
        # Fallback to seed if no synchronizing automaton found (unlikely)
        return [[(i + 1) % N for i in range(N)], [0] * N]

# EVOLVE-BLOCK-END

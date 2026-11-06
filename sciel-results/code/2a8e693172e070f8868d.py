"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

N = 256

def count_mono_4ap_fast(coloring):
    """
    Count monochromatic 4-term arithmetic progressions.
    Uses numpy vectorization for speed.
    """
    c = np.array(coloring, dtype=np.int8)
    count = 0
    # d is the difference. a is the start.
    # a, a+d, a+2d, a+3d must be in [0, N-1]
    # Max d is such that 3d < N => d < N/3
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        # Extract sequences
        # Indices: 0..max_a-1 for a
        # a+d: d..d+max_a-1
        # a+2d: 2d..2d+max_a-1
        # a+3d: 3d..3d+max_a-1
        seq1 = c[:max_a]
        seq2 = c[d:d+max_a]
        seq3 = c[2*d:2*d+max_a]
        seq4 = c[3*d:3*d+max_a]
        
        # Check if all are equal
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return int(count)

def flip_delta(coloring, pos):
    """
    Calculate the change in the number of monochromatic 4-APs when flipping coloring[pos].
    Returns the delta (change in count).
    """
    c = coloring
    delta = 0
    p = pos
    cp = c[p]
    
    # Role 1: p is 1st term (index i)
    # AP: i, i+d, i+2d, i+3d
    # Other 3: i+d, i+2d, i+3d
    for d in range(1, (N - p - 1) // 3 + 1):
        i1 = p + d
        i2 = p + 2*d
        i3 = p + 3*d
        if i3 >= N: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    # Role 2: p is 2nd term (index i+d)
    # AP: i, i+d, i+2d, i+3d where i+d = p => i = p-d
    # Other 3: p-d, p+d, p+2d
    for d in range(1, min(p, (N - p - 1) // 2) + 1):
        i1 = p - d
        i2 = p + d
        i3 = p + 2*d
        if i1 < 0 or i3 >= N: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    # Role 3: p is 3rd term (index i+2d)
    # AP: i, i+d, i+2d, i+3d where i+2d = p => i = p-2d
    # Other 3: p-2d, p-d, p+d
    for d in range(1, min(p // 2, N - p - 1) + 1):
        i1 = p - 2*d
        i2 = p - d
        i3 = p + d
        if i1 < 0 or i3 >= N: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    # Role 4: p is 4th term (index i+3d)
    # AP: i, i+d, i+2d, i+3d where i+3d = p => i = p-3d
    # Other 3: p-3d, p-2d, p-d
    for d in range(1, p // 3 + 1):
        i1 = p - 3*d
        i2 = p - 2*d
        i3 = p - d
        if i1 < 0: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    return delta

def local_search(coloring, deadline, max_passes=20):
    """
    Perform local search to refine a coloring.
    """
    c = coloring[:]
    current_score = count_mono_4ap_fast(c)
    best_score = current_score
    best_c = c[:]
    
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() < deadline - 2:
        improved = False
        passes += 1
        order = list(range(N))
        random.shuffle(order)
        
        for pos in order:
            if time.time() > deadline - 2:
                return best_c, best_score
            
            delta = flip_delta(c, pos)
            if delta < 0:
                c[pos] = 1 - c[pos]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c[:]
                improved = True
    
    return best_c, best_score

def crossover(parent1, parent2):
    """
    Uniform crossover.
    """
    child = [0] * N
    for i in range(N):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def mutate(coloring, mutation_rate=0.05):
    """
    Bit-flip mutation.
    """
    child = coloring[:]
    for i in range(N):
        if random.random() < mutation_rate:
            child[i] = 1 - child[i]
    return child

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Initialize population with diverse strategies
    population = []
    scores = []
    
    # 1. Thue-Morse
    tm = [bin(i).count('1') % 2 for i in range(N)]
    population.append(tm)
    scores.append(count_mono_4ap_fast(tm))
    
    # 2. Random
    for _ in range(10):
        r = [random.randint(0, 1) for _ in range(N)]
        population.append(r)
        scores.append(count_mono_4ap_fast(r))
        
    # 3. Block patterns
    for block_size in [2, 4, 8, 16]:
        b = [(i // block_size) % 2 for i in range(N)]
        population.append(b)
        scores.append(count_mono_4ap_fast(b))
        
    # 4. Alternating
    alt = [i % 2 for i in range(N)]
    population.append(alt)
    scores.append(count_mono_4ap_fast(alt))
    
    # Sort population by score
    pop_with_scores = list(zip(scores, population))
    pop_with_scores.sort(key=lambda x: x[0])
    
    best_score = pop_with_scores[0][0]
    best_c = pop_with_scores[0][1]
    
    # Genetic Algorithm Loop
    pop_size = len(population)
    # We will maintain a population of size ~20-30
    # Target: 20 individuals
    
    while time.time() < deadline - 5:
        # Selection: Tournament selection
        new_population = []
        
        # Keep the best individual (Elitism)
        new_population.append(pop_with_scores[0][1])
        
        # Generate offspring
        while len(new_population) < 20 and time.time() < deadline - 5:
            # Select 2 parents via tournament
            # Tournament size 3
            p1_idx = random.randint(0, min(10, len(pop_with_scores)-1))
            p2_idx = random.randint(0, min(10, len(pop_with_scores)-1))
            
            # Better selection: pick from top 10
            candidates = pop_with_scores[:10]
            p1 = candidates[random.randint(0, len(candidates)-1)][1]
            p2 = candidates[random.randint(0, len(candidates)-1)][1]
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            child = mutate(child, mutation_rate=0.02)
            
            # Local search refinement (limited passes to save time)
            # We only do local search on the child if we have time
            if time.time() < deadline - 10:
                refined_child, refined_score = local_search(child, deadline - 2, max_passes=5)
                child = refined_child
                score = refined_score
            else:
                score = count_mono_4ap_fast(child)
                
            new_population.append(child)
            
        # Update population
        # Evaluate new population
        new_pop_with_scores = []
        for ind in new_population:
            s = count_mono_4ap_fast(ind)
            new_pop_with_scores.append((s, ind))
            
        # Merge and keep top 20
        all_pop = pop_with_scores + new_pop_with_scores
        all_pop.sort(key=lambda x: x[0])
        pop_with_scores = all_pop[:20]
        
        # Update best
        if pop_with_scores[0][0] < best_score:
            best_score = pop_with_scores[0][0]
            best_c = pop_with_scores[0][1]
            
        # If we have time, do a deeper local search on the current best
        if time.time() < deadline - 15:
            deep_c, deep_score = local_search(best_c, deadline - 2, max_passes=50)
            if deep_score < best_score:
                best_score = deep_score
                best_c = deep_c

    return best_c
# EVOLVE-BLOCK-END

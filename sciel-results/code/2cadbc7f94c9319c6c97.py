import numpy as np
import time
import random
import sys

N = 256

def count_mono_4ap_fast(coloring):
    """Faster count using numpy vectorization per d."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N):
        max_a = N - 3*d
        if max_a < 1:
            break
        # Extract the four sequences
        seq1 = c[:max_a]           # a-1 for a=1..max_a
        seq2 = c[d:d+max_a]
        seq3 = c[2*d:2*d+max_a]
        seq4 = c[3*d:3*d+max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return int(count)

def flip_delta(coloring, pos):
    """
    Calculate the change in the number of monochromatic 4-APs when flipping coloring[pos].
    Returns delta.
    """
    c = coloring
    delta = 0
    p = pos
    cp = c[p]
    
    # Role 1: p is 1st term (index i, value i+1)
    # AP: i, i+d, i+2d, i+3d
    # Constraints: i+3d < N
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

    # Role 2: p is 2nd term (index i+d, value i+d+1)
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

    # Role 3: p is 3rd term (index i+2d, value i+2d+1)
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

    # Role 4: p is 4th term (index i+3d, value i+3d+1)
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

def local_search_incremental(coloring, deadline, max_passes=500):
    """
    Perform local search using incremental delta updates.
    coloring is a list of 0s and 1s.
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
        # Randomize order to avoid bias
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

def crossover(p1, p2):
    """Uniform crossover: pick each bit from p1 or p2 with 50% probability."""
    return [random.choice([p1[i], p2[i]]) for i in range(N)]

def mutate(ind, rate=0.05):
    """Bit flip mutation."""
    new_ind = ind[:]
    for i in range(N):
        if random.random() < rate:
            new_ind[i] = 1 - new_ind[i]
    return new_ind

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Generate diverse starting points
    starting_points = []
    
    # 1. Thue-Morse sequence
    tm = [bin(i).count('1') % 2 for i in range(N)]
    starting_points.append(tm)
    
    # 2. Random
    for _ in range(10):
        starting_points.append([random.randint(0, 1) for _ in range(N)])
        
    # 3. Block patterns
    for block_size in [2, 4, 8, 16, 32, 64]:
        starting_points.append([(i // block_size) % 2 for i in range(N)])
        
    # 4. Alternating
    starting_points.append([i % 2 for i in range(N)])
    
    # 5. Greedy construction
    greedy_c = [-1] * N
    for j in range(N):
        count0 = 0
        count1 = 0
        for d in range(1, (j // 3) + 1):
            a = j - 3*d
            if a < 0: break
            c1 = greedy_c[a]
            c2 = greedy_c[a+d]
            c3 = greedy_c[a+2*d]
            if c1 != -1 and c2 != -1 and c3 != -1:
                if c1 == c2 == c3:
                    if c1 == 0:
                        count0 += 1
                    else:
                        count1 += 1
        if count0 < count1:
            greedy_c[j] = 0
        elif count1 < count0:
            greedy_c[j] = 1
        else:
            greedy_c[j] = random.randint(0, 1)
    starting_points.append(greedy_c)

    # Initialize population
    pop_size = 20
    population = []
    scores = []
    
    for sp in starting_points[:pop_size]:
        c_res, score_res = local_search_incremental(sp, deadline - 5, max_passes=50)
        population.append(c_res)
        scores.append(score_res)
        
    # If we have fewer than pop_size, fill with random
    while len(population) < pop_size:
        sp = [random.randint(0, 1) for _ in range(N)]
        c_res, score_res = local_search_incremental(sp, deadline - 5, max_passes=50)
        population.append(c_res)
        scores.append(score_res)

    best_overall_c = population[0]
    best_overall_score = min(scores)
    
    # Sort population by score
    sorted_indices = np.argsort(scores)
    population = [population[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Genetic Algorithm Loop
    generation = 0
    while time.time() < deadline - 20:
        generation += 1
        new_population = []
        new_scores = []
        
        # Elitism: keep the best 2
        for i in range(2):
            new_population.append(population[i])
            new_scores.append(scores[i])
            if scores[i] < best_overall_score:
                best_overall_score = scores[i]
                best_overall_c = population[i]
                
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            if scores[p1_idx] < scores[p2_idx]:
                p1 = population[p1_idx]
            else:
                p1 = population[p2_idx]
                
            p3_idx = random.randint(0, pop_size - 1)
            p4_idx = random.randint(0, pop_size - 1)
            if scores[p3_idx] < scores[p4_idx]:
                p2 = population[p3_idx]
            else:
                p2 = population[p4_idx]
                
            child = crossover(p1, p2)
            child = mutate(child, rate=0.02)
            
            # Local search on child
            c_res, score_res = local_search_incremental(child, deadline - 10, max_passes=20)
            new_population.append(c_res)
            new_scores.append(score_res)
            
            if score_res < best_overall_score:
                best_overall_score = score_res
                best_overall_c = c_res
                
        population = new_population
        scores = new_scores
        
        # Re-sort
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

    # Final deep dive on the best found
    if best_overall_c is not None and time.time() < deadline - 5:
        final_c, final_score = local_search_incremental(best_overall_c, deadline - 2, max_passes=500)
        if final_score < best_overall_score:
            best_overall_c = final_c
            best_overall_score = final_score

    return best_overall_c

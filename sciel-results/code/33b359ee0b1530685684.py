import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)
    
    # For each position, find which APs contain it
    pos_to_aps = [[] for _ in range(n)]
    for i, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append(i)
        pos_to_aps[b].append(i)
        pos_to_aps[c].append(i)
        pos_to_aps[d].append(i)
    
    def count_mono(color):
        c0 = color[aps_arr[:, 0]]
        c1 = color[aps_arr[:, 1]]
        c2 = color[aps_arr[:, 2]]
        c3 = color[aps_arr[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))
    
    def delta_flip(color, pos):
        """Compute change in mono count when flipping position pos."""
        # Get APs containing pos
        ap_indices = pos_to_aps[pos]
        if not ap_indices:
            return 0
        
        # For each such AP, determine if it was mono before and after flip
        old_mono = 0
        new_mono = 0
        for idx in ap_indices:
            a, b, c, d = aps_arr[idx]
            ca, cb, cc, cd = color[a], color[b], color[c], color[d]
            # Was it mono?
            if ca == cb == cc == cd:
                old_mono += 1
            # After flip
            new_ca = 1 - ca if a == pos else ca
            new_cb = 1 - cb if b == pos else cb
            new_cc = 1 - cc if c == pos else cc
            new_cd = 1 - cd if d == pos else cd
            if new_ca == new_cb == new_cc == new_cd:
                new_mono += 1
        return new_mono - old_mono
    
    def local_search(color, max_iters=200):
        color = color.copy()
        current = count_mono(color)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            best_improve = 0
            best_pos = -1
            for pos in range(n):
                delta = delta_flip(color, pos)
                improve = -delta  # we want to decrease count
                if improve > best_improve:
                    best_improve = improve
                    best_pos = pos
            if best_pos >= 0:
                color[best_pos] = 1 - color[best_pos]
                current += delta_flip(color, best_pos)  # recompute after flip
                # Actually, let's just recompute
                current = count_mono(color)
            else:
                break
        return color, current
    
    # Thue-Morse initialization
    thue_morse = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    tm_score = count_mono(thue_morse)
    
    # Generate diverse initial population
    pop_size = 15
    population = [thue_morse.copy()]
    for _ in range(pop_size - 1):
        population.append(np.random.randint(0, 2, n).astype(np.int8))
    
    # Evaluate and local search on a few
    best_color = thue_morse.copy()
    best_score = tm_score
    
    # Local search on Thue-Morse
    ls_color, ls_score = local_search(thue_morse, max_iters=100)
    if ls_score < best_score:
        best_score = ls_score
        best_color = ls_color
    
    # Genetic algorithm with local search on elites
    gen = 0
    while time.time() < deadline - 15:
        gen += 1
        
        # Evaluate population
        scores = [count_mono(c) for c in population]
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_color = population[best_idx].copy()
        
        # Local search on top 3
        for i in np.argsort(scores)[:3]:
            if time.time() > deadline - 10:
                break
            ls_c, ls_s = local_search(population[i], max_iters=50)
            if ls_s < best_score:
                best_score = ls_s
                best_color = ls_c
            population[i] = ls_c
            scores[i] = ls_s
        
        # Create new population
        new_pop = [best_color.copy()]
        
        for _ in range(pop_size - 1):
            # Tournament selection
            idx1, idx2 = random.sample(range(pop_size), 2)
            p1 = population[idx1] if scores[idx1] <= scores[idx2] else population[idx2]
            idx1, idx2 = random.sample(range(pop_size), 2)
            p2 = population[idx1] if scores[idx1] <= scores[idx2] else population[idx2]
            
            # Uniform crossover
            mask = np.random.randint(0, 2, n)
            child = np.where(mask == 1, p1, p2).astype(np.int8)
            
            # Mutation
            n_mut = random.randint(1, 3)
            for _ in range(n_mut):
                child[random.randint(0, n - 1)] = 1 - child[random.randint(0, n - 1)]
            
            new_pop.append(child)
        
        population = new_pop
    
    # Final local search
    best_color, best_score = local_search(best_color, max_iters=300)
    
    return [int(c) for c in best_color]

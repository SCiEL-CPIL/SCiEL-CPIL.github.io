import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP: a, a+d, a+2d, a+3d with a >= 1, a+3d <= N, d >= 1
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # Precompute for each position, the indices of the other 3 elements in every AP containing it.
    # pos_ap_others[p] is a numpy array of shape (K, 3), where K is the number of APs containing p.
    pos_ap_others = [None] * N
    for p in range(N):
        others_list = []
        for i in range(total_aps):
            row = aps[i]
            # Check if p is in this AP
            if p in row:
                # Find the index of p in the row to exclude it
                idx = np.where(row == p)[0][0]
                others = [row[j] for j in range(4) if j != idx]
                others_list.append(others)
        if others_list:
            pos_ap_others[p] = np.array(others_list, dtype=np.int32)
        else:
            pos_ap_others[p] = np.empty((0, 3), dtype=np.int32)

    def count_mono(colors):
        """Count monochromatic 4-APs using numpy. colors is numpy array of 0/1."""
        c = colors[aps]  # shape (total_aps, 4)
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def fast_delta(colors, p):
        """Compute change in mono count if we flip position p using precomputed others."""
        old_color = colors[p]
        others = pos_ap_others[p]
        n_aps = others.shape[0]
        if n_aps == 0:
            return 0
        
        # Get colors of the other 3 positions in each AP
        # others has shape (n_aps, 3)
        other_colors = colors[others] # shape (n_aps, 3)
        
        # Check if the 3 other positions are all the same color
        all_same = (other_colors[:, 0] == other_colors[:, 1]) & (other_colors[:, 1] == other_colors[:, 2])
        
        # Before flip: AP is mono if all 3 others are same AND that color is old_color
        mono_before = all_same & (other_colors[:, 0] == old_color)
        
        # After flip: AP is mono if all 3 others are same AND that color is new_color (1 - old_color)
        new_color = 1 - old_color
        mono_after = all_same & (other_colors[:, 0] == new_color)
        
        return int(mono_after.sum()) - int(mono_before.sum())

    def local_search(colors, max_passes=3):
        """Perform hill-climbing local search. Returns (colors, count)."""
        count = count_mono(colors)
        for _ in range(max_passes):
            if time.time() > deadline - 3:
                break
            improved = False
            for p in range(N):
                if time.time() > deadline - 3:
                    break
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    count += delta
                    improved = True
            if not improved:
                break
        return colors, count

    def uniform_crossover(parent1, parent2):
        """Uniform crossover: each bit independently chosen from either parent with 50% probability."""
        mask = np.random.randint(0, 2, size=N).astype(np.int8)
        child = parent1 * mask + parent2 * (1 - mask)
        return child

    def mutate(colors, mutation_rate):
        """Flip each bit independently with probability mutation_rate."""
        flip_mask = (np.random.rand(N) < mutation_rate).astype(np.int8)
        colors[flip_mask] ^= 1
        return colors

    best_colors = None
    best_count = float('inf')
    
    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()

    # Initialize population with diverse starting points
    pop_size = 30
    population = []
    
    # Diverse initial individuals
    init_patterns = [
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8),      # Alternating pairs
        np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 3
        np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 4
        np.array([(i // 5) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 5
        np.array([(i // 6) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 6
        np.array([(i // 7) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 7
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 1
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 2
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 3
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 4
    ]
    
    for i in range(pop_size):
        if i < len(init_patterns):
            colors = init_patterns[i].copy()
        else:
            # More random individuals
            colors = np.random.randint(0, 2, size=N).astype(np.int8)
        
        # Apply local search to each initial individual
        colors, cnt = local_search(colors, max_passes=2)
        population.append((colors, cnt))
        update_best(colors, cnt)
    
    # Sort population by fitness (count)
    population.sort(key=lambda x: x[1])
    
    # Genetic Algorithm with Simulated Annealing hybrid
    elite_size = 5
    mutation_rate = 0.05
    sa_temperature = 2.0
    sa_cooling = 0.9995
    sa_min_temp = 0.05
    
    generation = 0
    max_generations = 500
    
    while generation < max_generations:
        if time.time() > deadline - 10:
            break
        
        generation += 1
        
        # Create next generation
        next_population = []
        
        # Elitism: keep the best elite_size individuals
        for i in range(min(elite_size, len(population))):
            next_population.append(population[i])
        
        # Fill the rest with offspring
        while len(next_population) < pop_size:
            if time.time() > deadline - 10:
                break
            
            # Tournament selection for parents
            def tournament_select(k=3):
                candidates = random.sample(range(len(population)), k)
                best_idx = min(candidates, key=lambda i: population[i][1])
                return population[best_idx][0]
            
            parent1 = tournament_select(3)
            parent2 = tournament_select(3)
            
            # Uniform crossover
            child = uniform_crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate)
            
            # Local search on child
            child, child_count = local_search(child, max_passes=2)
            
            next_population.append((child, child_count))
            update_best(child, child_count)
        
        # Replace population
        population = next_population
        population.sort(key=lambda x: x[1])
        
        # Periodically inject diversity to prevent premature convergence
        if generation % 20 == 0 and time.time() < deadline - 15:
            # Replace worst individuals with fresh random ones
            for i in range(len(population) - 1, max(len(population) - 5, -1), -1):
                if time.time() > deadline - 15:
                    break
                new_individual = np.random.randint(0, 2, size=N).astype(np.int8)
                new_individual, new_count = local_search(new_individual, max_passes=2)
                population[i] = (new_individual, new_count)
                update_best(new_individual, new_count)
            population.sort(key=lambda x: x[1])
        
        # Apply SA to the best individual periodically
        if generation % 10 == 0 and time.time() < deadline - 15:
            best_ind = population[0][0].copy()
            best_cnt = population[0][1]
            
            T = sa_temperature
            for step in range(2000):
                if time.time() > deadline - 15:
                    break
                
                p = np.random.randint(0, N)
                delta = fast_delta(best_ind, p)
                
                if delta <= 0:
                    best_ind[p] ^= 1
                    best_cnt += delta
                else:
                    if T > sa_min_temp and random.random() < np.exp(-delta / T):
                        best_ind[p] ^= 1
                        best_cnt += delta
                
                T = max(sa_min_temp, T * sa_cooling)
            
            # Local search on SA result
            best_ind, best_cnt = local_search(best_ind, max_passes=2)
            update_best(best_ind, best_cnt)
            
            # Insert into population if improved
            if best_cnt < population[-1][1]:
                population[-1] = (best_ind, best_cnt)
                population.sort(key=lambda x: x[1])
    
    # Final local search on the best solution
    if best_colors is not None:
        final_colors, final_count = local_search(best_colors.copy(), max_passes=5)
        update_best(final_colors, final_count)
    
    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]

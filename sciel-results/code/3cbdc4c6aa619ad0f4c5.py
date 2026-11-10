import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (0-based indices)
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # For each position, which APs contain it
    pos_to_aps = [[] for _ in range(N)]
    for i in range(total_aps):
        for j in range(4):
            pos_to_aps[aps[i, j]].append(i)
    
    # Precompute for each position: the AP indices
    pos_ap_indices = []
    for p in range(N):
        pos_ap_indices.append(np.array(pos_to_aps[p], dtype=np.int32))
    
    def count_mono(colors):
        """Count monochromatic 4-APs using numpy."""
        c = colors[aps]  # shape (total_aps, 4)
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())
    
    def fast_compute_delta(colors, p):
        """Faster delta computation using precomputed structure."""
        old_color = colors[p]
        ap_idxs = pos_ap_indices[p]
        n_aps = len(ap_idxs)
        if n_aps == 0:
            return 0
        
        sub_aps = aps[ap_idxs]  # shape (n_aps, 4)
        sub_colors = colors[sub_aps]  # shape (n_aps, 4)
        
        # Before: check if all 4 are same
        mono_before = (sub_colors.max(axis=1) == sub_colors.min(axis=1))
        
        # After: flip position p in each AP
        # Find which column p occupies in each AP
        positions = np.array([list(sub_aps[i]).index(p) for i in range(n_aps)], dtype=np.int8)
        
        sub_colors_after = sub_colors.copy()
        rows = np.arange(n_aps)
        sub_colors_after[rows, positions] = 1 - old_color
        
        mono_after = (sub_colors_after.max(axis=1) == sub_colors_after.min(axis=1))
        
        return int(mono_after.sum()) - int(mono_before.sum())
    
    def local_search(colors, max_time=None):
        """Hill climbing: flip bits that reduce mono count."""
        current_count = count_mono(colors)
        improved = True
        while improved:
            if max_time and time.time() > max_time:
                break
            improved = False
            for p in range(N):
                if max_time and time.time() > max_time:
                    break
                delta = fast_compute_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
        return current_count
    
    # Genetic Algorithm
    pop_size = 40
    elite_count = 4
    mutation_rate = 0.05
    crossover_prob = 0.8
    
    # Initialize population with diverse starting points
    population = []
    
    # Seed 1: alternating pairs
    population.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8))
    # Seed 2: blocks of 3
    population.append(np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8))
    # Seed 3: blocks of 4
    population.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8))
    # Seed 4: random
    population.append(np.random.randint(0, 2, size=N).astype(np.int8))
    # Seed 5: Thue-Morse-like pattern
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        tm[i] = bin(i).count('1') % 2
    population.append(tm.copy())
    
    # Fill rest with random perturbations of existing seeds
    while len(population) < pop_size:
        base = random.choice(population[:5])
        child = base.copy()
        n_flips = random.randint(10, 50)
        flip_pos = np.random.choice(N, size=n_flips, replace=False)
        child[flip_pos] ^= 1
        population.append(child)
    
    # Evaluate initial population
    scores = [count_mono(ind) for ind in population]
    
    best_count = min(scores)
    best_colors = population[scores.index(best_count)].copy()
    
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Update best
        if scores[0] < best_count:
            best_count = scores[0]
            best_colors = population[0].copy()
        
        # Create new generation
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(min(elite_count, pop_size)):
            new_population.append(population[i].copy())
        
        # Fill rest with children
        while len(new_population) < pop_size:
            # Selection: tournament selection
            def tournament_select():
                candidates = random.sample(range(pop_size), 3)
                best_idx = min(candidates, key=lambda i: scores[i])
                return population[best_idx].copy()
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if random.random() < crossover_prob:
                # Uniform crossover
                mask = np.random.randint(0, 2, size=N)
                child = np.where(mask == 1, parent1, parent2).astype(np.int8)
            else:
                child = parent1.copy()
            
            # Mutation
            n_mutations = int(N * mutation_rate)
            if n_mutations > 0:
                mut_pos = np.random.choice(N, size=n_mutations, replace=False)
                child[mut_pos] ^= 1
            
            # Local search on child (with time limit)
            remaining_time = deadline - 10 - time.time()
            if remaining_time > 2:
                local_search(child, max_time=time.time() + min(remaining_time, 2.0))
            
            new_population.append(child)
        
        population = new_population
        scores = [count_mono(ind) for ind in population]
        
        # Periodically inject the best known solution with perturbation
        if generation % 5 == 0 and best_colors is not None:
            perturbed = best_colors.copy()
            n_flips = random.randint(5, 20)
            flip_pos = np.random.choice(N, size=n_flips, replace=False)
            perturbed[flip_pos] ^= 1
            # Replace worst individual
            worst_idx = np.argmax(scores)
            population[worst_idx] = perturbed
            scores[worst_idx] = count_mono(perturbed)
    
    return [int(c) for c in best_colors]

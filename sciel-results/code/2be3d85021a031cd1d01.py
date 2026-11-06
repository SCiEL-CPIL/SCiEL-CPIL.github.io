import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # For each position, list of (ap_index, offset)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    def compute_score(color):
        """Compute mono count for a coloring."""
        count = 0
        for p0, p1, p2, p3 in aps:
            if color[p0] == color[p1] == color[p2] == color[p3]:
                count += 1
        return count
    
    def delta_flip(color, p):
        """Compute delta in mono count when flipping position p."""
        delta = 0
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            # Check if currently mono
            is_mono = (color[p0] == color[p1] == color[p2] == color[p3])
            if is_mono:
                delta -= 1
            else:
                new_val = 1 - color[p]
                others = []
                for j, pos in enumerate((p0, p1, p2, p3)):
                    if j != offset:
                        others.append(color[pos])
                if all(v == new_val for v in others):
                    delta += 1
        return delta
    
    def local_search(color, max_sweeps=50):
        """Greedy local search with incremental updates."""
        current = compute_score(color)
        
        for sweep in range(max_sweeps):
            # Find all positions in mono APs
            mono_positions = set()
            for p0, p1, p2, p3 in aps:
                if color[p0] == color[p1] == color[p2] == color[p3]:
                    mono_positions.update([p0, p1, p2, p3])
            
            if not mono_positions:
                break
            
            best_delta = 0
            best_pos = -1
            for p in mono_positions:
                d = delta_flip(color, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            
            if best_pos < 0:
                break
            
            color[best_pos] = 1 - color[best_pos]
            current += best_delta
            
            if time.time() > deadline - 5:
                break
        
        return color, current
    
    # Initialize population with diverse starting points
    random.seed(42)
    population = []
    
    # Thue-Morse
    tm = [bin(i).count('1') % 2 for i in range(n)]
    tm, _ = local_search(tm, max_sweeps=30)
    population.append(tm[:])
    
    # Random initializations
    for _ in range(7):
        rand_color = [random.randint(0, 1) for _ in range(n)]
        rand_color, _ = local_search(rand_color, max_sweeps=30)
        population.append(rand_color[:])
    
    # Block patterns
    for block_size in [2, 4, 8, 16]:
        block_color = [(i // block_size) % 2 for i in range(n)]
        block_color, _ = local_search(block_color, max_sweeps=30)
        population.append(block_color[:])
    
    # Evaluate initial population
    scores = [compute_score(ind) for ind in population]
    
    best_color = population[scores.index(min(scores))][:]
    best_score = min(scores)
    
    # Genetic algorithm
    pop_size = len(population)
    elite_count = 2
    tournament_size = 3
    crossover_rate = 0.8
    mutation_rate = 0.15
    max_generations = 50
    
    for gen in range(max_generations):
        if time.time() > deadline - 15:
            break
        
        # Sort by fitness
        indexed_pop = sorted(enumerate(population), key=lambda x: scores[x[0]])
        population = [p for _, p in indexed_pop]
        scores = [scores[i] for i, _ in indexed_pop]
        
        # Keep elites
        new_population = [population[i][:] for i in range(elite_count)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
            
            # Tournament selection
            parents = []
            for _ in range(2):
                candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
                best_idx = min(candidates, key=lambda i: scores[i])
                parents.append(population[best_idx])
            
            # Crossover (uniform)
            if random.random() < crossover_rate:
                child = [random.choice([parents[0][i], parents[1][i]]) for i in range(n)]
            else:
                child = parents[0][:]
            
            # Mutation
            for i in range(n):
                if random.random() < mutation_rate / n * 10:  # ~10 mutations per genome
                    child[i] = 1 - child[i]
            
            # Local search on child
            child, _ = local_search(child, max_sweeps=20)
            new_population.append(child[:])
        
        population = new_population
        scores = [compute_score(ind) for ind in population]
        
        # Update best
        gen_best_idx = scores.index(min(scores))
        if scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_color = population[gen_best_idx][:]
    
    # Final local search on best
    best_color, best_score = local_search(best_color, max_sweeps=100)
    
    return best_color

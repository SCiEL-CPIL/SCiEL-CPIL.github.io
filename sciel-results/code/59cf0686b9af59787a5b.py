import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin before 120s kill
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            
            ap = (idx1, idx2, idx3, idx4)
            ap_id = len(aps)
            aps.append(ap)
            
            index_to_aps[idx1].append((ap_id, 0))
            index_to_aps[idx2].append((ap_id, 1))
            index_to_aps[idx3].append((ap_id, 2))
            index_to_aps[idx4].append((ap_id, 3))
            
    total_aps = len(aps)
    
    # Helper to count monochromatic APs from scratch
    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    # Fast delta computation for single bit flip
    def compute_delta(colors, k):
        delta = 0
        c_k = colors[k]
        for ap_id, pos in index_to_aps[k]:
            ap = aps[ap_id]
            if pos == 0:
                o1, o2, o3 = ap[1], ap[2], ap[3]
            elif pos == 1:
                o1, o2, o3 = ap[0], ap[2], ap[3]
            elif pos == 2:
                o1, o2, o3 = ap[0], ap[1], ap[3]
            else:
                o1, o2, o3 = ap[0], ap[1], ap[2]
            if colors[o1] == colors[o2] == colors[o3]:
                if c_k == colors[o1]:
                    delta -= 1
                else:
                    delta += 1
        return delta
    
    # Local search refinement (greedy + limited SA)
    def local_search(colors, max_time=2.0):
        colors = colors[:]
        current_score = count_mono(colors)
        best_score = current_score
        best_colors = colors[:]
        
        # Phase 1: Greedy single-bit flips
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for k in range(N):
                delta = compute_delta(colors, k)
                if delta < 0:
                    colors[k] = 1 - colors[k]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
        
        # Phase 2: Short SA to escape local minima
        T = 0.5
        cooling = 0.999
        min_T = 0.01
        sa_start = time.time()
        while T > min_T and time.time() < sa_start + max_time:
            k = random.randint(0, N - 1)
            delta = compute_delta(colors, k)
            if delta < 0:
                colors[k] = 1 - colors[k]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            else:
                if random.random() < math.exp(-delta / T):
                    colors[k] = 1 - colors[k]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
            T *= cooling
        
        return best_colors, best_score
    
    # Genetic Algorithm components
    
    # Uniform crossover
    def crossover(parent1, parent2):
        child = [random.choice((p1, p2)) for p1, p2 in zip(parent1, parent2)]
        return child
    
    # Mutation: flip a small number of random bits
    def mutate(colors, mutation_rate=0.02):
        child = colors[:]
        for i in range(N):
            if random.random() < mutation_rate:
                child[i] = 1 - child[i]
        return child
    
    # Tournament selection
    def tournament_select(population, scores, k=3):
        indices = random.sample(range(len(population)), k)
        best_idx = min(indices, key=lambda i: scores[i])
        return population[best_idx][:]
    
    # Initialize population with diverse starting points
    population = []
    scores = []
    
    # Diverse initializations
    init_patterns = [
        [(i // 2) % 2 for i in range(N)],
        [bin(i).count('1') % 2 for i in range(N)],
        [i % 2 for i in range(N)],
        [(i // 4) % 2 for i in range(N)],
        [((i // 3) + (i % 3)) % 2 for i in range(N)],
        [random.randint(0, 1) for _ in range(N)],
        [random.randint(0, 1) for _ in range(N)],
        [random.randint(0, 1) for _ in range(N)],
    ]
    
    for pattern in init_patterns:
        score = count_mono(pattern)
        population.append(pattern)
        scores.append(score)
    
    # Refine initial population
    refined_pop = []
    refined_scores = []
    for i in range(len(population)):
        if time.time() > deadline - 3.0:
            refined_pop.append(population[i])
            refined_scores.append(scores[i])
            continue
        refined, ref_score = local_search(population[i], max_time=1.0)
        refined_pop.append(refined)
        refined_scores.append(ref_score)
    
    population = refined_pop
    scores = refined_scores
    
    # Track global best
    global_best_idx = min(range(len(scores)), key=lambda i: scores[i])
    global_best_score = scores[global_best_idx]
    global_best_colors = population[global_best_idx][:]
    
    # GA parameters
    pop_size = len(population)
    elite_size = max(2, pop_size // 4)
    generations = 0
    
    while time.time() < deadline - 5.0:
        generations += 1
        
        # Sort by fitness
        order = sorted(range(pop_size), key=lambda i: scores[i])
        
        # Elitism: keep top individuals
        new_population = []
        new_scores = []
        for i in range(elite_size):
            idx = order[i]
            new_population.append(population[idx][:])
            new_scores.append(scores[idx])
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline - 5.0:
            # Tournament selection for parents
            parent1 = tournament_select(population, scores, k=3)
            parent2 = tournament_select(population, scores, k=3)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate=0.03)
            
            # Evaluate
            child_score = count_mono(child)
            new_population.append(child)
            new_scores.append(child_score)
        
        # Update global best
        for i in range(pop_size):
            if new_scores[i] < global_best_score:
                global_best_score = new_scores[i]
                global_best_colors = new_population[i][:]
        
        # Apply local search to some offspring (not all, to save time)
        if time.time() < deadline - 3.0:
            # Refine the best few new individuals
            new_order = sorted(range(pop_size), key=lambda i: new_scores[i])
            num_to_refine = min(3, pop_size)
            for i in range(num_to_refine):
                idx = new_order[i]
                if time.time() > deadline - 2.0:
                    break
                refined, ref_score = local_search(new_population[idx], max_time=0.5)
                new_population[idx] = refined
                new_scores[idx] = ref_score
                if ref_score < global_best_score:
                    global_best_score = ref_score
                    global_best_colors = refined[:]
        
        population = new_population
        scores = new_scores
    
    return global_best_colors

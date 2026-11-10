import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Store as 0-indexed tuples
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(aps)

    # For each index, precompute the list of AP indices it belongs to
    # and for each such AP, which other 3 indices are in it
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, (i1, i2, i3, i4) in enumerate(aps):
        for idx in (i1, i2, i3, i4):
            others = [x for x in (i1, i2, i3, i4) if x != idx]
            index_to_aps[idx].append((ap_idx, others))

    def count_mono(color):
        """Count monochromatic 4-APs."""
        count = 0
        for i1, i2, i3, i4 in aps:
            c = color[i1]
            if c == color[i2] == color[i3] == color[i4]:
                count += 1
        return count

    def get_delta(color, idx, new_val):
        """Compute change in mono count if we flip color[idx] to new_val."""
        old_val = color[idx]
        if old_val == new_val:
            return 0
        delta = 0
        for ap_idx, others in index_to_aps[idx]:
            o0, o1, o2 = others
            if color[o0] == color[o1] == color[o2]:
                c_others = color[o0]
                if old_val == c_others:
                    # Currently mono. If new_val != c_others, it becomes non-mono.
                    if new_val != c_others:
                        delta -= 1
                else:
                    # Currently non-mono (idx is different from others).
                    # If new_val == c_others, it becomes mono.
                    if new_val == c_others:
                        delta += 1
        return delta

    def local_search(color, max_iter=200):
        """Hill climbing: flip bits if it improves the score."""
        current_score = count_mono(color)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for idx in range(N):
                new_val = 1 - color[idx]
                delta = get_delta(color, idx, new_val)
                if delta < 0:
                    color[idx] = new_val
                    current_score += delta
                    improved = True
            iter_count += 1
        return current_score

    def random_init():
        return np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)

    def block_init(block_size):
        return np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)

    def stripe_init(period):
        return np.array([(i % period) < period // 2 for i in range(N)]).astype(np.int8)

    def xor_init(p1, p2):
        """XOR of two periodic patterns."""
        a = (np.arange(N) % p1) < p1 // 2
        b = (np.arange(N) % p2) < p2 // 2
        return (a ^ b).astype(np.int8)

    # Generate diverse initial configurations
    candidates = []
    
    # Block patterns
    for bs in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]:
        candidates.append(block_init(bs))
    
    # Stripe patterns
    for p in [3, 5, 7, 9, 11, 13, 17]:
        candidates.append(stripe_init(p))
    
    # XOR patterns
    for p1 in [3, 4, 5, 7]:
        for p2 in [5, 7, 9, 11]:
            if p1 != p2:
                candidates.append(xor_init(p1, p2))
    
    # Random inits
    for _ in range(30):
        candidates.append(random_init())
    
    # Evaluate and refine candidates
    scored = []
    for c in candidates:
        c = c.copy()
        sc = local_search(c, max_iter=50)
        scored.append((sc, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Genetic Algorithm + SA Hybrid
    rng = random.Random(42)
    
    # Initialize population from top candidates
    pop_size = 24
    population = []
    for i in range(min(pop_size, len(scored))):
        population.append(scored[i][1].copy())
    # Fill remaining with random
    while len(population) < pop_size:
        population.append(random_init())
    
    # Track best overall
    best_score = scored[0][0]
    best_color = scored[0][1].copy()
    
    # GA parameters
    mutation_rate = 0.03
    elite_count = 4
    
    while time.time() < deadline - 15.0:
        # Evaluate population
        scored_pop = []
        for ind in population:
            sc = count_mono(ind)
            scored_pop.append((sc, ind))
        
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_color = scored_pop[0][1].copy()
        
        # Keep elites
        new_population = [ind.copy() for _, ind in scored_pop[:elite_count]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10.0:
                break
            
            # Tournament selection
            def tournament_select(pop, scores, k=3):
                cands = rng.sample(range(len(pop)), min(k, len(pop)))
                best_idx = min(cands, key=lambda i: scores[i])
                return pop[best_idx].copy()
            
            scores_list = [s for s, _ in scored_pop]
            parent1 = tournament_select(population, scores_list)
            parent2 = tournament_select(population, scores_list)
            
            # Uniform crossover
            child = np.array([
                parent1[i] if rng.random() < 0.5 else parent2[i] 
                for i in range(N)
            ], dtype=np.int8)
            
            # Mutation
            for i in range(N):
                if rng.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Quick local search on child
            local_search(child, max_iter=30)
            
            new_population.append(child)
        
        population = new_population
        
        # Apply SA to the best individual in population
        if time.time() < deadline - 15.0:
            best_in_pop = population[0].copy()
            current_score = count_mono(best_in_pop)
            
            # SA parameters
            T = 3.0
            T_min = 0.01
            cooling = 0.9995
            max_iter = 20000
            
            for it in range(max_iter):
                if time.time() > deadline - 10.0:
                    break
                
                # Choose move: 1-bit flip (70%), 2-bit flip (20%), 3-bit flip (10%)
                r = rng.random()
                if r < 0.7:
                    # 1-bit flip
                    idx = rng.randint(0, N - 1)
                    new_val = 1 - best_in_pop[idx]
                    delta = get_delta(best_in_pop, idx, new_val)
                    
                    if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
                        best_in_pop[idx] = new_val
                        current_score += delta
                        if current_score < best_score:
                            best_score = current_score
                            best_color = best_in_pop.copy()
                elif r < 0.9:
                    # 2-bit flip
                    idxs = rng.sample(range(N), 2)
                    idx1, idx2 = idxs
                    v1 = 1 - best_in_pop[idx1]
                    v2 = 1 - best_in_pop[idx2]
                    
                    delta1 = get_delta(best_in_pop, idx1, v1)
                    best_in_pop[idx1] = v1
                    delta2 = get_delta(best_in_pop, idx2, v2)
                    total_delta = delta1 + delta2
                    
                    if total_delta < 0 or (total_delta > 0 and rng.random() < np.exp(-total_delta / T)):
                        best_in_pop[idx2] = v2
                        current_score += total_delta
                        if current_score < best_score:
                            best_score = current_score
                            best_color = best_in_pop.copy()
                    else:
                        best_in_pop[idx1] = 1 - v1
                else:
                    # 3-bit flip
                    idxs = rng.sample(range(N), 3)
                    idx1, idx2, idx3 = idxs
                    v1 = 1 - best_in_pop[idx1]
                    v2 = 1 - best_in_pop[idx2]
                    v3 = 1 - best_in_pop[idx3]
                    
                    delta1 = get_delta(best_in_pop, idx1, v1)
                    best_in_pop[idx1] = v1
                    delta2 = get_delta(best_in_pop, idx2, v2)
                    best_in_pop[idx2] = v2
                    delta3 = get_delta(best_in_pop, idx3, v3)
                    total_delta = delta1 + delta2 + delta3
                    
                    if total_delta < 0 or (total_delta > 0 and rng.random() < np.exp(-total_delta / T)):
                        best_in_pop[idx3] = v3
                        current_score += total_delta
                        if current_score < best_score:
                            best_score = current_score
                            best_color = best_in_pop.copy()
                    else:
                        best_in_pop[idx1] = 1 - v1
                        best_in_pop[idx2] = 1 - v2
            
            # Replace worst in population with SA result
            population[-1] = best_in_pop
    
    # Final high-precision polish pass
    if time.time() < deadline - 2.0:
        # Multiple rounds of local search with increasing iterations
        for iters in [50, 100, 200, 500]:
            if time.time() > deadline - 2.0:
                break
            local_search(best_color, max_iter=iters)
            new_sc = count_mono(best_color)
            if new_sc < best_score:
                best_score = new_score
    
    return [int(x) for x in best_color]

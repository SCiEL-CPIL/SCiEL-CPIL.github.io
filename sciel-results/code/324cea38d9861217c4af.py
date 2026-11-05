import numpy as np
import random
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 110
    
    # Target n=34
    target_n = 34
    
    # Try to find a valid coloring for n=34 using the F_3 pair-based approach
    n = target_n
    
    # Number of pairs
    num_pairs = n * (n - 1) // 2
    
    # Precompute pair indices
    pair_indices = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices[(i, j)] = idx
            idx += 1
    
    # Precompute triples and their constituent pairs
    triples = list(itertools.combinations(range(n), 3))
    num_triples = len(triples)
    
    # For each triple, store the 3 pair indices
    triple_pairs = []
    for (a, b, c) in triples:
        p1 = pair_indices[(a, b)]
        p2 = pair_indices[(b, c)]
        p3 = pair_indices[(a, c)]
        triple_pairs.append((p1, p2, p3))
    
    # Precompute 4-sets and their triples
    four_sets = list(itertools.combinations(range(n), 4))
    four_set_triple_indices = []
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    for s in four_sets:
        ts = []
        for t in itertools.combinations(s, 3):
            ts.append(triple_to_idx[t])
        four_set_triple_indices.append(ts)
    
    # Precompute 5-sets and their triples
    five_sets = list(itertools.combinations(range(n), 5))
    five_set_triple_indices = []
    for s in five_sets:
        ts = []
        for t in itertools.combinations(s, 3):
            ts.append(triple_to_idx[t])
        five_set_triple_indices.append(ts)
    
    # Mapping from F_3 value to color: 0->red(0), 1->blue(1), 2->blue(1)
    mappings = [
        [0, 1, 1],  # 0->red, 1->blue, 2->blue
        [0, 0, 1],  # 0->red, 1->red, 2->blue
        [1, 0, 0],  # 0->blue, 1->red, 2->red
        [1, 1, 0],  # 0->blue, 1->blue, 2->red
    ]
    
    def compute_colors(pair_values, mapping):
        """Compute triple colors from pair values using given mapping."""
        colors = np.zeros(num_triples, dtype=np.int8)
        for i, (p1, p2, p3) in enumerate(triple_pairs):
            val = (pair_values[p1] + pair_values[p2] + pair_values[p3]) % 3
            colors[i] = mapping[val]
        return colors
    
    def compute_penalty(colors):
        """Compute number of violations."""
        penalty = 0
        for ts in four_set_triple_indices:
            if all(colors[t] == 0 for t in ts):
                penalty += 1
        for ts in five_set_triple_indices:
            if all(colors[t] == 1 for t in ts):
                penalty += 1
        return penalty
    
    best_penalty = float('inf')
    best_pair_values = None
    best_mapping = None
    
    # Genetic algorithm
    pop_size = 100
    generations = 500
    mutation_rate = 0.05
    crossover_rate = 0.8
    
    for mapping in mappings:
        if time.time() - start_time > time_limit:
            break
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            pv = np.random.randint(0, 3, size=num_pairs)
            population.append(pv)
        
        # Evaluate initial population
        best_in_pop = None
        best_pen_in_pop = float('inf')
        for pv in population:
            colors = compute_colors(pv, mapping)
            pen = compute_penalty(colors)
            if pen < best_pen_in_pop:
                best_pen_in_pop = pen
                best_in_pop = pv.copy()
        
        if best_pen_in_pop < best_penalty:
            best_penalty = best_pen_in_pop
            best_pair_values = best_in_pop.copy()
            best_mapping = mapping
        
        if best_penalty == 0:
            break
        
        # Evolution
        for gen in range(generations):
            if time.time() - start_time > time_limit:
                break
            
            # Evaluate population
            scored = []
            for pv in population:
                colors = compute_colors(pv, mapping)
                pen = compute_penalty(colors)
                scored.append((pen, pv))
            
            scored.sort(key=lambda x: x[0])
            
            # Check if we found a solution
            if scored[0][0] == 0:
                best_penalty = 0
                best_pair_values = scored[0][1].copy()
                best_mapping = mapping
                break
            
            if scored[0][0] < best_penalty:
                best_penalty = scored[0][0]
                best_pair_values = scored[0][1].copy()
                best_mapping = mapping
            
            # Create next generation
            new_population = []
            
            # Elitism: keep top 10%
            elite_count = max(1, pop_size // 10)
            for i in range(elite_count):
                new_population.append(scored[i][1].copy())
            
            # Fill rest with offspring
            while len(new_population) < pop_size:
                # Tournament selection
                def tournament_select():
                    candidates = random.sample(scored, 3)
                    return min(candidates, key=lambda x: x[0])[1]
                
                parent1 = tournament_select()
                parent2 = tournament_select()
                
                if random.random() < crossover_rate:
                    # Uniform crossover
                    child = np.where(np.random.random(num_pairs) < 0.5, parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                for i in range(num_pairs):
                    if random.random() < mutation_rate:
                        child[i] = random.randint(0, 2)
                
                new_population.append(child)
            
            population = new_population
    
    # If we found a solution with the F_3 approach
    if best_penalty == 0 and best_pair_values is not None:
        colors = compute_colors(best_pair_values, best_mapping)
        return {'n': n, 'colors': colors.tolist()}
    
    # Fallback: try smaller n with the same approach
    for n in range(33, 8, -1):
        if time.time() - start_time > time_limit:
            break
        
        num_pairs = n * (n - 1) // 2
        pair_indices = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_indices[(i, j)] = idx
                idx += 1
        
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_pairs = []
        for (a, b, c) in triples:
            p1 = pair_indices[(a, b)]
            p2 = pair_indices[(b, c)]
            p3 = pair_indices[(a, c)]
            triple_pairs.append((p1, p2, p3))
        
        four_sets = list(itertools.combinations(range(n), 4))
        four_set_triple_indices = []
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        for s in four_sets:
            ts = []
            for t in itertools.combinations(s, 3):
                ts.append(triple_to_idx[t])
            four_set_triple_indices.append(ts)
        
        five_sets = list(itertools.combinations(range(n), 5))
        five_set_triple_indices = []
        for s in five_sets:
            ts = []
            for t in itertools.combinations(s, 3):
                ts.append(triple_to_idx[t])
            five_set_triple_indices.append(ts)
        
        for mapping in mappings:
            if time.time() - start_time > time_limit:
                break
            
            # Quick random search
            for attempt in range(200):
                if time.time() - start_time > time_limit:
                    break
                pv = np.random.randint(0, 3, size=num_pairs)
                colors = np.zeros(num_triples, dtype=np.int8)
                for i, (p1, p2, p3) in enumerate(triple_pairs):
                    val = (pv[p1] + pv[p2] + pv[p3]) % 3
                    colors[i] = mapping[val]
                
                pen = 0
                for ts in four_set_triple_indices:
                    if all(colors[t] == 0 for t in ts):
                        pen += 1
                for ts in five_set_triple_indices:
                    if all(colors[t] == 1 for t in ts):
                        pen += 1
                
                if pen == 0:
                    return {'n': n, 'colors': colors.tolist()}
    
    # Final fallback: return a small valid coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

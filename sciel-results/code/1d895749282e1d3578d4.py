"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols using the standard
    recursive construction, with a genetic algorithm optimization for n=7
    to reorder the permutations of 1..6 to minimize the total length.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        
        # Optimization for m=7 (adding 7 to perms of 1..6)
        if m == 7:
            # We have 720 permutations.
            # Use genetic algorithm to find optimal ordering.
            
            num_perms = len(perms)
            
            # Precompute overlap matrix
            def get_overlap(p1, p2):
                # p1, p2 are tuples
                for k in range(6, 0, -1):
                    if p1[-k:] == p2[:k]:
                        return k
                return 0
            
            # Overlap from start (end of S_6) to each perm
            start_end = tuple(word[-6:])
            start_ov = np.zeros(num_perms, dtype=np.int8)
            for i in range(num_perms):
                start_ov[i] = get_overlap(start_end, perms[i])
            
            # Pairwise overlaps
            ov_matrix = np.zeros((num_perms, num_perms), dtype=np.int8)
            for i in range(num_perms):
                for j in range(num_perms):
                    if i != j:
                        ov_matrix[i][j] = get_overlap(perms[i], perms[j])
            
            # Fitness function: maximize total overlap
            def fitness(order):
                """Calculate total overlap for a given order."""
                total = start_ov[order[0]]
                for i in range(len(order) - 1):
                    total += ov_matrix[order[i]][order[i+1]]
                return total
            
            # Genetic algorithm
            pop_size = 50
            elite_size = 5
            mutation_rate = 0.1
            crossover_rate = 0.8
            max_generations = 100
            
            # Initialize population
            base_order = list(range(num_perms))
            population = [base_order[:]]
            for _ in range(pop_size - 1):
                ind = base_order[:]
                random.shuffle(ind)
                population.append(ind)
            
            # Evaluate initial population
            fitnesses = [fitness(ind) for ind in population]
            best_idx = fitnesses.index(max(fitnesses))
            best_order = population[best_idx][:]
            best_fitness = fitnesses[best_idx]
            
            # Run genetic algorithm
            for gen in range(max_generations):
                if time.time() > deadline - 10:
                    break
                
                # Sort by fitness
                sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
                
                # Keep elites
                new_population = [population[i][:] for i in sorted_indices[:elite_size]]
                
                # Generate rest of population
                while len(new_population) < pop_size:
                    if time.time() > deadline - 10:
                        break
                    
                    # Tournament selection
                    def tournament_select():
                        candidates = random.sample(range(pop_size), 3)
                        best_cand = max(candidates, key=lambda i: fitnesses[i])
                        return population[best_cand][:]
                    
                    parent1 = tournament_select()
                    parent2 = tournament_select()
                    
                    # Crossover
                    if random.random() < crossover_rate:
                        # Order crossover
                        child = [None] * num_perms
                        start = random.randint(0, num_perms - 2)
                        end = random.randint(start + 1, num_perms - 1)
                        
                        # Copy segment from parent1
                        for i in range(start, end + 1):
                            child[i] = parent1[i]
                        
                        # Fill rest from parent2
                        remaining = [x for x in parent2 if x not in set(parent1[start:end+1])]
                        pos = 0
                        for i in range(num_perms):
                            if child[i] is None:
                                child[i] = remaining[pos]
                                pos += 1
                    else:
                        child = parent1[:]
                    
                    # Mutation: swap two random positions
                    if random.random() < mutation_rate:
                        i, j = random.sample(range(num_perms), 2)
                        child[i], child[j] = child[j], child[i]
                    
                    new_population.append(child)
                
                # Evaluate new population
                population = new_population
                fitnesses = [fitness(ind) for ind in population]
                
                # Update best
                gen_best_idx = fitnesses.index(max(fitnesses))
                if fitnesses[gen_best_idx] > best_fitness:
                    best_fitness = fitnesses[gen_best_idx]
                    best_order = population[gen_best_idx][:]
            
            # Local search refinement on best order
            order = best_order[:]
            current_score = fitness(order)
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(order) - 1):
                    if time.time() > deadline - 5:
                        break
                    a = order[i-1] if i > 0 else -1
                    b = order[i]
                    c = order[i+1]
                    d = order[i+2] if i+2 < len(order) else -1
                    
                    # Old contribution
                    old_contrib = 0
                    if a != -1:
                        old_contrib += ov_matrix[a][b]
                    else:
                        old_contrib += start_ov[b]
                    old_contrib += ov_matrix[b][c]
                    if d != -1:
                        old_contrib += ov_matrix[c][d]
                    
                    # New contribution
                    new_contrib = 0
                    if a != -1:
                        new_contrib += ov_matrix[a][c]
                    else:
                        new_contrib += start_ov[c]
                    new_contrib += ov_matrix[c][b]
                    if d != -1:
                        new_contrib += ov_matrix[b][d]
                    
                    if new_contrib > old_contrib:
                        order[i], order[i+1] = order[i+1], order[i]
                        current_score += new_contrib - old_contrib
                        improved = True
            
            # Use the optimized order
            for idx in order:
                p = perms[idx]
                block = list(p) + [m] + list(p)
                
                # Find overlap
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        else:
            # Standard construction for m < 7
            for p in perms:
                block = list(p) + [m] + list(p)
                
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END

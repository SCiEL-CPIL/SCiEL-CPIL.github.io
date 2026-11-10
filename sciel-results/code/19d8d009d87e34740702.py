"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # We want to maximize n. Published construction is 34.
    # Strategy:
    # 1. Use a Genetic Algorithm with repair-based crossover and mutation.
    # 2. Try n values from 15 up to 34.
    # 3. For each n, run GA with a time budget.
    # 4. If a valid coloring (0 violations) is found, save it and try n+1.
    # 5. Return the best n found.
    
    best_n = 9
    best_colors = None
    
    # Helper to generate precomputed indices for fast NumPy evaluation
    def get_precomputed_indices(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 triple indices
        k4_triples = []
        for k4 in combinations(range(n), 4):
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
        k4_arr = np.array(k4_triples, dtype=np.int32)
        
        # Precompute K5 triple indices
        k5_triples = []
        for k5 in combinations(range(n), 5):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
        k5_arr = np.array(k5_triples, dtype=np.int32)
        
        return triples, num_triples, k4_arr, k5_arr

    def count_violations_vec(colors, k4_arr, k5_arr):
        """
        colors: 1D NumPy array of 0s and 1s.
        k4_arr: (num_k4, 4) array of triple indices.
        k5_arr: (num_k5, 10) array of triple indices.
        Returns total number of monochromatic K4 (red) and K5 (blue).
        """
        # Red K4: all 4 triples are 0
        red_k4_count = np.sum(colors[k4_arr], axis=1)
        red_violations = np.sum(red_k4_count == 0)
        
        # Blue K5: all 10 triples are 1
        blue_k5_count = np.sum(colors[k5_arr], axis=1)
        blue_violations = np.sum(blue_k5_count == 10)
        
        return int(red_violations + blue_violations)

    def find_violating_triples(colors, k4_arr, k5_arr):
        """Find all triple indices that are part of violating structures."""
        violating = set()
        
        # Red K4 violations: all 4 triples are 0
        red_k4_count = np.sum(colors[k4_arr], axis=1)
        red_viol_rows = np.where(red_k4_count == 0)[0]
        for row in red_viol_rows:
            for tri_idx in k4_arr[row]:
                violating.add(int(tri_idx))
        
        # Blue K5 violations: all 10 triples are 1
        blue_k5_count = np.sum(colors[k5_arr], axis=1)
        blue_viol_rows = np.where(blue_k5_count == 10)[0]
        for row in blue_viol_rows:
            for tri_idx in k5_arr[row]:
                violating.add(int(tri_idx))
        
        return violating

    def repair_coloring(colors, k4_arr, k5_arr, max_iter=1000):
        """
        Repair-based approach: iteratively flip triples that are part of violations
        to break monochromatic structures. This is a greedy local search.
        """
        for _ in range(max_iter):
            v = count_violations_vec(colors, k4_arr, k5_arr)
            if v == 0:
                break
            
            violating = find_violating_triples(colors, k4_arr, k5_arr)
            if not violating:
                break
            
            # Pick a random violating triple to flip
            tri_idx = random.choice(list(violating))
            colors[tri_idx] = 1 - colors[tri_idx]
        
        return colors

    def genetic_algorithm(n, k4_arr, k5_arr, time_budget):
        """
        Genetic Algorithm with repair-based crossover and mutation.
        """
        num_triples = len(k4_arr) * 4  # approximate, but we can get exact from k4_arr shape
        # Actually, let's compute num_triples properly
        num_triples = n * (n-1) * (n-2) // 6
        
        # Initialize population
        pop_size = 20
        population = []
        for _ in range(pop_size):
            c = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            population.append(c)
        
        # Evaluate initial population
        scores = [count_violations_vec(c, k4_arr, k5_arr) for c in population]
        
        # Track best
        best_idx = np.argmin(scores)
        best_colors = population[best_idx].copy()
        best_score = scores[best_idx]
        
        ga_start = time.time()
        generation = 0
        
        while time.time() < ga_start + time_budget:
            generation += 1
            
            # Selection: tournament selection
            def tournament_select():
                candidates = random.sample(range(pop_size), min(3, pop_size))
                best_cand = min(candidates, key=lambda i: scores[i])
                return population[best_cand].copy(), scores[best_cand]
            
            # Crossover: uniform crossover with repair
            new_population = []
            new_scores = []
            
            # Keep best individual (elitism)
            new_population.append(best_colors.copy())
            new_scores.append(best_score)
            
            while len(new_population) < pop_size:
                if time.time() > ga_start + time_budget:
                    break
                
                # Select parents
                parent1, score1 = tournament_select()
                parent2, score2 = tournament_select()
                
                # Uniform crossover
                mask = np.random.randint(0, 2, size=num_triples).astype(np.int8)
                child = np.where(mask == 1, parent1, parent2)
                
                # Apply repair to child
                child = repair_coloring(child, k4_arr, k5_arr, max_iter=50)
                
                # Mutation: flip a small number of random triples
                num_mutations = random.randint(1, max(1, num_triples // 50))
                for _ in range(num_mutations):
                    idx = random.randint(0, num_triples - 1)
                    child[idx] = 1 - child[idx]
                
                # Evaluate child
                child_score = count_violations_vec(child, k4_arr, k5_arr)
                new_population.append(child)
                new_scores.append(child_score)
            
            # Update population
            population = new_population
            scores = new_scores
            
            # Update best
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < best_score:
                best_score = scores[current_best_idx]
                best_colors = population[current_best_idx].copy()
            
            # Early termination
            if best_score == 0:
                break
            
            # Occasionally restart stuck individuals
            if generation % 50 == 0:
                for i in range(pop_size):
                    if scores[i] > best_score and random.random() < 0.1:
                        # Restart this individual
                        population[i] = np.random.randint(0, 2, size=num_triples).astype(np.int8)
                        scores[i] = count_violations_vec(population[i], k4_arr, k5_arr)
        
        return best_colors, best_score

    # Try increasing n
    n_start_val = 15
    n_end_val = 34
    
    current_valid_colors = None
    current_valid_n = 0
    
    for n in range(n_start_val, n_end_val + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Determine time budget for this n
        # Give more time to larger n as they are harder
        remaining_time = deadline - time.time()
        n_values_remaining = n_end_val - n + 1
        time_budget = min(remaining_time / max(1, n_values_remaining), 15.0)
        
        if time_budget < 1.0:
            break
        
        n_start_time = time.time()
        
        # Get precomputed indices
        triples, num_triples, k4_arr, k5_arr = get_precomputed_indices(n)
        
        # Run Genetic Algorithm
        colors, score = genetic_algorithm(n, k4_arr, k5_arr, time_budget)
        
        if score == 0:
            # Found valid coloring for n
            best_n = n
            best_colors = colors.tolist()
            current_valid_colors = colors
            current_valid_n = n
        else:
            # Did not find valid for n
            # If we fail for n, it's likely we fail for n+1
            # But let's give one more chance if we have time
            if n > n_start_val + 2:
                break
            # Otherwise, continue to next n
    
    # Fallback if we didn't find anything >= 15
    if best_n < 15:
        # Try a quick random search for small n?
        # Or just return the seed solution.
        # The seed solution n=9 is valid.
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    else:
        return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END

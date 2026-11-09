"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin from 120s limit
    
    def time_left():
        return deadline - time.time()

    def evaluate_colors(colors, n):
        """
        Counts violations.
        colors: list of 0/1 for triples in lexicographic order.
        Returns (red_violations, blue_violations)
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple to index
        # For speed in small n, we can just iterate. For n=34, C(34,3)=5984.
        # C(34,4)=46376, C(34,5)=278256.
        
        # Efficient counting using bitsets or just loops?
        # Let's use a dictionary for triple -> index for fast lookup if needed, 
        # but for counting violations, we need to check all K4s and K5s.
        
        # To make this fast, we precompute the list of triples for each K4 and K5?
        # That's expensive.
        
        # Alternative: Iterate through all K4s and K5s and check the colors.
        # For n=34:
        # 46k K4s * 4 checks = 186k ops.
        # 278k K5s * 10 checks = 2.7M ops.
        # This is ~3M operations. In Python, this might take ~0.1s - 0.5s.
        # If we do this for every individual in a GA, it's too slow.
        
        # We need a faster evaluation or a smarter search.
        
        # Let's stick to Local Search but optimize the evaluation.
        pass

    def solve_n_ga(n, time_limit):
        """
        Genetic Algorithm for finding a valid coloring.
        """
        if n < 4:
            return None

        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Precompute which triples constitute each K4 and K5
        # This is crucial for fast fitness evaluation in the GA
        k4_triples = []
        for k4 in k4s:
            k4_triples.append([triple_idx[tri] for tri in combinations(k4, 3)])
            
        k5_triples = []
        for k5 in k5s:
            k5_triples.append([triple_idx[tri] for tri in combinations(k5, 3)])

        def fitness(colors):
            """
            Returns 0 if valid, else number of violations.
            colors: numpy array of 0/1
            """
            red_v = 0
            for tri_indices in k4_triples:
                # Check if all 4 are red (0)
                if all(colors[i] == 0 for i in tri_indices):
                    red_v += 1
            if red_v > 0:
                return red_v # Early exit? No, we need total for GA pressure, but red is hard.
            
            blue_v = 0
            for tri_indices in k5_triples:
                # Check if all 10 are blue (1)
                if all(colors[i] == 1 for i in tri_indices):
                    blue_v += 1
            return blue_v

        # Initialize population
        pop_size = 50
        population = []
        
        # Seed with random and some structured
        for i in range(pop_size):
            if i == 0:
                # Random
                colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
            elif i == 1:
                # All 0
                colors = np.zeros(num_triples, dtype=np.int8)
            elif i == 2:
                # All 1
                colors = np.ones(num_triples, dtype=np.int8)
            else:
                # Random with bias?
                colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
            population.append(colors)
            
        best_fitness = float('inf')
        best_colors = None
        
        local_deadline = time.time() + time_limit
        
        generation = 0
        while time.time() < local_deadline:
            generation += 1
            
            # Evaluate population
            fitnesses = []
            for ind in population:
                f = fitness(ind)
                fitnesses.append(f)
                if f < best_fitness:
                    best_fitness = f
                    best_colors = ind.copy()
                    if f == 0:
                        return best_colors.tolist()
                        
            # Selection (Tournament)
            new_population = []
            # Elitism
            best_idx = fitnesses.index(min(fitnesses))
            new_population.append(population[best_idx].copy())
            
            while len(new_population) < pop_size:
                # Pick 2 parents
                p1_idx = random.randint(0, pop_size - 1)
                p2_idx = random.randint(0, pop_size - 1)
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                
                # Crossover (Uniform)
                mask = np.random.randint(0, 2, num_triples, dtype=bool)
                child = np.where(mask, p1, p2)
                
                # Mutation
                # Bias mutation to break blue K5s? 
                # Simple random mutation
                mutation_rate = 0.05
                for i in range(num_triples):
                    if random.random() < mutation_rate:
                        child[i] = 1 - child[i]
                        
                new_population.append(child)
                
            population = new_population
            
            # If we've been stuck at the same best fitness for a while, restart?
            # For now, just keep going.
            
        return None

    # Fallback: Simple local search if GA fails or for smaller n
    def solve_n_local(n, time_limit):
        if n < 4:
            return None
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        triple_to_k4 = [[] for _ in range(num_triples)]
        triple_to_k5 = [[] for _ in range(num_triples)]
        
        for i, k4 in enumerate(k4s):
            for tri in combinations(k4, 3):
                triple_to_k4[triple_idx[tri]].append(i)
                
        for i, k5 in enumerate(k5s):
            for tri in combinations(k5, 3):
                triple_to_k5[triple_idx[tri]].append(i)
                
        # Random init
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        red_count_k4 = np.zeros(len(k4s), dtype=np.int8)
        blue_count_k5 = np.zeros(len(k5s), dtype=np.int8)
        
        for t_idx in range(num_triples):
            col = colors[t_idx]
            for k4_idx in triple_to_k4[t_idx]:
                if col == 0:
                    red_count_k4[k4_idx] += 1
            for k5_idx in triple_to_k5[t_idx]:
                if col == 1:
                    blue_count_k5[k5_idx] += 1
                    
        red_violations = np.sum(red_count_k4 == 4)
        blue_violations = np.sum(blue_count_k5 == 10)
        
        local_deadline = time.time() + time_limit
        
        while time.time() < local_deadline:
            t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            for k4_idx in triple_to_k4[t_idx]:
                if old_color == 0:
                    old_cnt = red_count_k4[k4_idx]
                    new_cnt = old_cnt - 1
                    if old_cnt == 4: delta_red -= 1
                    elif old_cnt == 3: delta_red += 1
                    red_count_k4[k4_idx] = new_cnt
                else:
                    old_cnt = red_count_k4[k4_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 3: delta_red += 1
                    elif old_cnt == 4: delta_red -= 1
                    red_count_k4[k4_idx] = new_cnt
                    
            for k5_idx in triple_to_k5[t_idx]:
                if old_color == 1:
                    old_cnt = blue_count_k5[k5_idx]
                    new_cnt = old_cnt - 1
                    if old_cnt == 10: delta_blue -= 1
                    elif old_cnt == 9: delta_blue += 1
                    blue_count_k5[k5_idx] = new_cnt
                else:
                    old_cnt = blue_count_k5[k5_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 9: delta_blue += 1
                    elif old_cnt == 10: delta_blue -= 1
                    blue_count_k5[k5_idx] = new_cnt
                    
            delta_E = delta_red + delta_blue
            
            if delta_E <= 0:
                colors[t_idx] = new_color
                red_violations += delta_red
                blue_violations += delta_blue
            else:
                # Accept with probability exp(-delta/T) - simple SA
                T = 0.5
                if random.random() < math.exp(-delta_E / T):
                    colors[t_idx] = new_color
                    red_violations += delta_red
                    blue_violations += delta_blue
                else:
                    # Revert
                    for k4_idx in triple_to_k4[t_idx]:
                        if old_color == 0: red_count_k4[k4_idx] += 1
                        else: red_count_k4[k4_idx] -= 1
                    for k5_idx in triple_to_k5[t_idx]:
                        if old_color == 1: blue_count_k5[k5_idx] += 1
                        else: blue_count_k5[k5_idx] -= 1
            
            if red_violations == 0 and blue_violations == 0:
                return colors.tolist()
                
        return None

    best_n = 0
    best_colors = []
    
    # Try n from 34 down to 1
    # Allocate time based on n
    # n=34 is hard. Give it 60s.
    # n=33 give it 30s.
    # etc.
    
    time_budgets = {
        34: 60.0,
        33: 20.0,
        32: 10.0,
        31: 5.0,
        30: 2.0
    }
    
    for n in range(34, 0, -1):
        if time_left() < 5.0:
            break
            
        budget = time_budgets.get(n, 1.0)
        
        # Try GA first for larger n
        if n >= 20:
            result = solve_n_ga(n, budget)
        else:
            result = solve_n_local(n, budget)
            
        if result is not None:
            best_n = n
            best_colors = result
            break # Found valid, stop
            
    if best_n == 0:
        # Fallback to a trivial valid coloring for small n
        # n=3 is always valid (no K4 or K5)
        best_n = 3
        triples = list(combinations(range(3), 3))
        best_colors = [0] * len(triples)
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

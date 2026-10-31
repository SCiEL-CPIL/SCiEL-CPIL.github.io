import time
import math
import random
import numpy as np
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
        return s
    
    # Genetic Algorithm with Simulated Annealing
    def genetic_search(max_generations=200, pop_size=30):
        nonlocal best_score, best_set
        
        # Initialize population with diverse sets
        population = []
        
        # Add known good structures
        golomb_rulers = [
            [0, 1],
            [0, 1, 3],
            [0, 1, 4, 6],
            [0, 1, 4, 9, 11],
            [0, 1, 4, 10, 12, 17],
            [0, 1, 4, 10, 18, 23, 25],
            [0, 1, 4, 10, 18, 23, 31, 34],
            [0, 1, 4, 10, 18, 23, 31, 42, 45],
        ]
        for gr in golomb_rulers:
            population.append(list(gr))
        
        # Add random sets with various characteristics
        random.seed(123)
        for _ in range(pop_size - len(population)):
            max_val = random.randint(10, 120)
            n = random.randint(3, 10)
            inner = random.sample(range(1, max_val), min(n - 1, max_val - 1))
            cand = sorted([0] + inner + [max_val])
            population.append(cand)
        
        # Add arithmetic progressions
        for step in [2, 3, 5, 7]:
            for max_val in [30, 50, 80]:
                ap = list(range(0, max_val + 1, step))
                if len(ap) >= 3:
                    population.append(ap)
        
        # Add geometric-like sets
        for ratio in [2, 3]:
            g = [0, 1]
            while g[-1] * ratio <= 100:
                g.append(g[-1] * ratio)
            population.append(g)
        
        # Evaluate initial population
        pop_scores = []
        for i, ind in enumerate(population):
            s = try_set(ind)
            pop_scores.append(s)
        
        # Simulated annealing parameters
        T = 0.05  # Initial temperature
        cooling = 0.995
        T_min = 0.001
        
        generation = 0
        while time.time() < deadline - 15 and generation < max_generations and T > T_min:
            generation += 1
            
            # Sort by fitness
            indexed = list(zip(pop_scores, population))
            indexed.sort(key=lambda x: x[0], reverse=True)
            pop_scores = [x[0] for x in indexed]
            population = [x[1] for x in indexed]
            
            # Elitism: keep top individuals
            elite_count = max(3, pop_size // 5)
            new_population = [list(population[i]) for i in range(min(elite_count, len(population)))]
            
            # Crossover and mutation to fill rest of population
            while len(new_population) < pop_size:
                if time.time() > deadline - 15:
                    break
                
                # Select parents using tournament selection
                def tournament_select(k=3):
                    candidates = random.sample(range(len(population)), min(k, len(population)))
                    best_idx = max(candidates, key=lambda i: pop_scores[i])
                    return list(population[best_idx])
                
                parent1 = tournament_select()
                parent2 = tournament_select()
                
                # Crossover: blend two sets
                child = list(parent1)
                
                # With some probability, mix elements from parent2
                if random.random() < 0.7:
                    for el in parent2:
                        if el == 0:
                            continue
                        if random.random() < 0.3:
                            if el not in child:
                                child.append(el)
                
                # Mutation
                if random.random() < 0.8:
                    mutation_type = random.random()
                    if mutation_type < 0.3:
                        # Add element
                        max_cur = max(child) if child else 0
                        new_el = random.randint(1, max_cur + 50)
                        if new_el not in child:
                            child.append(new_el)
                    elif mutation_type < 0.5:
                        # Remove element
                        if len(child) > 2:
                            idx = random.randint(1, len(child) - 1)
                            child.pop(idx)
                    elif mutation_type < 0.7:
                        # Replace element
                        if len(child) > 2:
                            idx = random.randint(1, len(child) - 1)
                            old_el = child[idx]
                            new_el = max(1, old_el + random.randint(-20, 20))
                            if new_el not in child:
                                child[idx] = new_el
                                child = sorted(set(child))
                    else:
                        # Shift all elements
                        shift = random.randint(-5, 5)
                        new_child = []
                        for el in child:
                            if el == 0:
                                new_child.append(0)
                            else:
                                new_el = max(1, el + shift)
                                if new_el not in new_child:
                                    new_child.append(new_el)
                        if new_child and 0 in new_child:
                            child = sorted(set(new_child))
                
                child = sorted(set(child))
                if 0 in child and len(child) >= 2:
                    s = try_set(child)
                    if s is not None and s > -1:
                        new_population.append(child)
                
                if len(new_population) >= pop_size:
                    break
            
            if not new_population:
                break
            
            # Evaluate new population
            new_scores = []
            for ind in new_population:
                s = compute_score(ind)
                new_scores.append(s)
            
            # Accept based on simulated annealing
            # Keep best of old and new, but allow some worse individuals to escape local optima
            combined = list(zip(pop_scores, population)) + list(zip(new_scores, new_population))
            combined.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top pop_size individuals, but with some probability keep a random worse one
            population = [x[1] for x in combined[:pop_size]]
            pop_scores = [x[0] for x in combined[:pop_size]]
            
            # With small probability, replace a random individual with a fresh random set
            if random.random() < 0.1 and len(population) > 5:
                idx = random.randint(5, len(population) - 1)
                max_val = random.randint(10, 100)
                n = random.randint(3, 10)
                inner = random.sample(range(1, max_val), min(n - 1, max_val - 1))
                fresh = sorted([0] + inner + [max_val])
                s = try_set(fresh)
                if s is not None and s > -1:
                    population[idx] = fresh
                    pop_scores[idx] = s
            
            # Cool down
            T *= cooling
        
        return best_score, best_set
    
    # Run genetic search
    gen_score, gen_set = genetic_search(max_generations=150, pop_size=25)
    
    # Refine the best set with local search
    current = list(gen_set)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current) if current else 0
        
        # Try adding an element
        if not improved:
            for new_el in range(1, max_cur + 30):
                if time.time() > deadline - 10:
                    break
                if new_el in current:
                    continue
                cand = sorted(current + [new_el])
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # Try replacing an element
        if not improved and len(current) >= 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                old_el = current[i]
                for delta in range(-30, 31):
                    if time.time() > deadline - 10:
                        break
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current or new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # Try removing an element
        if not improved and len(current) > 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
    
    return list(best_set)

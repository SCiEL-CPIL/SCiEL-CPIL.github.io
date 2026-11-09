import time
import random
import math
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    N = 32
    MAX_VAL = 512

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        count = 0
        m = mask
        # Check d = 1..255
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        lst = []
        m = mask
        while m:
            if m & 1:
                lst.append(1)
            m >>= 1
            if m:
                lst.append(0)
        # This is inefficient. Better:
        lst = []
        m = mask
        for i in range(1, MAX_VAL + 1):
            if (mask >> i) & 1:
                lst.append(i)
        return lst

    def delta_score(mask, add_val, remove_val):
        """
        Calculate change in score when adding add_val and removing remove_val.
        Returns (new_score, delta)
        """
        if add_val == remove_val:
            return score_mask(mask), 0
            
        new_mask = mask
        if add_val:
            new_mask |= (1 << add_val)
        if remove_val:
            new_mask &= ~(1 << remove_val)
        
        new_s = score_mask(new_mask)
        old_s = score_mask(mask)
        return new_s, new_s - old_s

    # Generate initial population
    pop_size = 60
    population = []
    
    # Random starts
    for _ in range(pop_size // 2):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        population.append((lst, list_to_mask(lst)))
    
    # Structured starts
    population.append((list(range(1, 1 + 31 * 16, 16)), list_to_mask(list(range(1, 1 + 31 * 16, 16)))))
    population.append((list(range(1, 33)), list_to_mask(list(range(1, 33)))))
    population.append((list(range(481, 513)), list_to_mask(list(range(481, 513)))))
    population.append((list(range(1, 1 + 31 * 8, 8)), list_to_mask(list(range(1, 1 + 31 * 8, 8)))))
    population.append((list(range(1, 1 + 31 * 4, 4)), list_to_mask(list(range(1, 1 + 31 * 4, 4)))))
    
    # Greedy construction for a few individuals
    def greedy_construct():
        cur_set = set()
        cur_mask = 0
        first = random.randint(1, MAX_VAL)
        cur_set.add(first)
        cur_mask |= (1 << first)
        
        for _ in range(N - 1):
            best_elem = None
            best_new_score = -1
            for _ in range(30):
                elem = random.randint(1, MAX_VAL)
                if elem in cur_set:
                    continue
                new_mask = cur_mask | (1 << elem)
                s = score_mask(new_mask)
                if s > best_new_score:
                    best_new_score = s
                    best_elem = elem
            if best_elem is not None:
                cur_set.add(best_elem)
                cur_mask |= (1 << best_elem)
        
        return list(cur_set), cur_mask

    for _ in range(pop_size // 4):
        lst, mask = greedy_construct()
        population.append((lst, mask))

    # Fill remaining with random
    while len(population) < pop_size:
        lst = random.sample(range(1, MAX_VAL + 1), N)
        population.append((lst, list_to_mask(lst)))

    # Evaluate initial population
    best_score = -1
    best_list = None
    best_mask = 0

    for lst, mask in population:
        s = score_mask(mask)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = mask

    # Genetic Algorithm
    generation = 0
    max_generations = 1000
    elite_size = 5
    mutation_rate = 0.15
    crossover_rate = 0.8

    while time.time() < deadline and generation < max_generations:
        generation += 1
        
        # Evaluate population fitness
        scored_pop = []
        for lst, mask in population:
            s = score_mask(mask)
            scored_pop.append((s, lst, mask))
        
        # Sort by fitness
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_list = scored_pop[0][1][:]
            best_mask = scored_pop[0][2]
        
        # Selection: Tournament
        def tournament_select(pop, k=3):
            candidates = random.sample(pop, min(k, len(pop)))
            return max(candidates, key=lambda x: x[0])
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append((scored_pop[i][1][:], scored_pop[i][2]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            parent1 = tournament_select(scored_pop)
            parent2 = tournament_select(scored_pop)
            
            if random.random() < crossover_rate:
                # Crossover: Uniform crossover on sorted lists
                p1_sorted = sorted(parent1[1])
                p2_sorted = sorted(parent2[1])
                
                child_list = []
                for i in range(N):
                    if random.random() < 0.5:
                        child_list.append(p1_sorted[i])
                    else:
                        child_list.append(p2_sorted[i])
                
                # Ensure uniqueness
                child_set = set(child_list)
                if len(child_set) < N:
                    # Fix duplicates by replacing with random values
                    for i in range(len(child_list)):
                        if child_list[i] in child_set:
                            pass
                        else:
                            child_set.add(child_list[i])
                    
                    # This is getting complicated. Let's just take a random element from each parent
                    child_list = []
                    p1_set = set(p1_sorted)
                    p2_set = set(p2_sorted)
                    
                    for _ in range(N):
                        if random.random() < 0.5 and p1_set:
                            val = random.choice(list(p1_set))
                            p1_set.remove(val)
                            child_list.append(val)
                        elif p2_set:
                            val = random.choice(list(p2_set))
                            p2_set.remove(val)
                            child_list.append(val)
                        elif p1_set:
                            val = random.choice(list(p1_set))
                            p1_set.remove(val)
                            child_list.append(val)
                        else:
                            child_list.append(random.randint(1, MAX_VAL))
                    
                    # Ensure exactly N unique elements
                    child_set = set(child_list)
                    if len(child_set) < N:
                        while len(child_set) < N:
                            child_set.add(random.randint(1, MAX_VAL))
                    child_list = list(child_set)[:N]
                
                child_mask = list_to_mask(child_list)
            else:
                # Just copy parent
                child_list = parent1[1][:]
                child_mask = parent1[2]
            
            # Mutation
            if random.random() < mutation_rate:
                num_mutations = random.randint(1, 3)
                for _ in range(num_mutations):
                    idx = random.randint(0, N - 1)
                    old_val = child_list[idx]
                    new_val = random.randint(1, MAX_VAL)
                    
                    # Ensure uniqueness
                    if new_val in child_set:
                        # Find a new unique value
                        while new_val in child_set:
                            new_val = random.randint(1, MAX_VAL)
                    
                    child_set.discard(old_val)
                    child_set.add(new_val)
                    child_list[idx] = new_val
                    child_mask = list_to_mask(child_list)
            
            new_population.append((child_list, child_mask))
        
        population = new_population
    
    # Final local search on best individual
    current_list = best_list[:]
    current_mask = best_mask
    current_set = set(current_list)
    
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.01
    
    while time.time() < deadline:
        # Single swap
        idx = random.randint(0, N - 1)
        old_val = current_list[idx]
        new_val = random.randint(1, MAX_VAL)
        
        if new_val in current_set and new_val != old_val:
            continue
        
        new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
        new_score = score_mask(new_mask)
        delta = new_score - best_score
        
        if delta > 0:
            accept = True
        else:
            if T < 1e-10:
                accept = False
            else:
                accept = random.random() < math.exp(delta / T)
        
        if accept:
            current_list[idx] = new_val
            current_set.discard(old_val)
            current_set.add(new_val)
            current_mask = new_mask
            best_score = new_score
            best_list = current_list[:]
        
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_list

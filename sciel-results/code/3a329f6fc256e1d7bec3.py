import random
import time
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all K(2,2,2) structures
    struct_to_triples = []
    triple_to_structs = [[] for _ in range(num_triples)]
    
    def get_idx(a, b, c):
        return triple_to_idx[tuple(sorted((a, b, c)))]
        
    for six_pts in combinations(range(n), 6):
        p = list(six_pts)
        pairings = []
        def gen_pairings(remaining, current):
            if not remaining:
                pairings.append(tuple(sorted(current, key=lambda x: min(x))))
                return
            first = remaining[0]
            for i in range(1, len(remaining)):
                second = remaining[i]
                new_remaining = remaining[1:i] + remaining[i+1:]
                gen_pairings(new_remaining, current + [(first, second)])
        gen_pairings(p, [])
        
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_struct = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        idx = get_idx(x1, x2, x3)
                        triples_in_struct.append(idx)
                        
            struct_id = len(struct_to_triples)
            struct_to_triples.append(triples_in_struct)
            for t_idx in triples_in_struct:
                triple_to_structs[t_idx].append(struct_id)
                
    num_structs = len(struct_to_triples)
    
    # Helper functions
    def is_valid_add(idx, in_set, counts):
        if in_set[idx]:
            return False
        for s_idx in triple_to_structs[idx]:
            if counts[s_idx] == 7:
                return False
        return True

    def add_triple(idx, in_set, counts):
        in_set[idx] = True
        for s_idx in triple_to_structs[idx]:
            counts[s_idx] += 1

    def remove_triple(idx, in_set, counts):
        in_set[idx] = False
        for s_idx in triple_to_structs[idx]:
            counts[s_idx] -= 1
            
    def get_size(in_set):
        return sum(in_set)
        
    def greedy_construct(random_order):
        in_set = [False] * num_triples
        counts = [0] * num_structs
        for idx in random_order:
            if is_valid_add(idx, in_set, counts):
                add_triple(idx, in_set, counts)
        return [i for i in range(num_triples) if in_set[i]]

    def repair(union_indices):
        """
        Takes a list of triple indices (union of parents), 
        performs greedy construction to make it valid.
        """
        in_set = [False] * num_triples
        counts = [0] * num_structs
        # Randomize order of the union to allow different outcomes
        random.shuffle(union_indices)
        for idx in union_indices:
            if is_valid_add(idx, in_set, counts):
                add_triple(idx, in_set, counts)
        return [i for i in range(num_triples) if in_set[i]]

    def local_improve(solution):
        """
        Improve a solution by trying to add any valid triple.
        Uses multiple passes to ensure saturation.
        """
        in_set = [False] * num_triples
        counts = [0] * num_structs
        for idx in solution:
            add_triple(idx, in_set, counts)
            
        # Try to add more
        order = list(range(num_triples))
        random.shuffle(order)
        improved = True
        while improved:
            improved = False
            for idx in order:
                if is_valid_add(idx, in_set, counts):
                    add_triple(idx, in_set, counts)
                    improved = True
        return [i for i in range(num_triples) if in_set[i]]

    # Initialize Population
    pop_size = 100
    population = []
    random.seed(42)
    
    # Create diverse initial population
    for _ in range(pop_size):
        order = list(range(num_triples))
        random.shuffle(order)
        sol = greedy_construct(order)
        population.append(sol)
        
    # Track best
    best_sol = max(population, key=len)
    best_size = len(best_sol)
    
    start_time = time.time()
    time_limit = 110.0
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Sort population by fitness (size)
        population.sort(key=len, reverse=True)
        
        # Elitism: keep top 10%
        elite_count = max(5, pop_size // 10)
        new_population = population[:elite_count]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection from top 20%
            a = random.choice(population[:max(1, pop_size//5)])
            b = random.choice(population[:max(1, pop_size//5)])
            
            # Crossover: Union
            union = list(set(a) | set(b))
            
            # Repair
            child = repair(union)
            
            # Mutation/Improvement
            if random.random() < 0.7:
                child = local_improve(child)
            
            new_population.append(child)
            
        population = new_population
        
        # Update best
        current_best = max(population, key=len)
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_sol = current_best
            
    return [list(all_triples[i]) for i in best_sol]

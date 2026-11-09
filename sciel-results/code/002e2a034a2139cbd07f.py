import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit
    
    def evaluate(S):
        """Count distinct d > 0 such that a, a+d, a+2d are all in S."""
        S_set = set(S)
        valid_diffs = set()
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def evaluate_fast(S):
        """Faster evaluation using numpy for the inner loop."""
        S_arr = np.array(sorted(S))
        S_set = set(S)
        valid_diffs = set()
        n = len(S_arr)
        for i in range(n):
            a = S_arr[i]
            for j in range(i + 1, n):
                b = S_arr[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def random_set():
        return set(random.sample(range(1, 513), 32))
    
    def structured_set():
        """Create a set with block structure to encourage many progressions."""
        base = random.randint(1, 300)
        s = set()
        configs = [
            (4, 8),
            (8, 4),
            (16, 2),
            (32, 1),
        ]
        nblocks, bsize = random.choice(configs)
        if nblocks * bsize > 32:
            bsize = 32 // nblocks
        spacing = random.randint(max(1, (512 - base) // max(nblocks, 1) - bsize), max(2, (512 - base) // max(nblocks, 1)))
        for i in range(nblocks):
            start = base + i * spacing
            for k in range(bsize):
                val = start + k
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        return s
    
    def greedy_init():
        """Greedy construction: add elements that maximize new progressions."""
        s = set()
        s.add(random.randint(1, 512))
        for _ in range(31):
            best_elem = None
            best_score = -1
            candidates = random.sample(range(1, 513), min(200, 512))
            for cand in candidates:
                if cand in s:
                    continue
                new_s = s | {cand}
                sc = evaluate(new_s)
                if sc > best_score or (sc == best_score and random.random() < 0.1):
                    best_score = sc
                    best_elem = cand
            if best_elem is not None:
                s.add(best_elem)
            else:
                while True:
                    val = random.randint(1, 512)
                    if val not in s:
                        s.add(val)
                        break
        return s
    
    def greedy_init_full():
        """Greedy with full candidate search for better initialization."""
        s = set()
        s.add(random.randint(1, 512))
        for _ in range(31):
            best_elem = None
            best_score = -1
            # Sample more candidates for better greedy choice
            candidates = random.sample(range(1, 513), 300)
            for cand in candidates:
                if cand in s:
                    continue
                new_s = s | {cand}
                sc = evaluate(new_s)
                if sc > best_score or (sc == best_score and random.random() < 0.05):
                    best_score = sc
                    best_elem = cand
            if best_elem is not None:
                s.add(best_elem)
            else:
                while True:
                    val = random.randint(1, 512)
                    if val not in s:
                        s.add(val)
                        break
        return s
    
    # Generate diverse initial candidates
    candidates = [
        set(list(range(1, 27)) + [28, 29, 30, 31, 32, 48]),  # Known good
        set(range(1, 33)),
        random_set(),
        random_set(),
        structured_set(),
        structured_set(),
        greedy_init(),
        greedy_init_full(),
    ]
    
    # Evaluate all candidates
    scored = []
    for c in candidates:
        if len(c) == 32:
            sc = evaluate(c)
            scored.append((sc, c))
    
    scored.sort(key=lambda x: -x[0])
    
    # Genetic Algorithm
    POP_SIZE = 25
    ELITE_SIZE = 5
    MUTATION_RATE = 0.35
    
    # Initialize population
    population = []
    for sc, c in scored[:min(POP_SIZE, len(scored))]:
        population.append((sc, c.copy()))
    
    # Fill rest of population with random/structured
    while len(population) < POP_SIZE:
        if random.random() < 0.5:
            new_c = random_set()
        else:
            new_c = structured_set()
        sc = evaluate(new_c)
        population.append((sc, new_c))
    
    # Sort by fitness
    population.sort(key=lambda x: -x[0])
    
    best_score = population[0][0]
    best_set = population[0][1].copy()
    
    def tournament_select(pop, k=3):
        """Tournament selection."""
        subset = random.sample(pop, min(k, len(pop)))
        return max(subset, key=lambda x: x[0])[1]
    
    def crossover(parent1, parent2):
        """Uniform crossover between two sets."""
        p1 = list(parent1)
        p2 = list(parent2)
        child = set()
        for i in range(32):
            if random.random() < 0.5:
                child.add(p1[i])
            else:
                child.add(p2[i])
        if len(child) != 32:
            child_list = list(child)
            if len(child_list) > 32:
                excess = random.sample(child_list, len(child_list) - 32)
                for e in excess:
                    child.discard(e)
            while len(child) < 32:
                val = random.randint(1, 512)
                if val not in child:
                    child.add(val)
        return child
    
    def mutate(individual):
        """Apply mutations: swap elements."""
        ind = set(individual)
        n_mutations = random.randint(1, 3)
        for _ in range(n_mutations):
            if len(ind) != 32:
                break
            to_remove = random.choice(list(ind))
            ind.discard(to_remove)
            while True:
                to_add = random.randint(1, 512)
                if to_add not in ind:
                    ind.add(to_add)
                    break
        return ind
    
    def two_swap(individual):
        """Remove 2, add 2 for larger jumps."""
        ind = set(individual)
        if len(ind) != 32:
            return ind
        elems = list(ind)
        idx1 = random.randint(0, 31)
        idx2 = random.randint(0, 31)
        if idx1 == idx2:
            return ind
        to_remove1 = elems[idx1]
        to_remove2 = elems[idx2]
        ind.discard(to_remove1)
        ind.discard(to_remove2)
        for _ in range(2):
            while True:
                to_add = random.randint(1, 512)
                if to_add not in ind:
                    ind.add(to_add)
                    break
        return ind
    
    def three_swap(individual):
        """Remove 3, add 3 for even larger jumps."""
        ind = set(individual)
        if len(ind) != 32:
            return ind
        elems = list(ind)
        idxs = random.sample(range(32), 3)
        for idx in idxs:
            ind.discard(elems[idx])
        for _ in range(3):
            while True:
                to_add = random.randint(1, 512)
                if to_add not in ind:
                    ind.add(to_add)
                    break
        return ind
    
    def local_search(individual, max_iters=100):
        """More thorough local search with multiple move types."""
        local_best = set(individual)
        local_score = evaluate(local_best)
        for _ in range(max_iters):
            if time.time() - start_time > timeout - 3.0:
                break
            # Try different move types
            move_type = random.random()
            if move_type < 0.4:
                # Single swap
                new_set = set(local_best)
                to_remove = random.choice(list(new_set))
                new_set.discard(to_remove)
                while True:
                    to_add = random.randint(1, 512)
                    if to_add not in new_set:
                        new_set.add(to_add)
                        break
            elif move_type < 0.7:
                # Two swap
                new_set = two_swap(local_best)
            else:
                # Three swap
                new_set = three_swap(local_best)
            
            sc = evaluate(new_set)
            if sc > local_score:
                local_best = new_set
                local_score = sc
            elif sc == local_score and random.random() < 0.15:
                local_best = new_set
        return local_best, local_score
    
    generation = 0
    no_improvement = 0
    
    while time.time() - start_time < timeout:
        if time.time() - start_time > timeout - 3.0:
            break
        
        generation += 1
        new_population = []
        
        # Elitism: keep best individuals
        for sc, ind in population[:ELITE_SIZE]:
            new_population.append((sc, ind.copy()))
        
        # Generate new individuals
        while len(new_population) < POP_SIZE:
            if time.time() - start_time > timeout - 2.0:
                break
            
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            if random.random() < 0.6:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            if random.random() < MUTATION_RATE:
                r = random.random()
                if r < 0.25:
                    child = two_swap(child)
                elif r < 0.4:
                    child = three_swap(child)
                else:
                    child = mutate(child)
            
            sc = evaluate(child)
            new_population.append((sc, child))
        
        new_population.sort(key=lambda x: -x[0])
        population = new_population
        
        if population[0][0] > best_score:
            best_score = population[0][0]
            best_set = population[0][1].copy()
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement > 12:
            for _ in range(6):
                if time.time() - start_time > timeout - 2.0:
                    break
                new_c = random_set() if random.random() < 0.5 else structured_set()
                sc = evaluate(new_c)
                population.append((sc, new_c))
                population.sort(key=lambda x: -x[0])
                population = population[:POP_SIZE]
            no_improvement = 0
        
        # Local refinement on best individual with more iterations
        if time.time() - start_time < timeout - 5.0:
            local_best, local_score = local_search(best_set, max_iters=80)
            
            if local_score > best_score:
                best_score = local_score
                best_set = local_best.copy()
                population[0] = (best_score, best_set.copy())
                population.sort(key=lambda x: -x[0])
    
    return list(best_set)

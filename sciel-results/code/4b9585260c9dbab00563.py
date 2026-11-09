import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAX_VAL = 512
    random.seed(42)

    # Bitmask representation: bit i corresponds to value i+1 (1-indexed)
    # So value v is at bit position v-1
    # For AP check: a, a+d, a+2d all in set
    # Using 0-indexed positions: pos = v - 1
    # a at pos a-1, a+d at pos a+d-1, a+2d at pos a+2d-1
    # Shift: mask >> d gives (a+d-1) aligned to (a-1) position
    # mask >> (2*d) gives (a+2d-1) aligned to (a-1) position
    
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, 257):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score(mask, old_pos, new_pos):
        """Compute change in score when replacing element at old_pos with new_pos.
        Positions are 0-indexed (value = pos + 1).
        Returns (delta, new_mask)."""
        old_bit = 1 << old_pos
        new_bit = 1 << new_pos
        new_mask = mask ^ old_bit ^ new_bit
        
        delta = 0
        for d in range(1, 257):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta, new_mask

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def gen_initial_population(pop_size):
        pop = []
        seen = set()
        
        def add_individual(lst):
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                m = list_to_mask(lst)
                s = score_mask(m)
                pop.append((lst, m, s))
        
        # Interval
        add_individual(list(range(1, 33)))
        
        # APs
        for step in range(2, 17):
            lst = [1 + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                add_individual(lst)
        
        # Structured: small base + multiples
        for mult in [2, 3, 4, 5, 6, 7, 8]:
            for trial in range(20):
                sizeA = random.randint(3, 8)
                sizeB = random.randint(3, 8)
                A = set(random.sample(range(1, 60), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    lst = random.sample(sorted(S), N)
                    add_individual(lst)
        
        # Random
        while len(pop) < pop_size:
            lst = random.sample(range(1, MAX_VAL + 1), N)
            add_individual(lst)
        
        return pop

    def tournament_select(pop, k=3):
        best = None
        for _ in range(k):
            cand = random.choice(pop)
            if best is None or cand[2] > best[2]:
                best = cand
        return best

    def crossover(parent1, parent2):
        """Uniform crossover with deduplication."""
        child = []
        for i in range(N):
            if random.random() < 0.5:
                child.append(parent1[0][i])
            else:
                child.append(parent2[0][i])
        # Resolve duplicates
        seen = set()
        final = []
        for v in child:
            if v not in seen:
                seen.add(v)
                final.append(v)
        while len(final) < N:
            v = random.randint(1, MAX_VAL)
            if v not in seen:
                seen.add(v)
                final.append(v)
        m = list_to_mask(final)
        s = score_mask(m)
        return (final, m, s)

    def mutate(individual, rate=0.15):
        lst = individual[0][:]
        s_set = set(lst)
        m = individual[1]
        s = individual[2]
        for i in range(N):
            if random.random() < rate:
                old = lst[i]
                old_pos = old - 1
                old_bit = 1 << old_pos
                new = random.randint(1, MAX_VAL)
                attempts = 0
                while new in s_set and attempts < 50:
                    new = random.randint(1, MAX_VAL)
                    attempts += 1
                if new not in s_set:
                    new_pos = new - 1
                    new_bit = 1 << new_pos
                    delta, m = delta_score(m, old_pos, new_pos)
                    s += delta
                    s_set.discard(old)
                    s_set.add(new)
                    lst[i] = new
        return (lst, m, s)

    def local_search(individual, max_iter=100, time_limit=None):
        """Hill climbing using delta scoring."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        
        for _ in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            improved = False
            for i in range(N):
                old = lst[i]
                old_pos = old - 1
                old_bit = 1 << old_pos
                best_new = old
                best_s = s
                # Try several random replacements
                for _ in range(12):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    new_pos = new - 1
                    new_bit = 1 << new_pos
                    delta, _ = delta_score(m, old_pos, new_pos)
                    new_s = s + delta
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= (1 << old_pos) ^ (1 << (best_new - 1))
                    s = best_s
                    improved = True
            if not improved:
                break
        return (lst, m, s)

    # Initialize population
    pop_size = 40
    population = gen_initial_population(pop_size)
    population.sort(key=lambda x: x[2], reverse=True)
    
    best_individual = population[0]
    best_score = best_individual[2]
    best_list = best_individual[0][:]
    best_mask = best_individual[1]
    
    print(f"Initial best: {best_score}", file=sys.stderr)
    
    # Genetic algorithm loop
    generation = 0
    no_improve = 0
    elite_count = 6
    
    while time.time() < deadline:
        generation += 1
        
        if generation % 20 == 0:
            print(f"Gen {generation}, best={best_score}", file=sys.stderr)
        
        new_population = []
        
        # Elitism
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)
            
            if random.random() < 0.35:
                child = crossover(p1, p2)
            elif random.random() < 0.6:
                child = mutate(p1, rate=0.12)
            else:
                # Double mutation
                child = mutate(p1, rate=0.08)
                child = mutate(child, rate=0.08)
            
            # Local search
            if time.time() < deadline - 8:
                child = local_search(child, max_iter=40, time_limit=deadline - 3)
            
            new_population.append(child)
        
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        # Track best
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve = 0
        else:
            no_improve += 1
        
        # If stuck, inject some diversity
        if no_improve > 15:
            for _ in range(5):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
            population = population[:pop_size]
            population.sort(key=lambda x: x[2], reverse=True)
            no_improve = 0
    
    return best_list

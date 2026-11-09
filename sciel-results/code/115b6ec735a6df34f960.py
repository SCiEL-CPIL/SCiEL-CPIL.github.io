import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255
    random.seed(12345)

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score(mask, old_bit, new_bit):
        """Compute change in score when replacing old_bit with new_bit in mask."""
        new_mask = mask ^ old_bit ^ new_bit
        delta = 0
        for d in range(1, MAX_D + 1):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    # Precompute: for each element position, which d values could be affected
    # When we change element x to y, d is affected if:
    # - x or y is part of a 3-AP with difference d
    def gen_initial_population(pop_size):
        pop = []
        seen = set()
        
        def add_candidate(lst):
            if len(lst) != N or len(set(lst)) != N:
                return
            if any(v < 1 or v > MAX_VAL for v in lst):
                return
            m = list_to_mask(lst)
            s = score_mask(m)
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                pop.append((lst, m, s))
        
        # Interval
        add_candidate(list(range(1, 33)))
        
        # APs
        for step in range(3, 17):
            lst = [1 + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                add_candidate(lst)
        
        # Sumset structures
        for mult in [2, 3, 4, 5, 6]:
            for trial in range(50):
                sizeA = random.randint(2, 8)
                sizeB = random.randint(2, 8)
                A = set(random.sample(range(1, 80), sizeA))
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
                    add_candidate(lst)
        
        # Random
        while len(pop) < pop_size:
            lst = random.sample(range(1, MAX_VAL + 1), N)
            add_candidate(lst)
        
        return pop

    def tournament_select(pop, k=3):
        best = None
        for _ in range(k):
            cand = random.choice(pop)
            if best is None or cand[2] > best[2]:
                best = cand
        return best

    def crossover(parent1, parent2):
        """Uniform crossover with duplicate resolution."""
        child = []
        for i in range(N):
            if random.random() < 0.5:
                child.append(parent1[0][i])
            else:
                child.append(parent2[0][i])
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
                old_bit = 1 << old
                new = random.randint(1, MAX_VAL)
                attempts = 0
                while new in s_set and attempts < 50:
                    new = random.randint(1, MAX_VAL)
                    attempts += 1
                if new not in s_set:
                    new_bit = 1 << new
                    delta = delta_score(m, old_bit, new_bit)
                    s += delta
                    s_set.discard(old)
                    s_set.add(new)
                    lst[i] = new
                    m ^= old_bit ^ new_bit
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
                old_bit = 1 << old
                best_new = old
                best_s = s
                # Try several random replacements
                for _ in range(12):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    new_bit = 1 << new
                    delta = delta_score(m, old_bit, new_bit)
                    new_s = s + delta
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= old_bit ^ (1 << best_new)
                    s = best_s
                    improved = True
            if not improved:
                break
        return (lst, m, s)

    # Initialize population
    pop_size = 50
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
    elite_count = 5
    
    while time.time() < deadline - 10:
        generation += 1
        if generation % 5 == 0:
            print(f"Gen {generation}, best={best_score}, time={time.time()-start_time:.1f}s", file=sys.stderr)
        
        new_population = []
        
        # Elitism
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline - 10:
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)
            
            if random.random() < 0.3:
                child = crossover(p1, p2)
            else:
                child = mutate(p1, rate=0.1)
            
            # Local search on child
            if time.time() < deadline - 12:
                child = local_search(child, max_iter=30, time_limit=deadline - 12)
            
            new_population.append(child)
        
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve = 0
        else:
            no_improve += 1
        
        # Restart if stuck
        if no_improve > 20:
            for _ in range(8):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
            # Structured restarts
            for mult in [2, 3, 4]:
                sizeA = random.randint(3, 7)
                sizeB = random.randint(3, 7)
                A = set(random.sample(range(1, 100), sizeA))
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
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    population.append((lst, m, s))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve = 0
    
    # Final exhaustive polish: try all single-element replacements
    if time.time() < deadline - 3:
        lst = best_list[:]
        m = best_mask
        s = best_score
        s_set = set(lst)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                old = lst[i]
                old_bit = 1 << old
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    new_bit = 1 << new
                    delta = delta_score(m, old_bit, new_bit)
                    if delta > 0:
                        s_set.discard(old)
                        s_set.add(new)
                        lst[i] = new
                        m ^= old_bit ^ new_bit
                        s += delta
                        improved = True
                        break
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m
    
    # Second polish pass with 2-swap moves if time permits
    if time.time() < deadline - 2:
        lst = best_list[:]
        m = best_mask
        s = best_score
        s_set = set(lst)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    if time.time() > deadline - 1.5:
                        break
                    old_i = lst[i]
                    old_j = lst[j]
                    old_bit_i = 1 << old_i
                    old_bit_j = 1 << old_j
                    # Try swapping with two new values
                    for new_i in range(1, MAX_VAL + 1):
                        if new_i == old_i or new_i == old_j or new_i in s_set:
                            continue
                        for new_j in range(1, MAX_VAL + 1):
                            if new_j == old_i or new_j == old_j or new_j in s_set or new_j == new_i:
                                continue
                            new_bit_i = 1 << new_i
                            new_bit_j = 1 << new_j
                            new_mask = m ^ old_bit_i ^ old_bit_j ^ new_bit_i ^ new_bit_j
                            new_s = score_mask(new_mask)
                            if new_s > s:
                                s_set.discard(old_i)
                                s_set.discard(old_j)
                                s_set.add(new_i)
                                s_set.add(new_j)
                                lst[i] = new_i
                                lst[j] = new_j
                                m = new_mask
                                s = new_s
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m
    
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list
# EVOLVE-BLOCK-END

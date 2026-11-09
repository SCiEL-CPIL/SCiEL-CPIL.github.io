import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAX_VAL = 512
    MAX_D = 256  # max d such that a+2d <= 512 for some a >= 1, i.e., 2d <= 511, d <= 255
    random.seed(42)

    # Precompute for each d in 1..255, the list of a values where (a, a+d, a+2d) fits in 1..512
    # For a given d, valid a: 1 <= a and a + 2d <= 512, so a <= 512 - 2d
    # Number of valid a for difference d: max(0, 512 - 2d)
    
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask for some a."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # For fast incremental scoring: precompute for each x and d, the set of a values
    # such that x is one of {a, a+d, a+2d}
    # When we toggle x, for each d, we need to know if the AP count changes
    
    # Actually, let's use a different approach: 
    # For each d, precompute the list of triples (a, a+d, a+2d)
    # Then for a given mask, count how many d have at least one triple fully in mask
    
    # For incremental: when we add/remove element x, for each d, 
    # we need to check if any triple containing x becomes fully in/out of mask
    
    # Let's precompute: for each x (1..512) and each d (1..255), 
    # the list of a values such that x is in {a, a+d, a+2d}
    # i.e., x = a, x = a+d, or x = a+2d
    # a = x, a = x-d, a = x-2d
    # And we need 1 <= a and a + 2d <= 512
    
    # For fast scoring, let's precompute for each d the list of valid a's
    # and use bit operations
    
    # Let's try a hybrid: use the fast bit-parallel score for the GA,
    # and do local search with full rescore but fewer iterations
    
    # Population-based genetic algorithm
    
    def gen_initial_population(pop_size):
        pop = []
        # Structured: intervals
        for offset in range(0, 481, 10):  # 1..481+31 <= 512
            lst = list(range(offset + 1, offset + N + 1))
            if max(lst) <= MAX_VAL:
                m = list_to_mask(lst)
                s = score_mask(m)
                pop.append((lst, m, s))
        
        # Structured: APs
        for step in range(5, 16):
            lst = [1 + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                m = list_to_mask(lst)
                s = score_mask(m)
                pop.append((lst, m, s))
        
        # Random
        for _ in range(pop_size - len(pop)):
            lst = random.sample(range(1, MAX_VAL + 1), N)
            m = list_to_mask(lst)
            s = score_mask(m)
            pop.append((lst, m, s))
        
        return pop[:pop_size]

    def crossover(parent1, parent2):
        """Uniform crossover."""
        child1 = []
        child2 = []
        for i in range(N):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2

    def mutate(lst, rate=0.3):
        """Random mutation: replace some elements."""
        new_lst = lst[:]
        for i in range(N):
            if random.random() < rate:
                new_lst[i] = random.randint(1, MAX_VAL)
        # Ensure distinctness
        seen = set()
        valid = []
        for x in new_lst:
            if x not in seen and 1 <= x <= MAX_VAL:
                seen.add(x)
                valid.append(x)
        while len(valid) < N:
            x = random.randint(1, MAX_VAL)
            if x not in seen:
                seen.add(x)
                valid.append(x)
        return valid[:N]

    def local_search(lst, max_iters=500):
        """Greedy local search: try all single swaps for improvement."""
        lst = lst[:]
        m = list_to_mask(lst)
        s = score_mask(m)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            for i in range(N):
                old_val = lst[i]
                for new_val in range(1, MAX_VAL + 1):
                    if new_val == old_val or new_val in set(lst):
                        continue
                    new_mask = m ^ (1 << old_val) ^ (1 << new_val)
                    new_score = score_mask(new_mask)
                    if new_score > s:
                        lst[i] = new_val
                        m = new_mask
                        s = new_score
                        improved = True
                        break
        return lst, s

    # Initialize population
    pop_size = 60
    population = gen_initial_population(pop_size)
    
    # Sort by score
    population.sort(key=lambda x: -x[2])
    best_list = population[0][0][:]
    best_mask = population[0][1]
    best_score = population[0][2]
    
    print(f"Initial best score: {best_score}", file=sys.stderr)
    
    # Genetic algorithm loop
    gen = 0
    no_improve_gens = 0
    elite_size = 10
    
    while time.time() < deadline - 30:  # Leave 30s for final polish
        gen += 1
        if gen % 5 == 0:
            print(f"Gen {gen}, best={best_score}, top5={[p[2] for p in population[:5]]}", file=sys.stderr)
        
        # Selection: tournament
        def tournament_select(k=5):
            candidates = random.sample(population, min(k, len(population)))
            return max(candidates, key=lambda x: x[2])
        
        new_pop = []
        
        # Elitism
        for i in range(elite_size):
            new_pop.append(population[i])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 30:
                break
            p1 = tournament_select()
            p2 = tournament_select()
            c1, c2 = crossover(p1[0], p2[0])
            
            # Mutate
            c1 = mutate(c1, rate=0.2)
            c2 = mutate(c2, rate=0.2)
            
            m1 = list_to_mask(c1)
            s1 = score_mask(m1)
            m2 = list_to_mask(c2)
            s2 = score_mask(m2)
            
            new_pop.append((c1, m1, s1))
            new_pop.append((c2, m2, s2))
        
        population = new_pop
        population.sort(key=lambda x: -x[2])
        
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve_gens = 0
        else:
            no_improve_gens += 1
        
        # If stuck, inject diversity
        if no_improve_gens > 15:
            for i in range(5):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population[i] = (lst, m, s)
            population.sort(key=lambda x: -x[2])
            no_improve_gens = 0
    
    # Final local search polish on best and top few
    print(f"Before polish: best={best_score}", file=sys.stderr)
    
    candidates_to_polish = [p[0][:] for p in population[:10]]
    
    for lst in candidates_to_polish:
        if time.time() > deadline - 5:
            break
        polished, s = local_search(lst, max_iters=200)
        if s > best_score:
            best_score = s
            best_list = polished
            best_mask = list_to_mask(polished)
            print(f"Polish improved: {best_score}", file=sys.stderr)
    
    # One more round of local search on the best
    if time.time() < deadline - 5:
        polished, s = local_search(best_list, max_iters=500)
        if s > best_score:
            best_score = s
            best_list = polished
            best_mask = list_to_mask(polished)
    
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list

# EVOLVE-BLOCK-END

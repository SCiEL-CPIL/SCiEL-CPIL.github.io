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
    MAX_D = 255  # max d such that a + 2*d <= 512 for some a >= 1
    random.seed(12345)

    def time_left():
        return deadline - time.time()

    # Bitmask helpers: bit (x-1) corresponds to value x
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score(mask, old_val, new_val):
        """Compute change in score when replacing old_val with new_val in mask.
        Uses 0-indexed bits: value v -> bit (v-1)."""
        old_bit = 1 << (old_val - 1)
        new_bit = 1 << (new_val - 1)
        new_mask = mask ^ old_bit ^ new_bit
        # For each d, check if the AP status changes
        delta = 0
        for d in range(1, MAX_D + 1):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta

    def gen_initial_population(pop_size):
        pop = []
        seen = set()

        def add_individual(lst):
            if len(lst) != N or len(set(lst)) != N:
                return
            if any(x < 1 or x > MAX_VAL for x in lst):
                return
            m = list_to_mask(lst)
            s = score_mask(m)
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                pop.append((lst, m, s))

        # Interval
        add_individual(list(range(1, 33)))
        # APs
        for step in range(5, 16):
            lst = [1 + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                add_individual(lst)
        # Sumset structures
        for mult in [2, 3, 4, 5]:
            for trial in range(30):
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
        """Uniform crossover: each position picks from parent1 or parent2."""
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
        # Fill remaining with random values
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
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        for i in range(N):
            if random.random() < rate:
                old = lst[i]
                new = random.randint(1, MAX_VAL)
                attempts = 0
                while new in s_set and attempts < 50:
                    new = random.randint(1, MAX_VAL)
                    attempts += 1
                if new not in s_set:
                    delta = delta_score(m, old, new)
                    s += delta
                    s_set.discard(old)
                    s_set.add(new)
                    lst[i] = new
                    m ^= (1 << (old - 1)) ^ (1 << (new - 1))
        return (lst, m, s)

    def local_search(individual, max_iter=200):
        """Hill climbing on a single individual using delta scoring."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        for _ in range(max_iter):
            improved = False
            for i in range(N):
                old = lst[i]
                best_new = old
                best_s = s
                # Try a few random replacements
                for _ in range(8):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    delta = delta_score(m, old, new)
                    new_s = s + delta
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= (1 << (old - 1)) ^ (1 << (best_new - 1))
                    s = best_s
                    improved = True
            if not improved:
                break
        return (lst, m, s)

    def simulated_annealing(individual, max_iter=500):
        """Simulated annealing to escape local minima."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        T = 5.0
        cooling = 0.995
        best_lst = lst[:]
        best_m = m
        best_s = s
        for _ in range(max_iter):
            if time_left() < 10:
                break
            i = random.randint(0, N - 1)
            old = lst[i]
            new = random.randint(1, MAX_VAL)
            attempts = 0
            while new in s_set and new != old and attempts < 50:
                new = random.randint(1, MAX_VAL)
                attempts += 1
            if new == old or new in s_set:
                continue
            delta = delta_score(m, old, new)
            new_s = s + delta
            if new_s >= s or random.random() < math.exp(-delta / T):
                s_set.discard(old)
                s_set.add(new)
                lst[i] = new
                m ^= (1 << (old - 1)) ^ (1 << (new - 1))
                s = new_s
                if s > best_s:
                    best_s = s
                    best_lst = lst[:]
                    best_m = m
            T *= cooling
        return (best_lst, best_m, best_s)

    # Initialize population
    pop_size = 40
    population = gen_initial_population(pop_size)

    # Sort by fitness
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

    while time_left() > 15:
        generation += 1
        if generation % 10 == 0:
            print(f"Gen {generation}, best={best_score}", file=sys.stderr)

        new_population = []

        # Elitism: keep top individuals
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])

        # Generate offspring
        while len(new_population) < pop_size:
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)

            if random.random() < 0.3:
                # Crossover
                child = crossover(p1, p2)
            else:
                # Mutation of best parent
                child = mutate(p1, rate=0.1)

            # Local search on child (limited iterations)
            if time_left() > 20:
                child = local_search(child, max_iter=50)

            new_population.append(child)

        # Update population
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

        # Restart mechanism if stuck
        if no_improve > 30:
            # Add fresh random individuals to maintain diversity
            for _ in range(5):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
            # Also try structured restarts
            for mult in [2, 3]:
                sizeA = random.randint(3, 7)
                sizeB = random.randint(3, 7)
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
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    population.append((lst, m, s))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve = 0

    # Simulated annealing on best individual to escape local minima
    if time_left() > 20:
        sa_result = simulated_annealing((best_list, best_mask, best_score), max_iter=300)
        if sa_result[2] > best_score:
            best_score = sa_result[2]
            best_list = sa_result[0][:]
            best_mask = sa_result[1]
            print(f"After SA: {best_score}", file=sys.stderr)

    # Final polish: exhaustive single-swap local search on best using delta scoring
    if time_left() > 5:
        lst = best_list[:]
        m = best_mask
        s = best_score
        s_set = set(lst)
        improved = True
        while improved and time_left() > 2:
            improved = False
            for i in range(N):
                old = lst[i]
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    delta = delta_score(m, old, new)
                    new_s = s + delta
                    if new_s > s:
                        s_set.discard(old)
                        s_set.add(new)
                        lst[i] = new
                        m ^= (1 << (old - 1)) ^ (1 << (new - 1))
                        s = new_s
                        improved = True
                        break
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list
# EVOLVE-BLOCK-END

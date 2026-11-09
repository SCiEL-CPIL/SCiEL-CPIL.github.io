import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    N = 32
    MAX_VAL = 512
    random.seed(42)

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score(mask, old_bit, new_bit):
        new_mask = mask ^ old_bit ^ new_bit
        delta = 0
        for d in range(1, 256):
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

    def gen_structured_seed():
        """Generate a structured seed based on sumset or grid constructions."""
        strategy = random.random()
        
        if strategy < 0.2:
            # Simple interval
            offset = random.randint(1, MAX_VAL - N + 1)
            lst = list(range(offset, offset + N))
            return lst
        
        elif strategy < 0.4:
            # AP structure
            step = random.randint(4, 20)
            if (N - 1) * step + 1 <= MAX_VAL:
                lst = [1 + i * step for i in range(N)]
                return lst
            else:
                return random.sample(range(1, MAX_VAL + 1), N)
        
        elif strategy < 0.7:
            # Sumset A + k*B
            mult = random.choice([2, 3, 4, 5, 6])
            sizeA = random.randint(4, 10)
            sizeB = random.randint(4, 10)
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
                return lst
            else:
                return random.sample(range(1, MAX_VAL + 1), N)
        
        else:
            # Grid-like structure
            rows = random.randint(3, 8)
            cols = random.randint(3, 8)
            if rows * cols < N:
                return random.sample(range(1, MAX_VAL + 1), N)
            
            row_step = random.randint(10, 60)
            col_step = random.randint(3, 15)
            base = random.randint(1, 50)
            
            S = set()
            for r in range(rows):
                for c in range(cols):
                    v = base + r * row_step + c * col_step
                    if 1 <= v <= MAX_VAL:
                        S.add(v)
            
            if len(S) >= N:
                lst = random.sample(sorted(S), N)
                return lst
            else:
                return random.sample(range(1, MAX_VAL + 1), N)

    def gen_initial_population(pop_size):
        pop = []
        seen = set()
        
        for _ in range(pop_size * 2):
            if random.random() < 0.6:
                lst = gen_structured_seed()
            else:
                lst = random.sample(range(1, MAX_VAL + 1), N)
            
            m = list_to_mask(lst)
            s = score_mask(m)
            key = tuple(sorted(lst))
            
            if key not in seen:
                seen.add(key)
                pop.append((lst, m, s))
            
            if len(pop) >= pop_size:
                break
        
        # Fill with random if needed
        while len(pop) < pop_size:
            lst = random.sample(range(1, MAX_VAL + 1), N)
            m = list_to_mask(lst)
            s = score_mask(m)
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                pop.append((lst, m, s))
        
        return pop

    def tournament_select(pop, k=3):
        best = None
        for _ in range(k):
            cand = random.choice(pop)
            if best is None or cand[2] > best[2]:
                best = cand
        return best

    def crossover(parent1, parent2):
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

    def local_search(individual, max_iter=30, time_limit=2.0):
        """Hill climbing with delta scoring."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        
        for _ in range(max_iter):
            if time.time() > time_limit:
                break
            improved = False
            for i in range(N):
                old = lst[i]
                old_bit = 1 << old
                best_new = old
                best_s = s
                
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
                    m ^= (1 << old) ^ (1 << best_new)
                    s = best_s
                    improved = True
            
            if not improved:
                break
        
        return (lst, m, s)

    def two_opt_search(individual, max_iter=50, time_limit=3.0):
        """Try swapping pairs of elements."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        
        for _ in range(max_iter):
            if time.time() > time_limit:
                break
            improved = False
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            old_i = lst[i]
            old_j = lst[j]
            
            new_i = random.randint(1, MAX_VAL)
            new_j = random.randint(1, MAX_VAL)
            
            if new_i in s_set or new_j in s_set or new_i == new_j:
                continue
            if new_i == old_j or new_j == old_i:
                continue
            
            # Compute delta for both changes
            m_temp = m ^ (1 << old_i) ^ (1 << old_j)
            m_new = m_temp ^ (1 << new_i) ^ (1 << new_j)
            
            delta = 0
            for d in range(1, 256):
                old_has = (m & (m >> d) & (m >> (2 * d))) != 0
                new_has = (m_new & (m_new >> d) & (m_new >> (2 * d))) != 0
                if old_has != new_has:
                    delta += 1 if new_has else -1
            
            if delta > 0:
                s_set.discard(old_i)
                s_set.discard(old_j)
                s_set.add(new_i)
                s_set.add(new_j)
                lst[i] = new_i
                lst[j] = new_j
                m = m_new
                s += delta
                improved = True
            
            if not improved and random.random() < 0.3:
                # Accept worse move with some probability for diversification
                if delta > -2:
                    s_set.discard(old_i)
                    s_set.discard(old_j)
                    s_set.add(new_i)
                    s_set.add(new_j)
                    lst[i] = new_i
                    lst[j] = new_j
                    m = m_new
                    s += delta
                    improved = True
        
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
    
    # Main GA loop
    generation = 0
    no_improve = 0
    elite_count = 8
    
    while time.time() < deadline:
        generation += 1
        if generation % 15 == 0:
            print(f"Gen {generation}, best={best_score}, time={time.time()-start_time:.1f}s", file=sys.stderr)
        
        new_population = []
        
        # Elitism
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)
            
            r = random.random()
            if r < 0.25:
                child = crossover(p1, p2)
            elif r < 0.6:
                child = mutate(p1, rate=0.12)
            else:
                child = (p1[0][:], p1[1], p1[2])
            
            # Apply local search or two-opt
            if random.random() < 0.5:
                child = local_search(child, max_iter=20, time_limit=time.time() + 1.5)
            else:
                child = two_opt_search(child, max_iter=30, time_limit=time.time() + 2.0)
            
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
        
        # Restart mechanism
        if no_improve > 25:
            for _ in range(10):
                lst = gen_structured_seed()
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve = 0
    
    # Final polish: exhaustive single-swap local search
    if time.time() < deadline - 2:
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
    
    # Second pass: try to improve with different local search
    if time.time() < deadline - 3:
        lst2 = best_list[:]
        m2 = best_mask
        s2 = best_score
        s_set2 = set(lst2)
        
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            i = random.randint(0, N - 1)
            old = lst2[i]
            old_bit = 1 << old
            new = random.randint(1, MAX_VAL)
            if new in s_set2 or new == old:
                continue
            new_bit = 1 << new
            delta = delta_score(m2, old_bit, new_bit)
            if delta > 0:
                s_set2.discard(old)
                s_set2.add(new)
                lst2[i] = new
                m2 ^= old_bit ^ new_bit
                s2 += delta
        
        if s2 > best_score:
            best_score = s2
            best_list = lst2[:]
            best_mask = m2
    
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list

# EVOLVE-BLOCK-END

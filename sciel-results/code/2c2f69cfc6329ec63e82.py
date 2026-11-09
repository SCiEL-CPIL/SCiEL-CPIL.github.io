import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin

    def evaluate_fast(S):
        """Fast evaluation using difference counting."""
        S_set = set(S)
        S_list = sorted(S)
        n = len(S_list)
        # For each pair (i,j), check if a[i]+2*(a[j]-a[i]) = 2*a[j]-a[i] is in S
        # That is: 2*a[j] - a[i] in S_set
        valid_diffs = set()
        for j in range(n):
            aj = S_list[j]
            for i in range(j):
                ai = S_list[i]
                d = aj - ai
                c = aj + d  # = 2*aj - ai
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)

    def evaluate(S):
        return evaluate_fast(S)

    # Known good initial set from parent (score 4.8)
    initial_set = set(list(range(1, 27)) + [28, 29, 30, 31, 32, 48])

    def random_set():
        return set(random.sample(range(1, 513), 32))

    def structured_set():
        base = random.randint(1, 400)
        s = set()
        block_size = random.randint(8, 16)
        for i in range(block_size):
            val = base + i * random.randint(1, 3)
            if 1 <= val <= 512 and val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        return s

    def crossover(s1, s2):
        """Crossover between two sets: take random subset from each."""
        s1_list = list(s1)
        s2_list = list(s2)
        result = set()
        for _ in range(32):
            if random.random() < 0.5:
                result.add(random.choice(s1_list))
            else:
                result.add(random.choice(s2_list))
        while len(result) < 32:
            val = random.randint(1, 512)
            if val not in result:
                result.add(val)
        return result

    def mutate(s, num_mutations=1):
        """Apply random mutations to a set."""
        s = set(s)
        for _ in range(num_mutations):
            if random.random() < 0.7:
                # Single swap
                x = random.choice(list(s))
                y = random.randint(1, 512)
                if y not in s:
                    s.remove(x)
                    s.add(y)
            else:
                # Double swap
                x1 = random.choice(list(s))
                x2 = random.choice(list(s))
                if x1 != x2:
                    y1 = random.randint(1, 512)
                    y2 = random.randint(1, 512)
                    if y1 not in s and y2 not in s and y1 != y2:
                        s.remove(x1)
                        s.remove(x2)
                        s.add(y1)
                        s.add(y2)
        return s

    best_set = None
    best_score = 0

    candidates = [
        initial_set,
        set(range(1, 33)),
        random_set(),
        random_set(),
        structured_set(),
        structured_set(),
    ]

    # Initialize population for GA
    population = []
    for c in candidates:
        score = evaluate(c)
        population.append((score, c.copy()))
        if score > best_score:
            best_score = score
            best_set = c.copy()

    # Add more random/structured candidates to population
    pop_size = 12
    while len(population) < pop_size:
        c = random_set() if random.random() < 0.5 else structured_set()
        score = evaluate(c)
        population.append((score, c.copy()))
        if score > best_score:
            best_score = score
            best_set = c.copy()

    # Sort by score descending
    population.sort(key=lambda x: -x[0])

    all_elements = list(range(1, 513))

    # Main loop: combine GA with local search
    generation = 0
    stall_count = 0

    while time.time() - start_time < timeout:
        if time.time() - start_time > timeout - 5.0:
            break

        # Local search on the best individual
        current_set = population[0][1].copy()
        current_score = population[0][0]

        local_improvements = 0
        steps = 0
        max_local_steps = 2000

        while steps < max_local_steps:
            if time.time() - start_time > timeout - 5.0:
                break

            move_type = random.random()
            new_set = set(current_set)

            if move_type < 0.6:
                x_to_remove = random.choice(list(current_set))
                y_to_add = random.choice(all_elements)
                if y_to_add in current_set:
                    steps += 1
                    continue
                new_set.remove(x_to_remove)
                new_set.add(y_to_add)
            else:
                x1 = random.choice(list(current_set))
                x2 = random.choice(list(current_set))
                if x1 == x2:
                    steps += 1
                    continue
                y1 = random.choice(all_elements)
                y2 = random.choice(all_elements)
                if y1 in current_set or y2 in current_set or y1 == y2:
                    steps += 1
                    continue
                new_set.remove(x1)
                new_set.remove(x2)
                new_set.add(y1)
                new_set.add(y2)

            new_score = evaluate(new_set)

            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                local_improvements += 1
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set.copy()
                # Update population[0]
                population[0] = (current_score, current_set.copy())
            else:
                # Accept with small probability (simulated annealing flavor)
                if random.random() < 0.05:
                    current_set = new_set
                    current_score = new_score

            steps += 1

        # Update population[0] with best local search result
        population[0] = (evaluate(population[0][1]), population[0][1])

        # Generate next generation using GA operators
        new_population = [population[0]]  # Elitism: keep best

        # Crossover + mutation
        while len(new_population) < pop_size - 1:
            if random.random() < 0.6:
                # Tournament selection
                p1 = random.choice(population[:max(2, pop_size//2)])
                p2 = random.choice(population[:max(2, pop_size//2)])
                child = crossover(p1[1], p2[1])
                child = mutate(child, random.randint(1, 3))
            else:
                # Mutation of random parent
                parent = random.choice(population)
                child = mutate(parent[1], random.randint(1, 4))

            score = evaluate(child)
            new_population.append((score, child.copy()))
            if score > best_score:
                best_score = score
                best_set = child.copy()

        # Add some fresh random/structured individuals
        while len(new_population) < pop_size:
            c = random_set() if random.random() < 0.5 else structured_set()
            score = evaluate(c)
            new_population.append((score, c.copy()))
            if score > best_score:
                best_score = score
                best_set = c.copy()

        # Sort and keep best
        new_population.sort(key=lambda x: -x[0])
        population = new_population[:pop_size]

        generation += 1

        # Track stalls
        if population[0][0] <= best_score - 1:
            stall_count += 1
        else:
            stall_count = 0

        # If stalled, try a big perturbation
        if stall_count > 5:
            for i in range(pop_size):
                if random.random() < 0.3:
                    population[i] = (evaluate(mutate(population[i][1], random.randint(3, 8))), population[i][1])
            population.sort(key=lambda x: -x[0])
            stall_count = 0

    return list(best_set)

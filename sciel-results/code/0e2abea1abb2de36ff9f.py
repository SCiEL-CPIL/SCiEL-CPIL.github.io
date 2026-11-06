import time
import random

def _count_progression_diffs(S):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    S_list = list(S)
    diffs = set()
    for a in S_list:
        for b in S_list:
            d = b - a
            if d > 0:
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    N = 32
    MAXV = 512
    POP_SIZE = 40
    ELITE = 4

    def _score(S):
        return _count_progression_diffs(S)

    def _crossover(p1, p2):
        """Uniform crossover: each element from p1 or p2."""
        s1 = set(p1)
        s2 = set(p2)
        child = set()
        for v in range(1, MAXV + 1):
            if v in s1 or v in s2:
                if v in s1:
                    child.add(v)
                elif v in s2:
                    if random.random() < 0.5:
                        child.add(v)
        # Ensure exactly N elements
        if len(child) > N:
            # Remove excess randomly
            excess = list(child)
            random.shuffle(excess)
            for v in excess[:len(child) - N]:
                child.remove(v)
        elif len(child) < N:
            # Add random elements
            while len(child) < N:
                v = random.randint(1, MAXV)
                child.add(v)
        return child

    def _mutate(S, rate=0.15):
        S = set(S)
        num_swaps = max(1, int(N * rate))
        for _ in range(num_swaps):
            elem = random.choice(list(S))
            cand = random.randint(1, MAXV)
            if cand not in S:
                S.remove(elem)
                S.add(cand)
        return S

    # Initialize population
    random.seed(777)
    population = []
    # Add some structured individuals
    pop_structures = [
        set(range(1, N + 1)),  # consecutive
        set(range(1, MAXV + 1, MAXV // N)),  # spread
        set(random.sample(range(1, 100), N)),  # small range
        set(random.sample(range(400, 513), N)),  # large range
        set(random.sample(range(1, MAXV + 1), N)),  # random
    ]
    for s in pop_structures:
        while len(s) < N:
            s.add(random.randint(1, MAXV))
        population.append(list(s)[:N])
    
    while len(population) < POP_SIZE:
        population.append(list(random.sample(range(1, MAXV + 1), N)))

    # Evaluate initial population
    scores = [_score(set(ind)) for ind in population]
    best_idx = max(range(POP_SIZE), key=lambda i: scores[i])
    best_set = set(population[best_idx])
    best_score = scores[best_idx]

    gen = 0
    while time.time() < deadline - 5.0:
        gen += 1
        # Sort by score
        order = sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)
        
        # Elitism
        new_pop = [list(set(population[order[i]])) for i in range(ELITE)]
        new_scores = [scores[order[i]] for i in range(ELITE)]
        
        # Generate offspring
        while len(new_pop) < POP_SIZE:
            # Tournament selection
            def tournament(k=3):
                contenders = random.sample(range(POP_SIZE), k)
                return max(contenders, key=lambda i: scores[i])
            
            p1 = tournament()
            p2 = tournament()
            
            if random.random() < 0.7:
                child = _crossover(population[p1], population[p2])
            else:
                child = set(population[p1])
            child = _mutate(child)
            child = list(child)
            new_pop.append(child)
            new_scores.append(_score(set(child)))
        
        population = new_pop
        scores = new_scores
        
        # Update best
        for i in range(POP_SIZE):
            if scores[i] > best_score:
                best_score = scores[i]
                best_set = set(population[i])
        
        # Check time
        if time.time() > deadline - 10.0:
            break

    return list(best_set)

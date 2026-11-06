"""Combined program: fast scoring + structured init + targeted local search + genetic algorithm."""
import time
import random

def _score(S):
    """Number of distinct d > 0 such that a, a+d, a+2d in S."""
    S_set = set(S)
    valid_ds = set()
    elems = list(S)
    n = len(elems)
    for i in range(n):
        x = elems[i]
        for j in range(i + 1, n):
            y = elems[j]
            d = y - x
            if (x - d) in S_set or (y + d) in S_set:
                valid_ds.add(d)
    return len(valid_ds)

def _candidate_progressions(S):
    """Return a set of candidate new elements that would create a 3-AP.
    For each pair (x, y) in S with d = y-x, candidates are x-d and y+d (if in range)."""
    S_set = set(S)
    cands = set()
    elems = list(S)
    n = len(elems)
    for i in range(n):
        x = elems[i]
        for j in range(i + 1, n):
            y = elems[j]
            d = y - x
            c1 = x - d
            c2 = y + d
            if 1 <= c1 <= 512 and c1 not in S_set:
                cands.add(c1)
            if 1 <= c2 <= 512 and c2 not in S_set:
                cands.add(c2)
    return cands

def _local_search(best_set, deadline, rng, use_targeted=True):
    """Local search with optional targeted moves."""
    best_score = _score(best_set)
    while time.time() < deadline:
        improved = False
        # Try targeted moves first if available
        if use_targeted:
            cands = _candidate_progressions(best_set)
            if cands:
                # Shuffle candidates
                cand_list = list(cands)
                rng.shuffle(cand_list)
                for new_elem in cand_list:
                    if time.time() >= deadline:
                        break
                    # Try replacing each element
                    for to_remove in list(best_set):
                        new_set = set(best_set)
                        new_set.remove(to_remove)
                        new_set.add(new_elem)
                        new_score = _score(new_set)
                        if new_score > best_score:
                            best_set = new_set
                            best_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        # Random moves
        for _ in range(30):
            if time.time() >= deadline:
                break
            new_set = set(best_set)
            to_remove = rng.choice(list(new_set))
            new_set.remove(to_remove)
            available = [x for x in range(1, 513) if x not in new_set]
            new_elem = rng.choice(available)
            new_set.add(new_elem)
            new_score = _score(new_set)
            if new_score > best_score:
                best_set = new_set
                best_score = new_score
                improved = True
        if not improved:
            # Stagnation: try a few random restarts
            for _ in range(5):
                if time.time() >= deadline:
                    break
                new_set = set(best_set)
                for _ in range(3):
                    to_remove = rng.choice(list(new_set))
                    new_set.remove(to_remove)
                    available = [x for x in range(1, 513) if x not in new_set]
                    new_elem = rng.choice(available)
                    new_set.add(new_elem)
                new_score = _score(new_set)
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    break
    return best_set, best_score

def _genetic_algorithm(best_set, best_score, deadline, rng, pop_size=15):
    """Genetic algorithm with crossover and mutation."""
    n = 32
    # Initialize population with best set and variations
    population = [(set(best_set), best_score)]
    for _ in range(pop_size - 1):
        if rng.random() < 0.4:
            # Structured
            base = rng.randint(1, 50)
            d = rng.randint(1, 15)
            arr = [base + i * d for i in range(n)]
            arr = [x for x in arr if 1 <= x <= 512]
            while len(arr) < n:
                x = rng.randint(1, 512)
                if x not in arr:
                    arr.append(x)
            arr = arr[:n]
        else:
            arr = rng.sample(range(1, 513), n)
        s = set(arr)
        sc = _score(s)
        population.append((s, sc))
    
    best_in_pop = max(population, key=lambda x: x[1])
    if best_in_pop[1] > best_score:
        best_score = best_in_pop[1]
        best_set = best_in_pop[0]
    
    while time.time() < deadline:
        new_population = []
        for _ in range(pop_size):
            if time.time() >= deadline:
                break
            if rng.random() < 0.15:
                # Random new individual
                if rng.random() < 0.3:
                    base = rng.randint(1, 50)
                    d = rng.randint(1, 15)
                    arr = [base + i * d for i in range(n)]
                    arr = [x for x in arr if 1 <= x <= 512]
                    while len(arr) < n:
                        x = rng.randint(1, 512)
                        if x not in arr:
                            arr.append(x)
                    arr = arr[:n]
                else:
                    arr = rng.sample(range(1, 513), n)
            else:
                # Crossover from top
                top = sorted(population, key=lambda x: -x[1])[:5]
                p1, p2 = random.sample(top, 2)
                arr1 = list(p1[0])
                arr2 = list(p2[0])
                mask = [rng.random() < 0.5 for _ in range(n)]
                arr = [arr1[i] if mask[i] else arr2[i] for i in range(n)]
                # Ensure distinct and in range
                seen = set()
                new_arr = []
                for x in arr:
                    if x not in seen and 1 <= x <= 512:
                        seen.add(x)
                        new_arr.append(x)
                while len(new_arr) < n:
                    x = rng.randint(1, 512)
                    if x not in seen:
                        seen.add(x)
                        new_arr.append(x)
                arr = new_arr[:n]
            s = set(arr)
            sc = _score(s)
            new_population.append((s, sc))
        if not new_population:
            break
        population = new_population
        best_in_pop = max(population, key=lambda x: x[1])
        if best_in_pop[1] > best_score:
            best_score = best_in_pop[1]
            best_set = best_in_pop[0]
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    rng = random.Random(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Arithmetic progressions with small d
    for d in range(1, 17):
        arr = [1 + i * d for i in range(32)]
        if max(arr) <= 512:
            candidates.append(set(arr))
    
    # Structured sets
    for _ in range(40):
        base = rng.randint(1, 30)
        d = rng.randint(1, 15)
        arr = [base + i * d for i in range(32)]
        arr = [x for x in arr if 1 <= x <= 512]
        while len(arr) < 32:
            x = rng.randint(1, 512)
            if x not in arr:
                arr.append(x)
        candidates.append(set(arr[:32]))
    
    # Random sets
    for _ in range(40):
        candidates.append(set(rng.sample(range(1, 513), 32)))
    
    # Evaluate candidates
    best_set = max(candidates, key=lambda x: _score(x))
    best_score = _score(best_set)
    
    # Phase 1: Local search with targeted moves
    if time.time() < deadline - 60:
        phase1_deadline = min(deadline - 60, start_time + 40)
        best_set, best_score = _local_search(best_set, phase1_deadline, rng, use_targeted=True)
    
    # Phase 2: Genetic algorithm
    if time.time() < deadline - 20:
        ga_deadline = min(deadline - 20, start_time + 110)
        best_set, best_score = _genetic_algorithm(best_set, best_score, ga_deadline, rng, pop_size=15)
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2:
        best_set, best_score = _local_search(best_set, deadline - 2, rng, use_targeted=True)
    
    return list(best_set)

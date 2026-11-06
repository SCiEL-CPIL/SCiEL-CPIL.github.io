import time
import random

def _score(s):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in s."""
    diffs = set()
    elements = sorted(s)
    n = len(elements)
    s_set = set(s)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def _local_search(s, max_steps=10, time_limit=0.05):
    """
    Perform a randomized local search on the set s.
    """
    current_set = set(s)
    current_score = _score(current_set)
    start_time = time.time()
    
    for _ in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        elem_out = random.choice(list(current_set))
        elem_in = random.randint(1, 512)
        
        if elem_in in current_set:
            continue
            
        new_set = set(current_set)
        new_set.remove(elem_out)
        new_set.add(elem_in)
        
        new_score = _score(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
        elif new_score == current_score and random.random() < 0.1:
            current_set = new_set
            current_score = new_score
            
    return current_set, current_score

def _generate_initial_set():
    """Generate a diverse initial set."""
    r = random.random()
    if r < 0.3:
        return set(random.sample(range(1, 513), 32))
    elif r < 0.5:
        start = random.randint(1, 400)
        range_size = random.randint(40, 100)
        return set(random.sample(range(start, min(513, start + range_size)), 32))
    elif r < 0.7:
        d = random.choice([2, 3, 4, 5, 6, 8, 10])
        max_start = 512 - 31 * d
        if max_start >= 1:
            a = random.randint(1, max_start)
            return set(a + i * d for i in range(32))
        else:
            return set(random.sample(range(1, 513), 32))
    elif r < 0.9:
        s1_start = random.randint(1, 200)
        s2_start = random.randint(300, 450)
        s1 = set(random.sample(range(s1_start, s1_start + 50), 16))
        s2 = set(random.sample(range(s2_start, s2_start + 50), 16))
        s = s1 | s2
        if len(s) == 32:
            return s
        else:
            return set(random.sample(range(1, 513), 32))
    else:
        return set(random.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_set = None
    best_score = 0
    
    pop_size = 30
    population = []
    
    for _ in range(pop_size):
        s = _generate_initial_set()
        score = _score(s)
        population.append((s, score))
        if score > best_score:
            best_score = score
            best_set = s

    population.sort(key=lambda x: x[1], reverse=True)
    
    last_improvement_time = start_time
    stagnation_limit = 10.0
    
    while time.time() < deadline:
        if time.time() - last_improvement_time > stagnation_limit:
            new_pop = []
            for _ in range(pop_size // 2):
                s_best = set(best_set)
                for _ in range(5):
                    eo = random.choice(list(s_best))
                    ei = random.randint(1, 512)
                    if ei not in s_best:
                        s_best.remove(eo)
                        s_best.add(ei)
                score = _score(s_best)
                new_pop.append((s_best, score))
                if score > best_score:
                    best_score = score
                    best_set = s_best
                    last_improvement_time = time.time()
            
            for _ in range(pop_size // 2):
                s = _generate_initial_set()
                score = _score(s)
                new_pop.append((s, score))
                if score > best_score:
                    best_score = score
                    best_set = s
                    last_improvement_time = time.time()
            
            population = new_pop
            population.sort(key=lambda x: x[1], reverse=True)
            continue
        
        new_population = []
        for i in range(5):
            new_population.append(population[i])
            
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
                
            p1_idx = random.randint(0, min(10, len(population)-1))
            p2_idx = random.randint(0, min(10, len(population)-1))
            
            p1 = population[p1_idx][0]
            p2 = population[p2_idx][0]
            
            p1_sorted = sorted(p1)
            p2_sorted = sorted(p2)
            
            cut = random.randint(1, 512)
            
            child = set()
            for x in p1_sorted:
                if x < cut:
                    child.add(x)
            for x in p2_sorted:
                if x >= cut:
                    child.add(x)
            
            if len(child) < 32:
                while len(child) < 32:
                    x = random.randint(1, 512)
                    child.add(x)
            elif len(child) > 32:
                child_list = list(child)
                random.shuffle(child_list)
                child = set(child_list[:32])
            
            num_swaps = random.randint(1, 3)
            for _ in range(num_swaps):
                if len(child) != 32:
                    break
                eo = random.choice(list(child))
                ei = random.randint(1, 512)
                if ei not in child:
                    child.remove(eo)
                    child.add(ei)
            
            child, score = _local_search(child, max_steps=10, time_limit=0.05)
            
            new_population.append((child, score))
            
            if score > best_score:
                best_score = score
                best_set = child
                last_improvement_time = time.time()
                
        population = new_population
        population.sort(key=lambda x: x[1], reverse=True)

    return list(best_set)

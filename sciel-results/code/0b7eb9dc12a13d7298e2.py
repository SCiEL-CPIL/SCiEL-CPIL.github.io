"""Rewritten program: search for small finite sets U of nonnegative ints containing 0
to maximize the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""

import math
import random
import time
from itertools import combinations, permutations

# EVOLVE-BLOCK-START

def _score_of(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / log_denom
    return score


def _local_search(U, max_steps=5000, time_limit=10.0):
    """Local search: try adding/removing/replacing elements to improve score."""
    U_set = set(U)
    best_score = _score_of(U)
    best_set = set(U_set)
    start = time.time()
    
    # Candidate pool: numbers up to some bound
    max_u = max(U_set) if U_set else 0
    bound = max(2 * max_u + 10, 50)
    
    step = 0
    while step < max_steps:
        if time.time() - start > time_limit:
            break
        step += 1
        
        current = list(best_set)
        current_score = best_score
        
        # Try adding a random element
        if len(best_set) < 15:
            candidates = [x for x in range(1, bound + 1) if x not in best_set]
            if candidates:
                x = random.choice(candidates)
                new_set = best_set | {x}
                s = _score_of(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = new_set
                    bound = max(bound, 2 * max(new_set) + 10)
                    continue
        
        # Try removing an element (if size > 3)
        if len(best_set) > 3:
            x = random.choice(list(best_set - {0}))  # never remove 0
            new_set = best_set - {x}
            s = _score_of(list(new_set))
            if s > best_score:
                best_score = s
                best_set = new_set
                continue
        
        # Try replacing an element
        if len(best_set) > 3:
            x = random.choice(list(best_set - {0}))
            candidates = [y for y in range(1, bound + 1) if y not in best_set]
            if candidates:
                y = random.choice(candidates)
                new_set = (best_set - {x}) | {y}
                s = _score_of(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = new_set
                    bound = max(bound, 2 * max(new_set) + 10)
                    continue
    
    return sorted(best_set), best_score


def _genetic_search(num_generations=300, pop_size=40, time_limit=80.0):
    """Genetic algorithm to search for good sets."""
    start = time.time()
    
    # Initialize population with diverse small sets
    pop = []
    # Seed with known good structures
    seeds = [
        [0, 1, 2, 4, 8],
        [0, 1, 3, 7, 15],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 4, 9],
        [0, 1, 2, 3, 7, 14],
        [0, 1, 4, 5, 10, 20],
        [0, 1, 2, 6, 7, 14],
        [0, 1, 3, 6, 12, 13],
    ]
    for s in seeds:
        pop.append(set(s))
    
    # Add random small sets
    while len(pop) < pop_size:
        size = random.randint(3, 8)
        elems = random.sample(range(1, 40), size - 1)
        pop.append({0} | set(elems))
    
    best_set = max(pop, key=lambda s: _score_of(list(s)))
    best_score = _score_of(list(best_set))
    
    for gen in range(num_generations):
        if time.time() - start > time_limit:
            break
        
        # Evaluate all
        scored = [(s, _score_of(list(s))) for s in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if scored[0][1] > best_score:
            best_score = scored[0][1]
            best_set = scored[0][0]
        
        # Elitism: keep top 20%
        elite_count = max(2, pop_size // 5)
        new_pop = [s for s, sc in scored[:elite_count]]
        
        # Fill rest with children
        while len(new_pop) < pop_size:
            # Selection
            p1 = random.choices(scored, weights=[sc for _, sc in scored])[0][0]
            p2 = random.choices(scored, weights=[sc for _, sc in scored])[0][0]
            
            # Crossover: merge and truncate
            child = set(p1) | set(p2)
            if len(child) > 10:
                # Keep 0 and random others
                others = list(child - {0})
                random.shuffle(others)
                child = {0} | set(others[:random.randint(4, 8)])
            elif len(child) < 3:
                # Add random elements
                while len(child) < 3:
                    child.add(random.randint(1, 30))
            
            # Mutation
            if random.random() < 0.7:
                if random.random() < 0.4 and len(child) > 3:
                    x = random.choice(list(child - {0}))
                    child.discard(x)
                else:
                    child.add(random.randint(1, 50))
            
            # Ensure 0 is present
            child.add(0)
            new_pop.append(child)
        
        pop = new_pop
    
    return sorted(best_set), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    best_set = {0, 1, 3}
    best_score = _score_of([0, 1, 3])
    
    # Phase 1: Local search from multiple random starts
    for i in range(20):
        if time.time() > deadline - 40:
            break
        size = random.randint(3, 8)
        elems = random.sample(range(1, 30), size - 1)
        U = {0} | set(elems)
        found_set, found_score = _local_search(list(U), max_steps=2000, time_limit=3.0)
        if found_score > best_score:
            best_score = found_score
            best_set = set(found_set)
    
    # Phase 2: Genetic search
    if time.time() < deadline - 20:
        found_set, found_score = _genetic_search(
            num_generations=200, pop_size=40, time_limit=50.0
        )
        if found_score > best_score:
            best_score = found_score
            best_set = set(found_set)
    
    # Phase 3: Final local search polish on the best found
    if time.time() < deadline - 5:
        found_set, found_score = _local_search(
            list(best_set), max_steps=5000, time_limit=10.0
        )
        if found_score > best_score:
            best_score = found_score
            best_set = set(found_set)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END

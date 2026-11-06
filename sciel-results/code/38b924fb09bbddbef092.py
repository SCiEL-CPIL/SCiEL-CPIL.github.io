"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _score_set(arr):
    """Count distinct d > 0 such that a, a+d, a+2d all in set."""
    s = set(arr)
    n = len(arr)
    distinct = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = arr[j] - arr[i]
            if d <= 0:
                continue
            if arr[i] + 2 * d in s:
                distinct.add(d)
    return len(distinct)


def _score_fast(arr):
    """Faster scoring using sorted array and set lookup."""
    s = set(arr)
    distinct = set()
    for a in arr:
        for d in range(1, 513 - a):
            if a + d in s and a + 2 * d in s:
                distinct.add(d)
                # Can't break early since we need all d
    return len(distinct)


def _score_sorted(arr):
    """Score using sorted array - check all pairs for 3rd element."""
    s = set(arr)
    distinct = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d = arr[j] - arr[i]
            if arr[i] + 2 * d in s:
                distinct.add(d)
    return len(distinct)


def _local_search(arr, max_iters=20000, rng=None):
    """Local search: try add, remove, swap operations."""
    if rng is None:
        rng = random.Random()
    current = sorted(arr)
    current_score = _score_sorted(current)
    best = list(current)
    best_score = current_score
    
    for it in range(max_iters):
        op = rng.random()
        new = list(current)
        
        if op < 0.35:
            # Add a random element
            candidates = [x for x in range(1, 513) if x not in set(current)]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new.append(x)
        elif op < 0.65:
            # Remove a random element
            if len(new) <= 32:
                continue
            idx = rng.randrange(len(new))
            new.pop(idx)
        else:
            # Swap: remove one, add one
            if len(new) != 32:
                continue
            idx = rng.randrange(len(new))
            removed = new.pop(idx)
            candidates = [x for x in range(1, 513) if x not in set(new)]
            if not candidates:
                new.insert(idx, removed)
                continue
            x = rng.choice(candidates)
            new.append(x)
        
        if len(new) != 32:
            continue
        
        new_score = _score_sorted(new)
        if new_score > current_score:
            current = sorted(new)
            current_score = new_score
            if current_score > best_score:
                best = list(current)
                best_score = current_score
    
    return best, best_score


def _hybrid_genome(rng, n_dense=14, n_spread=18):
    """Generate a hybrid genome: dense block + spread points."""
    # Dense block: consecutive or near-consecutive numbers
    start = rng.randint(1, 512 - n_dense + 1)
    dense = list(range(start, start + n_dense))
    
    # Spread points: random distinct values not in dense
    dense_set = set(dense)
    spread = []
    while len(spread) < n_spread:
        x = rng.randint(1, 512)
        if x not in dense_set and x not in spread:
            spread.append(x)
            dense_set.add(x)
    
    genome = dense + spread
    return genome


def _random_genome(rng, n=32):
    """Fully random genome."""
    return rng.sample(range(1, 513), n)


def _genetic_algorithm(seed=42, time_limit=140):
    """Genetic algorithm to find optimal 32-element set."""
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize population
    pop_size = 40
    population = []
    
    # Add some structured individuals
    for i in range(10):
        nd = rng.randint(10, 20)
        genome = _hybrid_genome(rng, n_dense=nd, n_spread=32 - nd)
        population.append(genome)
    
    for i in range(pop_size - 10):
        genome = _random_genome(rng)
        population.append(genome)
    
    # Evaluate initial population
    scores = [_score_sorted(p) for p in population]
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_genome = list(population[best_idx])
    best_score = scores[best_idx]
    
    # Run local search on best few
    for i in range(min(5, pop_size)):
        idx = sorted(range(pop_size), key=lambda j: scores[j], reverse=True)[i]
        improved, imp_score = _local_search(population[idx], max_iters=3000, rng=rng)
        if imp_score > best_score:
            best_score = imp_score
            best_genome = improved
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining < 10:
        return best_genome, best_score
    
    # Iterative local search with random restarts
    gen = 0
    while time.time() - start_time < time_limit - 15:
        # Pick a random individual and improve it
        idx = rng.randrange(pop_size)
        improved, imp_score = _local_search(population[idx], max_iters=5000, rng=rng)
        
        if imp_score > best_score:
            best_score = imp_score
            best_genome = improved
        
        # Replace worst individual
        worst_idx = min(range(pop_size), key=lambda j: scores[j])
        population[worst_idx] = list(improved)
        scores[worst_idx] = imp_score
        
        # Occasionally add a fresh random/hybrid individual
        if gen % 5 == 0:
            if rng.random() < 0.5:
                nd = rng.randint(10, 20)
                new_g = _hybrid_genome(rng, n_dense=nd, n_spread=32 - nd)
            else:
                new_g = _random_genome(rng)
            # Replace another random individual
            replace_idx = rng.randrange(pop_size)
            population[replace_idx] = new_g
            scores[replace_idx] = _score_sorted(new_g)
        
        gen += 1
    
    return best_genome, best_score


def construct():
    """Find 32 distinct integers in 1..512 maximizing distinct 3-term AP differences."""
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill
    
    # Try multiple seeds with decreasing time allocation
    best_genome = None
    best_score = -1
    
    seeds = [42, 123, 999, 7, 2024, 314, 271, 555, 1337, 888]
    time_per_seed = time_limit / len(seeds)
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break
        
        remaining_for_this = min(time_per_seed, time_limit - elapsed - 3)
        if remaining_for_this < 3:
            break
        
        genome, score = _genetic_algorithm(seed=seed, time_limit=remaining_for_this)
        
        if score > best_score:
            best_score = score
            best_genome = genome
    
    # Final polish: local search on best found
    if best_genome is None:
        # Fallback: dense set
        best_genome = list(range(1, 33))
        best_score = _score_sorted(best_genome)
    else:
        elapsed = time.time() - start_time
        if time_limit - elapsed > 5:
            improved, imp_score = _local_search(best_genome, max_iters=10000, rng=random.Random(0))
            if imp_score > best_score:
                best_score = imp_score
                best_genome = improved
    
    # Ensure validity
    assert len(best_genome) == 32
    assert len(set(best_genome)) == 32
    assert all(1 <= x <= 512 for x in best_genome)
    
    return list(best_genome)

# EVOLVE-BLOCK-END

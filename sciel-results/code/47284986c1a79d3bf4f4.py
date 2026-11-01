"""Covering design C(24,7,3): genetic algorithm with greedy initialization and local search."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _fast_greedy(v, k, t, rng, time_limit, all_tsets, tset_index):
    """
    Fast greedy covering using precomputed tset index for speed.
    """
    deadline = time.time() + time_limit
    n_tsets = len(all_tsets)
    covered = np.zeros(n_tsets, dtype=np.int8)
    blocks = []
    
    while time.time() < deadline:
        # Find uncovered t-sets
        uncovered_mask = covered == 0
        if not np.any(uncovered_mask):
            break
        
        # Pick a random uncovered t-set to start the block
        uncovered_indices = np.where(uncovered_mask)[0]
        start_idx = rng.choice(uncovered_indices)
        start_tset = all_tsets[start_idx]
        block = set(start_tset)
        
        while len(block) < k:
            # Compute gain for each candidate point
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tset = tuple(sorted(rest + (x,)))
                    if new_tset in tset_index:
                        idx = tset_index[new_tset]
                        if covered[idx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.add(best_point)
        
        block_list = sorted(block)
        blocks.append(block_list)
        # Mark covered t-sets
        for ts in itertools.combinations(block_list, t):
            if ts in tset_index:
                covered[tset_index[ts]] = 1
    
    return blocks


def _compute_coverage(blocks, v, k, t, all_tsets, tset_index):
    """Compute coverage vector for a set of blocks."""
    n_tsets = len(all_tsets)
    covered = np.zeros(n_tsets, dtype=np.int8)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            if ts in tset_index:
                covered[tset_index[ts]] = 1
    return covered


def _fitness(blocks, v, k, t, all_tsets, tset_index):
    """
    Fitness: lower is better.
    Uses sum of inverse coverage counts for uncovered triples + block count penalty.
    """
    covered = _compute_coverage(blocks, v, k, t, all_tsets, tset_index)
    n_uncovered = np.sum(covered == 0)
    # Primary: number of uncovered triples
    # Secondary: number of blocks
    return (n_uncovered, len(blocks))


def _crossover(parent1, parent2, v, k, t, rng, all_tsets, tset_index):
    """
    Crossover: take blocks from both parents, then fill gaps greedily.
    """
    child_blocks = []
    used_blocks = set()
    
    # Take a random mix of blocks from parents
    p1_blocks = list(parent1)
    p2_blocks = list(parent2)
    rng.shuffle(p1_blocks)
    rng.shuffle(p2_blocks)
    
    # Start with a subset from each parent
    n_from_p1 = rng.randint(len(p1_blocks) // 3, 2 * len(p1_blocks) // 3)
    n_from_p2 = max(0, len(parent1) - n_from_p1)
    
    for b in p1_blocks[:n_from_p1]:
        key = tuple(b)
        if key not in used_blocks:
            child_blocks.append(b)
            used_blocks.add(key)
    
    for b in p2_blocks[:n_from_p2]:
        key = tuple(b)
        if key not in used_blocks:
            child_blocks.append(b)
            used_blocks.add(key)
    
    # Fill remaining gaps greedily
    covered = _compute_coverage(child_blocks, v, k, t, all_tsets, tset_index)
    uncovered_mask = covered == 0
    
    while np.any(uncovered_mask) and time.time() < _crossover_deadline:
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = rng.choice(uncovered_indices)
        start_tset = all_tsets[start_idx]
        block = set(start_tset)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tset = tuple(sorted(rest + (x,)))
                    if new_tset in tset_index:
                        idx = tset_index[new_tset]
                        if covered[idx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.add(best_point)
        
        block_list = sorted(block)
        key = tuple(block_list)
        if key not in used_blocks:
            child_blocks.append(block_list)
            used_blocks.add(key)
            for ts in itertools.combinations(block_list, t):
                if ts in tset_index:
                    covered[tset_index[ts]] = 1
        
        uncovered_mask = covered == 0
    
    return child_blocks


def _mutate(blocks, v, k, t, rng, all_tsets, tset_index, time_limit):
    """
    Mutation: try to replace or remove blocks to improve coverage.
    """
    deadline = time.time() + time_limit
    current = list(blocks)
    
    while time.time() < deadline:
        improved = False
        n = len(current)
        
        # Try removing a redundant block
        for i in range(n):
            if time.time() >= deadline:
                break
            trial = current[:i] + current[i+1:]
            covered = _compute_coverage(trial, v, k, t, all_tsets, tset_index)
            if np.all(covered >= 1):
                current = trial
                improved = True
                break
        
        if not improved:
            # Try replacing a block with a better one
            for i in range(n):
                if time.time() >= deadline:
                    break
                
                # Find t-sets only covered by block i
                covered_without_i = _compute_coverage(current[:i] + current[i+1:], v, k, t, all_tsets, tset_index)
                unique_to_i = set()
                for ts in itertools.combinations(current[i], t):
                    if ts in tset_index:
                        idx = tset_index[ts]
                        if covered_without_i[idx] == 0:
                            unique_to_i.add(ts)
                
                if not unique_to_i:
                    continue
                
                # Greedily build a new block covering unique_to_i
                must_cover = set(unique_to_i)
                new_block = set()
                
                if must_cover:
                    start_ts = next(iter(must_cover))
                    new_block.add(start_ts[0])
                
                while len(new_block) < k and must_cover:
                    best_point = -1
                    best_gain = -1
                    for x in range(v):
                        if x in new_block:
                            continue
                        gain = 0
                        for ts in must_cover:
                            if x in ts:
                                remaining = set(ts) - new_block - {x}
                                if not remaining:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point == -1:
                        break
                    new_block.add(best_point)
                
                new_block_list = sorted(new_block)
                existing_set = set(tuple(b) for b in current)
                if tuple(new_block_list) not in existing_set:
                    trial = current[:i] + [new_block_list] + current[i+1:]
                    covered = _compute_coverage(trial, v, k, t, all_tsets, tset_index)
                    if np.all(covered >= 1):
                        current = trial
                        improved = True
                        break
        
        if not improved:
            break
    
    return current


def _remove_redundant_blocks(blocks, v, k, t, all_tsets, tset_index):
    """Remove blocks that don't cover any unique t-set."""
    n_tsets = len(all_tsets)
    n_blocks = len(blocks)
    
    # Build coverage matrix: block b covers t-set s
    cover = np.zeros((n_blocks, n_tsets), dtype=np.int8)
    for b, block in enumerate(blocks):
        for ts in itertools.combinations(block, t):
            if ts in tset_index:
                cover[b, tset_index[ts]] = 1
    
    # For each block, check if removing it leaves all t-sets covered
    result = []
    for b in range(n_blocks):
        # Temporarily remove block b
        covered_without = np.sum(cover, axis=0) - cover[b]
        if np.all(covered_without >= 1):
            # Block b is redundant, skip it
            continue
        result.append(blocks[b])
    
    return result


def _local_search(blocks, v, k, t, rng, time_limit, all_tsets, tset_index):
    """Local search: try to replace blocks to reduce count."""
    deadline = time.time() + time_limit
    
    current = list(blocks)
    
    while time.time() < deadline:
        improved = False
        n = len(current)
        
        # Try removing each block and checking validity
        for i in range(n):
            if time.time() >= deadline:
                break
            trial = current[:i] + current[i+1:]
            covered = _compute_coverage(trial, v, k, t, all_tsets, tset_index)
            if np.all(covered >= 1):
                current = trial
                improved = True
                break
        
        if not improved:
            # Try replacing a block with a better one
            for i in range(n):
                if time.time() >= deadline:
                    break
                
                # Find t-sets only covered by block i
                covered_without_i = _compute_coverage(current[:i] + current[i+1:], v, k, t, all_tsets, tset_index)
                unique_to_i = set()
                for ts in itertools.combinations(current[i], t):
                    if ts in tset_index:
                        idx = tset_index[ts]
                        if covered_without_i[idx] == 0:
                            unique_to_i.add(ts)
                
                if not unique_to_i:
                    continue
                
                # Try to find a new block that covers all unique_to_i t-sets
                # and is different from existing blocks
                existing_set = set(tuple(b) for b in current)
                
                # Greedily build a block that covers unique_to_i
                must_cover = set(unique_to_i)
                new_block = set()
                
                # Start with a point from a unique t-set
                if must_cover:
                    start_ts = next(iter(must_cover))
                    new_block.add(start_ts[0])
                
                while len(new_block) < k and must_cover:
                    best_point = -1
                    best_gain = -1
                    for x in range(v):
                        if x in new_block:
                            continue
                        gain = 0
                        for ts in must_cover:
                            if x in ts:
                                remaining = set(ts) - new_block - {x}
                                if not remaining:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point == -1:
                        break
                    new_block.add(best_point)
                
                new_block_list = sorted(new_block)
                if tuple(new_block_list) not in existing_set:
                    # Check if replacing block i with new_block_list is valid
                    trial = current[:i] + [new_block_list] + current[i+1:]
                    covered = _compute_coverage(trial, v, k, t, all_tsets, tset_index)
                    if np.all(covered >= 1):
                        current = trial
                        improved = True
                        break
        
        if not improved:
            break
    
    return current


def construct(v, k, t):
    """
    Construct a covering design using genetic algorithm with greedy initialization.
    """
    global _crossover_deadline
    start_time = time.time()
    total_time_limit = 140  # Leave 10s margin for the 150s limit
    
    # Precompute all t-sets and index
    all_tsets = list(itertools.combinations(range(v), t))
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    
    rng = random.Random(42)
    
    # Phase 1: Generate initial population with greedy
    population = []
    pop_size = 6
    greedy_time = total_time_limit * 0.15
    
    for seed in range(pop_size):
        if time.time() - start_time > total_time_limit * 0.3:
            break
        rng_seed = random.Random(seed * 1000 + 42)
        blocks = _fast_greedy(v, k, t, rng_seed, time_limit=greedy_time, all_tsets=all_tsets, tset_index=tset_index)
        if blocks:
            population.append(blocks)
    
    if not population:
        # Fallback
        blocks = _fast_greedy(v, k, t, random.Random(0), time_limit=total_time_limit * 0.5, all_tsets=all_tsets, tset_index=tset_index)
        population = [blocks]
    
    # Remove redundant blocks from initial population
    for i in range(len(population)):
        if time.time() - start_time > total_time_limit * 0.4:
            break
        population[i] = _remove_redundant_blocks(population[i], v, k, t, all_tsets, tset_index)
    
    # Phase 2: Genetic algorithm
    ga_deadline = start_time + total_time_limit * 0.85
    _crossover_deadline = ga_deadline
    
    best_fitness = min(_fitness(pop, v, k, t, all_tsets, tset_index) for pop in population)
    best_blocks = min(population, key=lambda p: _fitness(p, v, k, t, all_tsets, tset_index))
    
    gen = 0
    while time.time() < ga_deadline:
        gen += 1
        new_population = []
        
        # Elitism: keep the best individual
        sorted_pop = sorted(population, key=lambda p: _fitness(p, v, k, t, all_tsets, tset_index))
        new_population.append(sorted_pop[0])
        
        # Generate offspring via crossover and mutation
        while len(new_population) < pop_size and time.time() < ga_deadline:
            # Select parents (tournament selection)
            p1 = sorted_pop[rng.randint(0, min(3, len(sorted_pop)-1))]
            p2 = sorted_pop[rng.randint(0, min(3, len(sorted_pop)-1))]
            
            # Crossover
            child = _crossover(p1, p2, v, k, t, rng, all_tsets, tset_index)
            
            # Mutation with limited time
            mutation_time = min(2.0, (ga_deadline - time.time()) * 0.3)
            if mutation_time > 0.5:
                child = _mutate(child, v, k, t, rng, all_tsets, tset_index, mutation_time)
            
            new_population.append(child)
        
        # Evaluate new population
        for i in range(len(new_population)):
            if time.time() >= ga_deadline:
                break
            fitness = _fitness(new_population[i], v, k, t, all_tsets, tset_index)
            if fitness < best_fitness:
                best_fitness = fitness
                best_blocks = new_population[i]
        
        population = new_population
        
        # Update sorted population
        sorted_pop = sorted(population, key=lambda p: _fitness(p, v, k, t, all_tsets, tset_index))
    
    # Phase 3: Final local search on best solution
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 3:
        best_blocks = _local_search(best_blocks, v, k, t, rng, time_limit=remaining_time * 0.9, all_tsets=all_tsets, tset_index=tset_index)
    
    return best_blocks

"""
Genetic algorithm approach for covering design C(19,7,4).
Strategy: 
  1. Fast greedy initialization to get a valid covering.
  2. Local removal pass to trim redundant blocks.
  3. Genetic algorithm with population of valid coverings, 
     block-level crossover, and mutation (replace/insert blocks).
  4. Numpy-accelerated coverage checks for speed.
"""
import itertools
import time
import random
import numpy as np
from copy import deepcopy

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each possible block (k-subset), which t-subsets it covers
    # But C(19,7) = 50388, and each covers C(7,4)=35 t-subsets
    # Store as list of arrays of t-subset indices
    all_blocks_list = list(itertools.combinations(range(v), k))
    n_all_blocks = len(all_blocks_list)
    
    # Map from block tuple to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks_list)}
    
    # For each block, precompute the list of t-subset indices it covers
    block_tsub_arrays = []
    for b in all_blocks_list:
        tsubs = []
        for ts in itertools.combinations(b, t):
            tsubs.append(tsub_idx[ts])
        block_tsub_arrays.append(np.array(tsubs, dtype=np.int32))
    
    def coverage_vector(blocks):
        """Compute coverage count vector for a list of blocks (tuples)."""
        cover = np.zeros(n_tsubsets, dtype=np.int32)
        for b in blocks:
            idx = block_to_idx.get(b)
            if idx is not None:
                cover[block_tsub_arrays[idx]] += 1
        return cover
    
    def is_valid_cover(cover):
        return np.all(cover > 0)
    
    def greedy_init(seed=None):
        """Greedy construction to get a valid covering."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        cover = np.zeros(n_tsubsets, dtype=np.int32)
        blocks = []
        
        while True:
            uncovered = np.where(cover == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a random uncovered t-subset
            ts_idx = rng.choice(uncovered.tolist())
            start_subset = all_tsubsets[ts_idx]
            block = list(start_subset)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_x = None
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_idx[ts]
                        if cover[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                block.append(best_x)
            
            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)
            
            # Update coverage
            idx = block_to_idx[block_tuple]
            cover[block_tsub_arrays[idx]] += 1
            
            if time.time() > deadline - 30.0:
                break
        
        return blocks
    
    def local_remove(blocks, max_time=10.0):
        """Try to remove redundant blocks."""
        blocks = list(blocks)
        cover = coverage_vector(blocks)
        removed = 0
        t0 = time.time()
        
        changed = True
        while changed and time.time() - t0 < max_time:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 20.0:
                    return blocks
                # Check if removing block i keeps all covered
                b_idx = block_to_idx[blocks[i]]
                b_tsubs = block_tsub_arrays[b_idx]
                # If all t-subsets in this block have cover > 1, we can remove
                if np.all(cover[b_tsubs] > 1):
                    cover[b_tsubs] -= 1
                    blocks.pop(i)
                    removed += 1
                    changed = True
                    break  # Restart scan since indices changed
        
        return blocks
    
    # Phase 1: Greedy initialization
    initial_blocks = greedy_init(seed=42)
    initial_blocks = local_remove(initial_blocks, max_time=5.0)
    
    # Ensure validity
    cover_check = coverage_vector(initial_blocks)
    if not is_valid_cover(cover_check):
        # Add more blocks if needed
        uncovered = np.where(cover_check == 0)[0]
        while len(uncovered) > 0:
            ts_idx = uncovered[0]
            start_subset = all_tsubsets[ts_idx]
            block = list(start_subset)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_idx[ts]
                        if cover_check[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            block_tuple = tuple(block)
            initial_blocks.append(block_tuple)
            b_idx = block_to_idx[block_tuple]
            cover_check[block_tsub_arrays[b_idx]] += 1
            uncovered = np.where(cover_check == 0)[0]
            if time.time() > deadline - 20.0:
                break
    
    best_blocks = list(initial_blocks)
    best_len = len(best_blocks)
    
    # Phase 2: Genetic Algorithm
    if time.time() < deadline - 40.0:
        pop_size = 30
        # Initialize population with variants of the best solution
        population = [list(best_blocks)]
        rng = random.Random(123)
        
        # Add diverse initial solutions
        for i in range(pop_size - 1):
            if time.time() > deadline - 40.0:
                break
            seed = 100 + i
            sol = greedy_init(seed=seed)
            sol = local_remove(sol, max_time=2.0)
            # Ensure validity
            cv = coverage_vector(sol)
            if is_valid_cover(cv):
                population.append(sol)
            else:
                population.append(list(best_blocks))
        
        # GA parameters
        gen = 0
        max_gens = 200
        elite_size = 5
        
        while gen < max_gens and time.time() < deadline - 15.0:
            # Sort by length
            population.sort(key=lambda x: len(x))
            
            # Update best
            if len(population[0]) < best_len:
                best_blocks = list(population[0])
                best_len = len(best_blocks)
            
            # Select parents (tournament selection)
            new_population = []
            
            # Elitism
            for i in range(min(elite_size, len(population))):
                new_population.append(list(population[i]))
            
            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline - 15.0:
                    break
                
                # Tournament selection for two parents
                def tournament():
                    candidates = rng.sample(population, min(3, len(population)))
                    candidates.sort(key=lambda x: len(x))
                    return candidates[0]
                
                parent1 = tournament()
                parent2 = tournament()
                
                # Crossover: block-level recombination
                # Take a random prefix from parent1 and random suffix from parent2
                if len(parent1) == 0 or len(parent2) == 0:
                    child = list(parent1) if len(parent1) > 0 else list(parent2)
                else:
                    # Random split point
                    split1 = rng.randint(0, len(parent1))
                    split2 = rng.randint(0, len(parent2))
                    
                    # Combine blocks, ensuring uniqueness
                    child_set = set()
                    child = []
                    for b in parent1[:split1]:
                        if b not in child_set:
                            child_set.add(b)
                            child.append(b)
                    for b in parent2[split2:]:
                        if b not in child_set:
                            child_set.add(b)
                            child.append(b)
                    
                    # If too few blocks, fill from parents
                    if len(child) < max(len(parent1), len(parent2)) // 2:
                        for b in parent1:
                            if b not in child_set:
                                child_set.add(b)
                                child.append(b)
                        for b in parent2:
                            if b not in child_set:
                                child_set.add(b)
                                child.append(b)
                
                # Mutation: replace some blocks with random ones
                if len(child) > 0 and rng.random() < 0.3:
                    n_mutate = max(1, len(child) // 10)
                    for _ in range(n_mutate):
                        if len(child) == 0:
                            break
                        pos = rng.randrange(len(child))
                        # Replace with a random block that covers uncovered areas
                        cv = coverage_vector(child)
                        uncovered = np.where(cv == 0)[0]
                        if len(uncovered) > 0:
                            ts_idx = rng.choice(uncovered.tolist())
                            start_subset = all_tsubsets[ts_idx]
                            new_block = list(start_subset)
                            while len(new_block) < k:
                                best_x = None
                                best_gain = -1
                                for x in range(v):
                                    if x in new_block:
                                        continue
                                    gain = 0
                                    for rest in itertools.combinations(new_block, t - 1):
                                        ts = tuple(sorted(rest + (x,)))
                                        idx = tsub_idx[ts]
                                        if cv[idx] == 0:
                                            gain += 1
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_x = x
                                new_block.append(best_x)
                            new_block.sort()
                            child[pos] = tuple(new_block)
                        else:
                            # Try to remove a redundant block
                            cv = coverage_vector(child)
                            removable = []
                            for i, b in enumerate(child):
                                b_idx = block_to_idx[b]
                                if np.all(cv[block_tsub_arrays[b_idx]] > 1):
                                    removable.append(i)
                            if removable:
                                pos2 = rng.choice(removable)
                                child.pop(pos2)
                
                # Ensure validity; if not, repair
                cv = coverage_vector(child)
                if not is_valid_cover(cv):
                    uncovered = np.where(cv == 0)[0]
                    while len(uncovered) > 0:
                        ts_idx = rng.choice(uncovered.tolist())
                        start_subset = all_tsubsets[ts_idx]
                        new_block = list(start_subset)
                        while len(new_block) < k:
                            best_x = None
                            best_gain = -1
                            for x in range(v):
                                if x in new_block:
                                    continue
                            gain = 0
                            for rest in itertools.combinations(new_block, t - 1):
                                ts = tuple(sorted(rest + (x,)))
                                idx = tsub_idx[ts]
                                if cv[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                            new_block.append(best_x)
                        new_block.sort()
                        child.append(tuple(new_block))
                        b_idx = block_to_idx[tuple(new_block)]
                        cv[block_tsub_arrays[b_idx]] += 1
                        uncovered = np.where(cv == 0)[0]
                        if time.time() > deadline - 15.0:
                            break
                
                new_population.append(child)
            
            population = new_population
            gen += 1
            
            # Local optimization on best individual each generation
            if gen % 10 == 0 and time.time() < deadline - 20.0:
                best_in_pop = min(population, key=lambda x: len(x))
                optimized = local_remove(best_in_pop, max_time=3.0)
                cv = coverage_vector(optimized)
                if is_valid_cover(cv) and len(optimized) < best_len:
                    best_blocks = list(optimized)
                    best_len = len(best_blocks)
                    population[0] = list(optimized)
    
    # Final validation
    final_cover = coverage_vector(best_blocks)
    if not is_valid_cover(final_cover):
        # Fallback to greedy
        best_blocks = greedy_init(seed=999)
        cv = coverage_vector(best_blocks)
        while not is_valid_cover(cv):
            uncovered = np.where(cv == 0)[0]
            if len(uncovered) == 0:
                break
            ts_idx = uncovered[0]
            start_subset = all_tsubsets[ts_idx]
            block = list(start_subset)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_idx[ts]
                        if cv[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            best_blocks.append(tuple(block))
            b_idx = block_to_idx[tuple(block)]
            cv[block_tsub_arrays[b_idx]] += 1
    
    return [list(b) for b in best_blocks]

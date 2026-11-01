"""Covering design C(24,7,3): GA-evolved block sets with greedy initialization and local search."""
import itertools
import time
import random
from collections import defaultdict
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsub_set = set(all_tsubs)
    n_tsubs = len(all_tsubs)
    
    # Index mapping for numpy arrays
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    def _build_greedy(uncovered_set, seed=None):
        """Build a greedy covering from the given uncovered set."""
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng
            
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered:
            if time.time() > deadline:
                break
            # Pick a random uncovered triple to start
            start_triple = local_rng.choice(list(uncovered))
            block = list(start_triple)
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline:
                    break
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)

            if len(block) < k:
                break

            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        
        return blocks

    def _evaluate_blocks(blocks):
        """
        Returns (fitness, coverage_counts, is_valid)
        fitness: sum of 1/count for covered, 1 for uncovered (lower is better)
        coverage_counts: numpy array of counts for each t-subset
        """
        counts = np.zeros(n_tsubs, dtype=np.int32)
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                counts[idx] += 1
        
        # Calculate fitness
        # Uncovered (0) contribute 1.0 each
        # Covered (>0) contribute 1.0/count
        fitness = 0.0
        for c in counts:
            if c == 0:
                fitness += 1.0
            else:
                fitness += 1.0 / c
        
        is_valid = np.all(counts > 0)
        return fitness, counts, is_valid

    def _local_search(blocks, counts, is_valid):
        """
        Local search: remove redundant blocks, try to replace blocks to improve coverage/efficiency.
        Returns improved blocks, counts, is_valid, fitness
        """
        blocks = [list(b) for b in blocks]
        counts = counts.copy()
        n = len(blocks)
        
        # Pass 1: Remove redundant blocks
        for _pass in range(3):
            if time.time() > deadline:
                break
            if not blocks:
                break
            
            removed_any = False
            for i in range(n - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                # Check if block i is redundant
                redundant = True
                for tsub in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[tsub]
                    if counts[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for tsub in itertools.combinations(blocks[i], t):
                        idx = tsub_to_idx[tsub]
                        counts[idx] -= 1
                    blocks.pop(i)
                    n -= 1
                    removed_any = True
            
            if not removed_any:
                break
        
        # Pass 2: Try to replace blocks with better ones
        # We only do this if we have time and the solution is valid
        if is_valid and time.time() < deadline - 5:
            for _pass in range(2):
                if time.time() > deadline:
                    break
                improved = False
                n = len(blocks)
                
                for i in range(n):
                    if time.time() > deadline:
                        break
                    
                    old_block = blocks[i]
                    # Temporarily remove
                    for tsub in itertools.combinations(old_block, t):
                        idx = tsub_to_idx[tsub]
                        counts[idx] -= 1
                    
                    # Find uncovered triples
                    newly_uncovered = []
                    for tsub in itertools.combinations(old_block, t):
                        idx = tsub_to_idx[tsub]
                        if counts[idx] == 0:
                            newly_uncovered.append(tsub)
                    
                    if not newly_uncovered:
                        # Block was redundant, keep it removed
                        improved = True
                        continue
                    
                    # Greedily build a new block to cover newly_uncovered
                    start_triple = list(newly_uncovered[0])
                    new_block = list(start_triple)
                    new_block_set = set(new_block)
                    uncovered = set(newly_uncovered)
                    
                    while len(new_block) < k and uncovered:
                        if time.time() > deadline:
                            break
                        best_x = -1
                        best_gain = -1
                        for x in range(v):
                            if x in new_block_set:
                                continue
                            gain = 0
                            for rest in itertools.combinations(new_block, t - 1):
                                candidate = tuple(sorted(rest + (x,)))
                                if candidate in uncovered:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x == -1:
                            break
                        new_block.append(best_x)
                        new_block_set.add(best_x)
                    
                    if len(new_block) < k:
                        # Restore old block
                        for tsub in itertools.combinations(old_block, t):
                            idx = tsub_to_idx[tsub]
                            counts[idx] += 1
                        continue
                    
                    new_block.sort()
                    
                    # Check if new block covers all newly_uncovered
                    new_tsubs = set(itertools.combinations(new_block, t))
                    if all(ts in new_tsubs for ts in newly_uncovered):
                        # Update counts for new block
                        for tsub in new_tsubs:
                            idx = tsub_to_idx[tsub]
                            counts[idx] += 1
                        blocks[i] = new_block
                        improved = True
                    else:
                        # Restore old block
                        for tsub in itertools.combinations(old_block, t):
                            idx = tsub_to_idx[tsub]
                            counts[idx] += 1
                
                if not improved:
                    break
        
        fitness, counts, is_valid = _evaluate_blocks(blocks)
        return blocks, counts, is_valid, fitness

    # --- Main Logic ---

    # 1. Generate initial population using greedy with different seeds
    pop_size = 10
    population = []  # List of (blocks, fitness, counts, is_valid)
    
    for i in range(pop_size):
        if time.time() > deadline - 20:
            break
        seed = 42 + i * 1000
        blocks = _build_greedy(all_tsub_set, seed=seed)
        fitness, counts, is_valid = _evaluate_blocks(blocks)
        population.append((blocks, fitness, counts, is_valid))
    
    if not population:
        # Fallback
        blocks = _build_greedy(all_tsub_set, seed=42)
        return blocks

    # Sort by fitness
    population.sort(key=lambda x: x[1])
    
    best_blocks = population[0][0]
    best_fitness = population[0][1]
    best_counts = population[0][2]
    best_valid = population[0][3]
    
    # Run local search on the best initial solution
    if time.time() < deadline - 10:
        lb_blocks, lb_counts, lb_valid, lb_fitness = _local_search(population[0][0], population[0][2], population[0][3])
        if lb_fitness < best_fitness:
            best_blocks = lb_blocks
            best_fitness = lb_fitness
            best_counts = lb_counts
            best_valid = lb_valid

    # 2. Genetic Algorithm Loop
    ga_time_limit = deadline - 5
    generation = 0
    stagnation_counter = 0
    
    while time.time() < ga_time_limit:
        generation += 1
        
        # Selection: Keep top half, generate offspring for the rest
        # We'll use a simple tournament selection or just keep best and mutate
        
        new_population = []
        
        # Keep the best few
        elite_count = max(2, pop_size // 4)
        for i in range(elite_count):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > ga_time_limit:
                break
            
            # Select parents
            # Tournament selection
            p1 = population[rng.randint(0, len(population)-1)]
            p2 = population[rng.randint(0, len(population)-1)]
            
            # Crossover: Take random subset of blocks from p1 and p2
            # But we need to ensure we cover everything.
            # Strategy: Start with a random mix of blocks, then greedily add blocks to cover gaps.
            
            # Let's try a different crossover:
            # Take all blocks from p1, then remove redundant ones, then add blocks from p2 if they help.
            # Or simpler: Randomly shuffle blocks from p1 and p2, take first N, then repair.
            
            # Let's do: Take a random number of blocks from p1 and p2, then repair greedily.
            
            n_blocks_p1 = len(p1[0])
            n_blocks_p2 = len(p2[0])
            
            # Mix
            mixed_blocks = []
            for b in p1[0]:
                if rng.random() < 0.5:
                    mixed_blocks.append(list(b))
            for b in p2[0]:
                if rng.random() < 0.5:
                    mixed_blocks.append(list(b))
            
            # Ensure no duplicates
            unique_blocks = []
            seen = set()
            for b in mixed_blocks:
                key = tuple(sorted(b))
                if key not in seen:
                    seen.add(key)
                    unique_blocks.append(b)
            
            # Repair: Greedily add blocks to cover all uncovered triples
            # Calculate current coverage
            counts = np.zeros(n_tsubs, dtype=np.int32)
            for b in unique_blocks:
                for tsub in itertools.combinations(b, t):
                    idx = tsub_to_idx[tsub]
                    counts[idx] += 1
            
            uncovered = set()
            for i in range(n_tsubs):
                if counts[i] == 0:
                    uncovered.add(all_tsubs[i])
            
            # Greedily add blocks
            while uncovered:
                if time.time() > ga_time_limit:
                    break
                start_triple = rng.choice(list(uncovered))
                block = list(start_triple)
                block_set = set(block)
                
                while len(block) < k:
                    if time.time() > ga_time_limit:
                        break
                    best_x = -1
                    best_gain = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            candidate = tuple(sorted(rest + (x,)))
                            if candidate in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                
                if len(block) < k:
                    break
                block.sort()
                unique_blocks.append(block)
                for tsub in itertools.combinations(block, t):
                    counts[tsub_to_idx[tsub]] += 1
                    uncovered.discard(tsub)
            
            # Evaluate offspring
            fitness, counts, is_valid = _evaluate_blocks(unique_blocks)
            new_population.append((unique_blocks, fitness, counts, is_valid))
        
        population = new_population
        population.sort(key=lambda x: x[1])
        
        # Check for improvement
        if population[0][1] < best_fitness:
            best_fitness = population[0][1]
            best_blocks = population[0][0]
            best_counts = population[0][2]
            best_valid = population[0][3]
            stagnation_counter = 0
            
            # Try local search on the new best
            if time.time() < ga_time_limit - 5:
                lb_blocks, lb_counts, lb_valid, lb_fitness = _local_search(best_blocks, best_counts, best_valid)
                if lb_fitness < best_fitness:
                    best_blocks = lb_blocks
                    best_fitness = lb_fitness
                    best_counts = lb_counts
                    best_valid = lb_valid
                    # Insert improved solution into population
                    population[0] = (best_blocks, best_fitness, best_counts, best_valid)
                    population.sort(key=lambda x: x[1])
        else:
            stagnation_counter += 1
            # If stagnating, introduce more randomness in next generation
            # (Handled implicitly by random restarts in crossover)

    return best_blocks


if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    # Verify
    all_tsubs = set(itertools.combinations(range(24), 3))
    covered = set()
    for b in blocks:
        covered.update(itertools.combinations(b, 3))
    print(f"Covered: {len(covered)} / {len(all_tsubs)}")
    print(f"Valid: {all_tsubs == covered}")

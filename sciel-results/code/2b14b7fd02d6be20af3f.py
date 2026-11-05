import itertools
import time
import sys
import numpy as np
import random
from scipy.sparse import csr_matrix

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using genetic algorithm with local search.
    """
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute t-subset indices for each block
    block_tsub_indices = [None] * num_kblocks
    for i, b in enumerate(all_kblocks):
        indices = []
        for ts in itertools.combinations(b, t):
            indices.append(tsub_to_idx[ts])
        block_tsub_indices[i] = indices
    
    # Greedy construction for initial population
    def greedy_construct():
        uncovered = [0] * num_tsubs
        for i in range(num_tsubs):
            uncovered[i] = 1
        selected = []
        
        while any(uncovered) and time.monotonic() < deadline - 30:
            best_gain = 0
            best_block = -1
            for i in range(num_kblocks):
                gain = sum(uncovered[j] for j in block_tsub_indices[i])
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            
            if best_gain == 0:
                break
            
            selected.append(best_block)
            for j in block_tsub_indices[best_block]:
                uncovered[j] = 0
        
        # Prune
        if selected:
            counts = [0] * num_tsubs
            for b_idx in selected:
                for j in block_tsub_indices[b_idx]:
                    counts[j] += 1
            
            changed = True
            while changed and time.monotonic() < deadline - 25:
                changed = False
                new_selected = []
                for b_idx in selected:
                    redundant = True
                    for j in block_tsub_indices[b_idx]:
                        if counts[j] == 1:
                            redundant = False
                            break
                    if redundant:
                        for j in block_tsub_indices[b_idx]:
                            counts[j] -= 1
                        changed = True
                    else:
                        new_selected.append(b_idx)
                selected = new_selected
        
        return selected
    
    # Compute coverage counts for a set of blocks
    def compute_counts(block_set):
        counts = [0] * num_tsubs
        for b_idx in block_set:
            for j in block_tsub_indices[b_idx]:
                counts[j] += 1
        return counts
    
    # Check if all t-subsets are covered
    def is_complete(counts):
        return all(c >= 1 for c in counts)
    
    # Local search: 1-swap and 2-swap moves
    def local_search(block_set, max_time=None):
        if max_time is None:
            max_time = deadline - 5
        block_set = list(block_set)
        counts = compute_counts(block_set)
        
        while time.monotonic() < max_time:
            improved = False
            
            # Try 1-swap: remove one block, add one block
            for i in range(len(block_set)):
                b_remove = block_set[i]
                # Check if removing b_remove keeps coverage
                temp_counts = counts[:]
                for j in block_tsub_indices[b_remove]:
                    temp_counts[j] -= 1
                if any(c < 1 for c in temp_counts):
                    continue
                
                # Try adding each block
                for b_add in range(num_kblocks):
                    if b_add == b_remove:
                        continue
                    if b_add in block_set:
                        continue
                    # Check if this reduces total "excess"
                    old_excess = sum(c - 1 for c in temp_counts if c > 1)
                    new_counts = temp_counts[:]
                    for j in block_tsub_indices[b_add]:
                        new_counts[j] += 1
                    new_excess = sum(c - 1 for c in new_counts if c > 1)
                    if new_excess < old_excess:
                        # Accept the move
                        block_set[i] = b_add
                        counts = new_counts
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove two blocks, add two blocks
            if len(block_set) >= 2:
                for i in range(len(block_set)):
                    for j in range(i + 1, len(block_set)):
                        b1 = block_set[i]
                        b2 = block_set[j]
                        temp_counts = counts[:]
                        for idx in block_tsub_indices[b1]:
                            temp_counts[idx] -= 1
                        for idx in block_tsub_indices[b2]:
                            temp_counts[idx] -= 1
                        if any(c < 1 for c in temp_counts):
                            continue
                        
                        # Try adding pairs of blocks
                        for b_add1 in range(num_kblocks):
                            if b_add1 in (b1, b2) or b_add1 in block_set:
                                continue
                            for b_add2 in range(b_add1 + 1, num_kblocks):
                                if b_add2 in (b1, b2) or b_add2 in block_set:
                                    continue
                                new_counts = temp_counts[:]
                                for idx in block_tsub_indices[b_add1]:
                                    new_counts[idx] += 1
                                for idx in block_tsub_indices[b_add2]:
                                    new_counts[idx] += 1
                                if any(c < 1 for c in new_counts):
                                    continue
                                old_excess = sum(c - 1 for c in temp_counts if c > 1)
                                new_excess = sum(c - 1 for c in new_counts if c > 1)
                                if new_excess < old_excess:
                                    block_set[i] = b_add1
                                    block_set[j] = b_add2
                                    counts = new_counts
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return block_set
    
    # Genetic algorithm
    def genetic_algorithm(pop_size, generations, elite_frac=0.1):
        # Initialize population with greedy + random perturbations
        population = []
        for i in range(pop_size):
            if i == 0:
                sol = greedy_construct()
            else:
                sol = greedy_construct()
                # Randomly perturb
                for _ in range(min(5, len(sol) // 4)):
                    if sol and random.random() < 0.5:
                        idx = random.randint(0, len(sol) - 1)
                        new_block = random.randint(0, num_kblocks - 1)
                        if new_block not in sol:
                            sol[idx] = new_block
            if sol:
                population.append(sol)
        
        if not population:
            return greedy_construct()
        
        best_sol = population[0]
        best_len = len(best_sol)
        
        for gen in range(generations):
            if time.monotonic() > deadline - 10:
                break
            
            # Evaluate fitness (length)
            population.sort(key=lambda s: len(s))
            
            # Keep elites
            elite_count = max(1, int(pop_size * elite_frac))
            new_population = population[:elite_count]
            
            # Generate offspring
            while len(new_population) < pop_size and time.monotonic() < deadline - 10:
                # Crossover
                parent1 = random.choice(population[:max(2, pop_size // 2)])
                parent2 = random.choice(population[:max(2, pop_size // 2)])
                
                if len(parent1) < len(parent2):
                    parent1, parent2 = parent2, parent1
                
                cross_point = random.randint(1, len(parent1) - 1) if len(parent1) > 1 else 0
                child = parent1[:cross_point] + parent2[cross_point:]
                # Remove duplicates
                seen = set()
                unique_child = []
                for b in child:
                    if b not in seen:
                        seen.add(b)
                        unique_child.append(b)
                child = unique_child
                
                # Mutation
                for _ in range(random.randint(1, 3)):
                    if random.random() < 0.3 and child:
                        # Swap
                        idx = random.randint(0, len(child) - 1)
                        new_block = random.randint(0, num_kblocks - 1)
                        if new_block not in child:
                            child[idx] = new_block
                    elif random.random() < 0.2 and child:
                        # Remove
                        idx = random.randint(0, len(child) - 1)
                        child.pop(idx)
                    elif random.random() < 0.2:
                        # Add
                        new_block = random.randint(0, num_kblocks - 1)
                        if new_block not in child:
                            child.append(new_block)
                
                new_population.append(child)
            
            population = new_population
        
        # Find best
        for sol in population:
            if sol and len(sol) < best_len:
                best_len = len(sol)
                best_sol = sol
        
        return best_sol
    
    # Run initial greedy
    best_blocks = greedy_construct()
    best_count = len(best_blocks)
    
    # Run genetic algorithm with multiple restarts
    random.seed(42)
    
    remaining = deadline - time.monotonic() - 5
    if remaining > 30:
        # Determine population size and generations based on time
        pop_size = min(20, max(10, int(remaining / 10)))
        generations = max(5, int(remaining / 15))
        
        ga_sol = genetic_algorithm(pop_size, generations)
        if ga_sol and len(ga_sol) < best_count:
            best_blocks = ga_sol
            best_count = len(best_blocks)
    
    # Local search on best solution
    if best_blocks and time.monotonic() < deadline - 10:
        ls_sol = local_search(best_blocks)
        if ls_sol and len(ls_sol) < best_count:
            best_blocks = ls_sol
            best_count = len(best_blocks)
    
    # Final local search with more time if available
    if best_blocks and time.monotonic() < deadline - 5:
        ls_sol = local_search(best_blocks)
        if ls_sol and len(ls_sol) < best_count:
            best_blocks = ls_sol
            best_count = len(best_blocks)
    
    if not best_blocks:
        return []
    
    return [list(all_kblocks[i]) for i in best_blocks]

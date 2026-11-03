import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Precompute: for each t-subset, which blocks cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    # Convert block_covers to numpy arrays for vectorized gain computation
    block_covers_arr = np.array(block_covers, dtype=np.int32)
    
    def greedy_mcv(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        counts = [0] * n_t
        selected_indices = []
        uncovered = set(range(n_t))
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
                
            # Find uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in uncovered:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            
            # Pick candidate that maximizes coverage of other uncovered subsets
            best_block = -1
            max_gain = -1
            best_candidates = []
            
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    best_candidates = [b]
                elif gain == max_gain:
                    best_candidates.append(b)
                    
            if len(best_candidates) > 1:
                best_block = random.choice(best_candidates)
            else:
                best_block = best_candidates[0]
                
            selected_indices.append(best_block)
            
            for s in block_covers[best_block]:
                if s in uncovered:
                    uncovered.discard(s)
                    
        return selected_indices

    def prune_redundant(selected_indices):
        final_counts = [0] * n_t
        for i in selected_indices:
            for s in block_covers[i]:
                final_counts[s] += 1
                
        changed = True
        selected_set = set(selected_indices)
        
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                redundant = True
                for s in block_covers[i]:
                    if final_counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(i)
                    for s in block_covers[i]:
                        final_counts[s] -= 1
                    changed = True
                    if not _deadline_ok(deadline):
                        break
                        
        return list(selected_set)

    def repair_coverage(selected_set):
        """Given a set of selected block indices, add blocks until all t-subsets are covered."""
        if not _deadline_ok(deadline):
            return list(selected_set)
        
        # Count coverage
        counts = [0] * n_t
        for i in selected_set:
            for s in block_covers[i]:
                counts[s] += 1
        
        # Find uncovered subsets
        uncovered = set(s for s in range(n_t) if counts[s] == 0)
        
        if not uncovered:
            return list(selected_set)
        
        # Greedily add blocks to cover remaining subsets
        selected_list = list(selected_set)
        while uncovered and _deadline_ok(deadline):
            # Find uncovered subset with fewest candidates
            min_count = float('inf')
            target_sub = -1
            for s in uncovered:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            
            # Pick block that covers most uncovered subsets
            best_block = -1
            max_gain = -1
            best_candidates = []
            
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    best_candidates = [b]
                elif gain == max_gain:
                    best_candidates.append(b)
                    
            if len(best_candidates) > 1:
                best_block = random.choice(best_candidates)
            else:
                best_block = best_candidates[0]
                
            selected_list.append(best_block)
            selected_set.add(best_block)
            
            for s in block_covers[best_block]:
                if s in uncovered:
                    uncovered.discard(s)
                    
        return selected_list

    def crossover(parent1, parent2):
        """Combine two parents by randomly selecting blocks from each, then repair."""
        if not _deadline_ok(deadline):
            return list(parent1)
        
        p1_set = set(parent1)
        p2_set = set(parent2)
        
        # Randomly select blocks from each parent
        child_set = set()
        for b in p1_set:
            if random.random() < 0.5:
                child_set.add(b)
        for b in p2_set:
            if random.random() < 0.5:
                child_set.add(b)
        
        # Repair coverage
        child = repair_coverage(child_set)
        
        # Prune redundant blocks
        child = prune_redundant(child)
        
        return child

    def mutate(selected_indices):
        """Remove a random block and re-add minimum-cost blocks to restore validity."""
        if not _deadline_ok(deadline):
            return list(selected_indices)
        
        if len(selected_indices) == 0:
            return list(selected_indices)
        
        # Remove a random block
        idx_to_remove = random.randint(0, len(selected_indices) - 1)
        removed_block = selected_indices[idx_to_remove]
        
        new_selected = [b for i, b in enumerate(selected_indices) if i != idx_to_remove]
        new_set = set(new_selected)
        
        # Repair coverage
        new_selected = repair_coverage(new_set)
        
        # Prune redundant blocks
        new_selected = prune_redundant(new_selected)
        
        return new_selected

    def local_search_swap(selected_indices, max_swaps=10):
        """Try to improve by swapping blocks: remove one, add a different one if it helps."""
        if not _deadline_ok(deadline):
            return list(selected_indices)
        
        selected_set = set(selected_indices)
        counts = [0] * n_t
        for i in selected_set:
            for s in block_covers[i]:
                counts[s] += 1
        
        improved = True
        while improved and _deadline_ok(deadline):
            improved = False
            
            # Try removing each block and see if we can add a better one
            for b in list(selected_set):
                if not _deadline_ok(deadline):
                    break
                    
                # Check if removing b leaves any subset uncovered
                would_uncover = []
                for s in block_covers[b]:
                    if counts[s] == 1:
                        would_uncover.append(s)
                
                if not would_uncover:
                    # Safe to remove - do it
                    selected_set.remove(b)
                    for s in block_covers[b]:
                        counts[s] -= 1
                    improved = True
                    break
                else:
                    # Need to find a replacement that covers all would_uncover subsets
                    # Find blocks that cover all of would_uncover
                    candidate_blocks = set(sub_to_blocks[would_uncover[0]])
                    for s in would_uncover[1:]:
                        candidate_blocks &= set(sub_to_blocks[s])
                    
                    if not candidate_blocks:
                        continue
                    
                    # Pick the candidate that maximizes additional coverage
                    best_block = -1
                    max_gain = -1
                    for cb in candidate_blocks:
                        gain = 0
                        for s in block_covers[cb]:
                            if counts[s] == 0:
                                gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_block = cb
                            
                    if best_block != -1 and best_block not in selected_set:
                        # Remove b, add best_block
                        for s in block_covers[b]:
                            counts[s] -= 1
                        selected_set.remove(b)
                        
                        for s in block_covers[best_block]:
                            counts[s] += 1
                        selected_set.add(best_block)
                        improved = True
                        break
        
        return list(selected_set)

    # Initial population: diverse greedy restarts
    pop_size = 5
    population = []
    
    for i in range(pop_size):
        if not _deadline_ok(deadline):
            break
        selected = greedy_mcv(i)
        selected = prune_redundant(selected)
        population.append(selected)
    
    if not population:
        # Fallback
        selected = greedy_mcv()
        selected = prune_redundant(selected)
        population = [selected]
    
    best_result = min(population, key=len)
    best_len = len(best_result)
    
    # Genetic algorithm loop
    generation = 0
    while _deadline_ok(deadline - 10.0):  # Leave 10s for final polish
        generation += 1
        
        # Sort population by size (fitness)
        population.sort(key=len)
        
        # Elitism: keep best 2
        elites = population[:2]
        
        new_population = list(elites)
        
        # Fill rest with offspring
        while len(new_population) < pop_size and _deadline_ok(deadline - 10.0):
            # Select parents (tournament selection)
            p1 = random.choice(population[:max(2, len(population)//2)])
            p2 = random.choice(population[:max(2, len(population)//2)])
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation with some probability
            if random.random() < 0.3:
                child = mutate(child)
            
            # Local search
            if _deadline_ok(deadline - 10.0):
                child = local_search_swap(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Update best
        current_best = min(population, key=len)
        if len(current_best) < best_len:
            best_len = len(current_best)
            best_result = current_best
    
    # Final local search on best result
    if _deadline_ok(deadline - 5.0):
        best_result = local_search_swap(best_result)
        best_result = prune_redundant(best_result)
    
    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
             pass
             
    return result

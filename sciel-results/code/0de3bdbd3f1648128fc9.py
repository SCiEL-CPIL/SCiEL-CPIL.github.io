import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block bitmasks (Python ints)
    block_bitmasks = []
    # For fast pruning, we also need to know which triples a block covers
    block_triples = []
    
    for block in all_blocks:
        mask = 0
        tr_list = []
        for tup in itertools.combinations(block, t):
            idx = triple_to_idx[tup]
            mask |= (1 << idx)
            tr_list.append(idx)
        block_bitmasks.append(mask)
        block_triples.append(tr_list)
        
    # Precompute triple_to_blocks for greedy
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, tr_list in enumerate(block_triples):
        for tr_idx in tr_list:
            triple_to_blocks[tr_idx].append(i)
            
    triple_degree = [len(triple_to_blocks[j]) for j in range(num_triples)]

    def prune(blocks):
        """
        Remove redundant blocks from a list of block indices.
        Returns a list of block indices that forms a valid cover (if the input did)
        and is minimal with respect to this specific removal order.
        """
        if not blocks:
            return []
        
        # Count coverage of each triple
        counts = [0] * num_triples
        for bi in blocks:
            for tr_idx in block_triples[bi]:
                counts[tr_idx] += 1
                
        new_blocks = []
        # To make pruning effective, we might want to try removing blocks in a specific order.
        # Blocks covering "rare" triples (low count) are less likely to be redundant.
        # But simple iteration is faster. We can shuffle for diversity.
        # For speed, we just iterate.
        for bi in blocks:
            removable = True
            for tr_idx in block_triples[bi]:
                if counts[tr_idx] == 1:
                    removable = False
                    break
            if not removable:
                new_blocks.append(bi)
            else:
                for tr_idx in block_triples[bi]:
                    counts[tr_idx] -= 1
                    
        return new_blocks

    def greedy_cover(random_seed):
        """
        Fast randomized greedy set cover.
        """
        random.seed(random_seed)
        uncovered_mask = (1 << num_triples) - 1
        block_indices = []
        
        while uncovered_mask:
            # Find an uncovered triple
            # Randomly pick a bit from the mask
            # To be fast, we can pick a random index and check, but picking a set bit is better.
            # For 2024 bits, finding a set bit is O(1) amortized if we use random sampling?
            # Or just iterate.
            # Let's pick a random triple index and check if it's uncovered.
            # If not, try again. Since coverage is high, this is fast.
            
            # Better: maintain a list of uncovered triples? Updating is expensive.
            # Let's stick to bitmask.
            # Pick a random position
            tr_idx = random.randrange(num_triples)
            if not (uncovered_mask & (1 << tr_idx)):
                continue
            
            # Get candidates
            candidates = triple_to_blocks[tr_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            # Evaluate gain
            for bi in candidates:
                gain = (block_bitmasks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    if best_gain == 35: # Max possible for k=7, t=3
                        break
                elif gain == best_gain:
                    # Random tie-break
                    if random.random() < 0.5:
                        best_block_idx = bi
            
            if best_block_idx != -1:
                block_indices.append(best_block_idx)
                uncovered_mask &= ~block_bitmasks[best_block_idx]
            else:
                break
                
        return block_indices

    # Initial population
    pop_size = 10
    population = []
    for i in range(pop_size):
        if time.time() - start_time > timeout - 20:
            break
        blocks = greedy_cover(i)
        blocks = prune(blocks)
        population.append(blocks)
        
    # Sort population by size
    population.sort(key=lambda x: len(x))
    
    best_blocks = population[0] if population else []
    best_size = len(best_blocks)
    
    # Evolution
    # We will mix pairs and mutate.
    # Crossover: Union + Prune
    # Mutation: Swap + Prune
    
    gen = 0
    while time.time() - start_time < timeout - 10:
        new_population = []
        
        # Number of offspring to generate per generation
        # Try to keep it fast
        num_offspring = 5
        
        for _ in range(num_offspring):
            if time.time() - start_time > timeout - 5:
                break
            
            # Select parents (Tournament or random)
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover: Union
            child_blocks = list(set(p1 + p2))
            
            # Prune
            child_blocks = prune(child_blocks)
            
            # Mutation: Try a few swaps
            # Pick a block to remove and a random block to add
            if child_blocks and random.random() < 0.7:
                # Try 3 random swaps
                for _ in range(3):
                    if time.time() - start_time > timeout - 5:
                        break
                    # Pick a block to swap out
                    swap_out_idx = random.randrange(len(child_blocks))
                    swap_out_block = child_blocks[swap_out_idx]
                    
                    # Pick a random block to swap in
                    swap_in_block = random.randrange(num_blocks)
                    
                    # Create new set
                    new_set = set(child_blocks)
                    new_set.remove(swap_out_block)
                    new_set.add(swap_in_block)
                    
                    # Prune
                    candidate = prune(list(new_set))
                    
                    # Accept if valid (prune ensures validity if input was valid and we didn't break it? 
                    # No, swap might break validity. Prune only removes redundant blocks.
                    # We must check if candidate is a valid cover.
                    # How to check validity fast?
                    # Check if union of bitmasks of candidate is all ones.
                    
                    # Optimization: Only accept if size is <= current size
                    if len(candidate) <= len(child_blocks):
                        # Check validity
                        mask = 0
                        for bi in candidate:
                            mask |= block_bitmasks[bi]
                        if mask == (1 << num_triples) - 1:
                            child_blocks = candidate
                            break # Stop swapping if we found a good one? Or keep trying?
                            # Let's keep trying if it improved, or just break to save time.
                            # If it's same size, it's a lateral move. If smaller, it's a hill climb.
                            # We want to explore.
            
            new_population.append(child_blocks)
            
        # Update population
        population.extend(new_population)
        # Keep only the best pop_size individuals
        population.sort(key=lambda x: len(x))
        population = population[:pop_size]
        
        # Update best
        if len(population[0]) < best_size:
            best_size = len(population[0])
            best_blocks = population[0]
            
        gen += 1

    return [list(all_blocks[bi]) for bi in best_blocks]

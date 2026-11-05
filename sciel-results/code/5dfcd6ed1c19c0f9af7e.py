"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Precompute all 4-subsets of 19 points and their indices
_ALL_SUBSETS_19_4 = list(itertools.combinations(range(19), 4))
_SUBSET_INDEX_19_4 = {s: i for i, s in enumerate(_ALL_SUBSETS_19_4)}
_NUM_SUBSETS = len(_ALL_SUBSETS_19_4)  # 3876

# Precompute all 7-subsets of 19 points
_ALL_BLOCKS_19_7 = list(itertools.combinations(range(19), 7))
_NUM_BLOCKS = len(_ALL_BLOCKS_19_7)  # 50388

# Precompute coverage matrix: for each block, which 4-subsets it covers
_COVERAGE_MATRIX = np.zeros((_NUM_BLOCKS, _NUM_SUBSETS), dtype=np.uint8)
for _i, _block in enumerate(_ALL_BLOCKS_19_7):
    for _combo in itertools.combinations(_block, 4):
        _idx = _SUBSET_INDEX_19_4[_combo]
        _COVERAGE_MATRIX[_i, _idx] = 1

# Precompute for each 4-subset, which blocks contain it (for fast repair)
_SUBSET_TO_BLOCKS = {}
for _i, _block in enumerate(_ALL_BLOCKS_19_7):
    for _combo in itertools.combinations(_block, 4):
        _idx = _SUBSET_INDEX_19_4[_combo]
        if _idx not in _SUBSET_TO_BLOCKS:
            _SUBSET_TO_BLOCKS[_idx] = []
        _SUBSET_TO_BLOCKS[_idx].append(_i)

def _block_mask(block):
    """Get the coverage mask for a single block (returns numpy uint8 array)."""
    mask = np.zeros(_NUM_SUBSETS, dtype=np.uint8)
    for combo in itertools.combinations(block, 4):
        idx = _SUBSET_INDEX_19_4[combo]
        mask[idx] = 1
    return mask

def _greedy_cover(time_limit=5.0):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    uncovered = np.ones(_NUM_SUBSETS, dtype=np.uint8)
    block_indices = []
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        # Compute gain for each block
        gains = _COVERAGE_MATRIX @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        block_indices.append(best_idx)
        uncovered[_COVERAGE_MATRIX[best_idx]] = 0
    
    return [_ALL_BLOCKS_19_7[i] for i in block_indices]

def _repair(blocks):
    """Repair a list of blocks to ensure full coverage, then reduce."""
    if not blocks:
        return _greedy_cover(time_limit=5.0)
    
    # Compute current coverage
    total_mask = np.zeros(_NUM_SUBSETS, dtype=np.uint8)
    for b in blocks:
        for combo in itertools.combinations(b, 4):
            idx = _SUBSET_INDEX_19_4[combo]
            total_mask[idx] = 1
    
    uncovered = 1 - total_mask
    
    # Add blocks to cover remaining gaps
    if uncovered.any():
        # Use subset-to-blocks mapping for fast candidate generation
        while uncovered.any():
            first_idx = int(np.where(uncovered)[0][0])
            # Get all blocks containing this 4-subset
            candidates = _SUBSET_TO_BLOCKS[first_idx]
            best_block = None
            best_gain = -1
            for bi in candidates:
                gain = int(np.sum(_COVERAGE_MATRIX[bi] * uncovered))
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            if best_block is not None and best_gain > 0:
                blocks.append(_ALL_BLOCKS_19_7[best_block])
                uncovered[_COVERAGE_MATRIX[best_block]] = 0
            else:
                break
    
    # Reduction phase: remove redundant blocks
    n = len(blocks)
    if n <= 1:
        return blocks
    
    # Compute mask for each block
    block_masks = []
    for b in blocks:
        mask = np.zeros(_NUM_SUBSETS, dtype=np.uint8)
        for combo in itertools.combinations(b, 4):
            idx = _SUBSET_INDEX_19_4[combo]
            mask[idx] = 1
        block_masks.append(mask)
    
    # Repeatedly try to remove blocks
    changed = True
    while changed:
        changed = False
        n = len(blocks)
        for i in range(n):
            # Check if removing block i still leaves full coverage
            # total_mask_without_i = total_mask - block_masks[i] (but only where total_mask was 1)
            # If total_mask_without_i has any 0 where total_mask had 1, block i is needed
            without_i = total_mask.copy()
            without_i[block_masks[i] == 1] = 0
            # But we need to recompute: union of all other blocks
            # Faster: check if any 4-subset is ONLY covered by block i
            # i.e., (total_mask & block_masks[i]) has a position where no other block covers it
            only_i = total_mask & block_masks[i]
            # For each position in only_i, check if any other block covers it
            # This is expensive. Alternative: recompute union without i
            if len(blocks) > 1:
                union_others = np.zeros(_NUM_SUBSETS, dtype=np.uint8)
                for j in range(len(blocks)):
                    if j != i:
                        union_others |= block_masks[j]
                # Check if total_mask is subset of union_others
                if np.all((total_mask == 1) & (union_others == 0) == 0):
                    # Block i is redundant
                    del blocks[i]
                    del block_masks[i]
                    # Recompute total_mask
                    total_mask = np.zeros(_NUM_SUBSETS, dtype=np.uint8)
                    for bm in block_masks:
                        total_mask |= bm
                    changed = True
                    break
    
    return blocks

def _crossover(parent1, parent2):
    """Merge two coverings by greedy selection from their union."""
    # Union of blocks (deduplicate)
    pool = list(set([tuple(b) for b in parent1 + parent2]))
    
    if not pool:
        return _greedy_cover(time_limit=3.0)
    
    # Compute masks for pool
    pool_masks = []
    for b in pool:
        mask = np.zeros(_NUM_SUBSETS, dtype=np.uint8)
        for combo in itertools.combinations(b, 4):
            idx = _SUBSET_INDEX_19_4[combo]
            mask[idx] = 1
        pool_masks.append(mask)
    
    uncovered = np.ones(_NUM_SUBSETS, dtype=np.uint8)
    selected = []
    available = list(range(len(pool)))
    
    while uncovered.any() and available:
        best_idx = -1
        best_gain = -1
        for i in available:
            gain = int(np.sum(pool_masks[i] * uncovered))
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        selected.append(pool[best_idx])
        uncovered[pool_masks[best_idx]] = 0
        available.remove(best_idx)
    
    if uncovered.any():
        selected = _repair(selected)
    
    return selected

def _mutation(blocks):
    """Replace a random block with a random one, then repair."""
    if not blocks:
        return _greedy_cover(time_limit=3.0)
    
    mutated = list(blocks)
    idx = random.randint(0, len(mutated) - 1)
    mutated[idx] = random.choice(_ALL_BLOCKS_19_7)
    return _repair(mutated)

def construct(v, k, t):
    """Main entry point."""
    start_time = time.time()
    time_limit = 115.0
    
    # Initial population
    greedy1 = _greedy_cover(time_limit=8.0)
    
    random_blocks = [random.choice(_ALL_BLOCKS_19_7) for _ in range(80)]
    random1 = _repair(random_blocks)
    
    population = [greedy1, random1]
    
    # Add more diverse individuals
    for _ in range(3):
        rand_blocks = [random.choice(_ALL_BLOCKS_19_7) for _ in range(50)]
        pop_ind = _repair(rand_blocks)
        population.append(pop_ind)
    
    best_individual = min(population, key=len)
    best_len = len(best_individual)
    
    while time.time() - start_time < time_limit:
        # Sort by length
        population.sort(key=len)
        
        # Elitism
        elite_size = max(2, len(population) // 2)
        elite = population[:elite_size]
        
        new_population = list(elite)
        
        # Generate children
        while len(new_population) < len(population):
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.4 and child:
                child = _mutation(child)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=len)
        if len(current_best) < best_len:
            best_len = len(current_best)
            best_individual = current_best
            if best_len <= 151:
                break
    
    # Final repair
    final_blocks = _repair(best_individual)
    
    # Validate
    total_mask = np.zeros(_NUM_SUBSETS, dtype=np.uint8)
    for b in final_blocks:
        for combo in itertools.combinations(b, 4):
            idx = _SUBSET_INDEX_19_4[combo]
            total_mask[idx] = 1
    
    if not np.all(total_mask == 1):
        # Fallback to greedy
        final_blocks = _greedy_cover(time_limit=110.0)
    
    return final_blocks
# EVOLVE-BLOCK-END

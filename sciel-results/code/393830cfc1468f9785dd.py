"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Global cache for pre-computed data to avoid re-computation in loops
_CACHE = {}

def _get_precomputed(v, k, t):
    """Pre-compute all blocks, their bitmasks, and subset mapping."""
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
        
    subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(subsets)
    subset_to_idx = {s: i for i, s in enumerate(subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute bitmask for each block
    # Each block covers C(k, t) subsets.
    # We represent coverage as a Python integer bitmask.
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            idx = subset_to_idx[combo]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    _CACHE[key] = (all_blocks, block_masks, subset_to_idx, num_subsets)
    return _CACHE[key]

def _greedy_cover(v, k, t, time_limit=10.0, seed=None):
    """Fast greedy covering using bitmasks."""
    if seed is not None:
        random.seed(seed)
        
    start_time = time.time()
    all_blocks, block_masks, subset_to_idx, num_subsets = _get_precomputed(v, k, t)
    
    # Full coverage mask
    full_mask = (1 << num_subsets) - 1
    uncovered = full_mask
    blocks = []
    used_indices = set()
    
    while uncovered:
        best_idx = -1
        best_gain = 0
        
        # Find block with max gain
        # We can skip blocks already used
        for i in range(len(all_blocks)):
            if i in used_indices:
                continue
            gain = bin(uncovered & block_masks[i]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break # Should not happen if uncovered is non-zero and we have valid blocks
            
        blocks.append(list(all_blocks[best_idx]))
        used_indices.add(best_idx)
        uncovered &= ~block_masks[best_idx]
        
        if time.time() - start_time > time_limit:
            break
            
    return blocks

def _reduce_and_repair(v, k, t, blocks, time_limit=5.0):
    """Reduce redundant blocks and repair if invalid."""
    start_time = time.time()
    all_blocks, block_masks, subset_to_idx, num_subsets = _get_precomputed(v, k, t)
    
    # Map block tuple to index for quick lookup
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Convert input blocks to indices
    block_indices = []
    for b in blocks:
        b_tuple = tuple(b)
        if b_tuple in block_to_idx:
            block_indices.append(block_to_idx[b_tuple])
        else:
            # If block is not in pre-computed list (should not happen if v,k,t match), 
            # we need to handle it. But construct is called with fixed v,k,t.
            pass
            
    if not block_indices:
        return _greedy_cover(v, k, t, time_limit=time_limit)

    # Calculate union of current masks
    current_union = 0
    for idx in block_indices:
        current_union |= block_masks[idx]
        
    full_mask = (1 << num_subsets) - 1
    
    # If invalid, we need to add blocks
    if current_union != full_mask:
        # Greedy add
        uncovered = full_mask & ~current_union
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(len(all_blocks)):
                if i in block_indices:
                    continue
                gain = bin(uncovered & block_masks[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            block_indices.append(best_idx)
            current_union |= block_masks[best_idx]
            uncovered &= ~block_masks[best_idx]
            if time.time() - start_time > time_limit:
                break

    # Reduction Phase: Remove redundant blocks
    # A block is redundant if its mask is a subset of the union of all other blocks.
    # i.e., (current_union & ~mask_i) == current_union  => mask_i is subset of (current_union & ~mask_i) is FALSE?
    # No. mask_i is redundant if (current_union_without_i) == current_union.
    # current_union_without_i = current_union & ~mask_i.
    # So if (current_union & ~mask_i) == current_union, then mask_i contributes nothing new.
    
    changed = True
    while changed:
        changed = False
        for i in range(len(block_indices)):
            if time.time() - start_time > time_limit:
                break
            idx = block_indices[i]
            mask_i = block_masks[idx]
            # Check if removing idx leaves the union unchanged
            # We need to know the union of all OTHERS.
            # Efficient way: 
            # If mask_i is subset of (current_union & ~mask_i), it's redundant? No.
            # Let's compute union of others.
            # This is O(N) per check, O(N^2) total. N~150. 150^2 = 22500. Fast enough.
            
            union_others = 0
            for j in range(len(block_indices)):
                if j != i:
                    union_others |= block_masks[block_indices[j]]
                    
            if (union_others & mask_i) == mask_i:
                # mask_i is fully covered by others
                block_indices.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    # Convert back to list of lists
    return [list(all_blocks[idx]) for idx in block_indices]

def _local_search(v, k, t, blocks, time_limit=5.0):
    """Try to reduce block count via swaps or deletions."""
    start_time = time.time()
    all_blocks, block_masks, subset_to_idx, num_subsets = _get_precomputed(v, k, t)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    full_mask = (1 << num_subsets) - 1
    
    # Get current indices
    current_indices = [block_to_idx[tuple(b)] for b in blocks]
    
    # Try removing each block and repairing
    for i in range(len(current_indices)):
        if time.time() - start_time > time_limit:
            break
            
        # Remove block i
        trial_indices = current_indices[:i] + current_indices[i+1:]
        
        # Calculate union of trial
        union_trial = 0
        for idx in trial_indices:
            union_trial |= block_masks[idx]
            
        if union_trial == full_mask:
            # We successfully removed a block!
            current_indices = trial_indices
            break # Restart search with new set
            
    # Try swapping a block with another
    # This is more complex. Let's stick to deletion for now as it's high impact.
    # But we can also try: Pick a block, remove it, greedily add 1 block. If count same, keep? No, we want reduce.
    # So deletion is the main driver.
    
    return [list(all_blocks[idx]) for idx in current_indices]

def _crossover(parent1, parent2, v, k, t):
    """Merge two coverings."""
    # Union of blocks
    pool = list(set([tuple(b) for b in parent1 + parent2]))
    
    # Reduce and repair the pool
    child = _reduce_and_repair(v, k, t, pool, time_limit=5.0)
    
    # Local search
    if len(child) > 0:
        child = _local_search(v, k, t, child, time_limit=5.0)
        
    return child

def construct(v, k, t):
    """Main entry point."""
    start_time = time.time()
    time_limit = 115.0
    
    # Pre-compute
    all_blocks, block_masks, subset_to_idx, num_subsets = _get_precomputed(v, k, t)
    
    # Initial population
    # 1. Greedy
    greedy1 = _greedy_cover(v, k, t, time_limit=10.0, seed=42)
    
    # 2. Random + Repair
    random_blocks = [list(random.sample(range(v), k)) for _ in range(100)]
    random1 = _reduce_and_repair(v, k, t, random_blocks, time_limit=10.0)
    
    population = [greedy1, random1]
    
    # Add diverse individuals
    for i in range(4):
        rand_blocks = [list(random.sample(range(v), k)) for _ in range(50)]
        pop_ind = _reduce_and_repair(v, k, t, rand_blocks, time_limit=5.0)
        population.append(pop_ind)
        
    # Initial Local Search on best
    best_individual = min(population, key=len)
    best_individual = _local_search(v, k, t, best_individual, time_limit=10.0)
    
    while time.time() - start_time < time_limit:
        # Sort
        population.sort(key=len)
        
        # Elitism
        elite = population[:max(2, len(population)//2)]
        
        new_population = list(elite)
        
        # Generate children
        while len(new_population) < len(population):
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t)
            
            # Mutation: Replace a random block with a random one, then repair
            if random.random() < 0.3 and child:
                idx = random.randint(0, len(child)-1)
                child[idx] = list(random.sample(range(v), k))
                child = _reduce_and_repair(v, k, t, child, time_limit=5.0)
                child = _local_search(v, k, t, child, time_limit=5.0)
            
            new_population.append(child)
            
        population = new_population
        current_best = min(population, key=len)
        
        if len(current_best) < len(best_individual):
            best_individual = current_best
            # Optional: run local search on the new best
            best_individual = _local_search(v, k, t, best_individual, time_limit=5.0)
            if len(best_individual) <= 151:
                break
                
    # Final repair and reduction
    final_blocks = _reduce_and_repair(v, k, t, best_individual, time_limit=5.0)
    final_blocks = _local_search(v, k, t, final_blocks, time_limit=5.0)
    
    # Validate
    # Check if all subsets are covered
    current_union = 0
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    for b in final_blocks:
        idx = block_to_idx.get(tuple(b))
        if idx is not None:
            current_union |= block_masks[idx]
            
    full_mask = (1 << num_subsets) - 1
    if current_union != full_mask:
        # Fallback
        final_blocks = _greedy_cover(v, k, t, time_limit=110.0)
        
    return final_blocks
# EVOLVE-BLOCK-END

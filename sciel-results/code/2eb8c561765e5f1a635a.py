"""Genetic algorithm covering design C(19,7,4) with custom crossover."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict

# Global cache for precomputed data
_CACHE = {}

def _get_precomputed(v, k, t):
    """Get or compute precomputed data for the given parameters."""
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map from frozenset to index for O(1) lookup
    block_to_idx = {frozenset(b): i for i, b in enumerate(all_blocks)}
    
    # Build coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    # Using sparse representation might be faster for large v, but for v=19, dense is fine.
    # Actually, let's use a list of sets of indices for faster union/intersection if needed,
    # but numpy boolean arrays are very fast for bitwise ops.
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.int8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
    
    # Store block tuples for easy access
    block_tuples = [tuple(b) for b in all_blocks]
    
    data = {
        'all_subsets': all_subsets,
        't_index_map': t_index_map,
        'num_subsets': num_subsets,
        'all_blocks': all_blocks,
        'block_tuples': block_tuples,
        'coverage_matrix': coverage_matrix,
        'num_blocks': num_blocks,
        'block_to_idx': block_to_idx,
    }
    
    _CACHE[key] = data
    return data

def _greedy_cover(v, k, t, time_limit=15.0, randomize=False):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    num_blocks = data['num_blocks']
    
    uncovered = np.ones(num_subsets, dtype=np.int8)
    blocks = []
    used_blocks = set()
    
    while uncovered.any():
        gains = coverage_matrix @ uncovered
        # Mask out already used blocks to avoid duplicates in the list
        # Actually, we don't strictly need to avoid duplicates in the *list* if we check coverage,
        # but it's cleaner to avoid. However, adding a block that doesn't add new coverage is wasteful.
        # We just pick the best available.
        
        if randomize:
            # Add small noise to break ties randomly
            noise = np.random.uniform(0, 0.1, num_blocks)
            gains = gains + noise
        
        best_idx = int(np.argmax(gains))
        
        if gains[best_idx] == 0:
            break
        
        blocks.append(list(data['block_tuples'][best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        used_blocks.add(best_idx)
        
        if time.time() - start_time > time_limit:
            break
    
    return blocks

def _repair(v, k, t, blocks, time_limit=10.0):
    """Repair a list of blocks to ensure full coverage, then reduce."""
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    block_to_idx = data['block_to_idx']
    
    # Normalize blocks to tuples for lookup
    normalized_blocks = []
    for b in blocks:
        b_tuple = tuple(sorted(b))
        if b_tuple in block_to_idx:
            normalized_blocks.append(b_tuple)
        # If a block is invalid (duplicates etc), it's skipped, but input should be valid
    
    # Build current coverage mask
    total_mask = np.zeros(num_subsets, dtype=bool)
    block_indices = []
    for b_tuple in normalized_blocks:
        bi = block_to_idx[b_tuple]
        block_indices.append(bi)
        total_mask[coverage_matrix[bi] == 1] = True
    
    uncovered = ~total_mask
    
    # Add blocks to cover remaining gaps
    if uncovered.any():
        uncovered_int = uncovered.astype(np.int8)
        while uncovered.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered_int
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            # Append new block
            block_indices.append(best_idx)
            total_mask[coverage_matrix[best_idx] == 1] = True
            uncovered = ~total_mask
            uncovered_int = uncovered.astype(np.int8)
    
    # Reduction phase: iteratively remove redundant blocks
    if len(block_indices) <= 1:
        return [list(data['block_tuples'][bi]) for bi in block_indices]
    
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_block_indices = []
        for i, bi in enumerate(block_indices):
            # Check if removing this block still covers everything
            if len(block_indices) <= 1:
                new_block_indices.append(bi)
                continue
            
            # Efficient check: count how many subsets are covered ONLY by this block
            # If count == 0, it's redundant.
            # mask_without = total_mask & ~coverage_matrix[bi]
            # If mask_without.all(), it's redundant.
            
            # Optimized: 
            # subsets_covered_by_bi = coverage_matrix[bi]
            # subsets_covered_by_others = total_mask & ~subsets_covered_by_bi
            # if not (subsets_covered_by_bi & ~total_mask).any() -> wait, total_mask includes bi.
            
            # Let's just compute the mask without bi.
            # Since we are in a loop, doing this O(N^2 * M) might be slow.
            # But N is small (~150). M is ~7752. 150*150*7752 ~ 175M ops. Might be tight in Python/NumPy if not vectorized well.
            
            # Vectorized approach for redundancy check:
            # Create a count array? No, that's O(N*M) to build.
            # Let's stick to the simple boolean mask check but optimize the inner loop.
            
            # Actually, a better way: 
            # If we maintain a count of how many times each subset is covered, we can just decrement.
            # But we are rebuilding the list.
            
            # Let's use the property: A block is redundant if for every subset it covers, 
            # at least one other block in the set also covers it.
            
            # Simple check:
            mask_without = np.zeros(num_subsets, dtype=bool)
            for j, bj in enumerate(block_indices):
                if j != i:
                    mask_without[coverage_matrix[bj] == 1] = True
            
            if mask_without.all():
                changed = True
            else:
                new_block_indices.append(bi)
        
        if changed:
            block_indices = new_block_indices
    
    # Reconstruct blocks
    unique_indices = list(dict.fromkeys(block_indices))
    result = [list(data['block_tuples'][bi]) for bi in unique_indices]
    
    return result

def _crossover(parent1, parent2, v, k, t, time_limit=10.0):
    """Custom crossover: merge two coverings by greedy selection from union."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    block_to_idx = data['block_to_idx']
    
    # Build union of blocks from both parents
    seen = set()
    pool_indices = []
    for b in parent1 + parent2:
        b_tuple = tuple(sorted(b))
        if b_tuple in block_to_idx:
            bi = block_to_idx[b_tuple]
            if bi not in seen:
                seen.add(bi)
                pool_indices.append(bi)
    
    # Greedy selection from pool
    uncovered = np.ones(num_subsets, dtype=np.int8)
    child_blocks = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for all available blocks
        best_idx = -1
        best_gain = -1
        best_pos = -1
        
        # Optimization: If available is small, loop is fine. If large, vectorize.
        # For now, loop is simple and safe.
        for pos, bi in enumerate(available):
            gain = int(np.sum(coverage_matrix[bi] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_idx = bi
                best_pos = pos
        
        if best_idx == -1 or best_gain == 0:
            break
        
        child_blocks.append(list(data['block_tuples'][best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        available.pop(best_pos)
    
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = _repair(v, k, t, child_blocks, time_limit=time_limit)
    
    return child_blocks

def _local_search(v, k, t, blocks, time_limit=8.0):
    """Local search: try removing blocks and repairing to find smaller coverings."""
    start_time = time.time()
    current_blocks = [list(b) for b in blocks]
    current_len = len(current_blocks)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try removing each block and repairing
        for i in range(len(current_blocks)):
            if time.time() - start_time >= time_limit:
                break
            
            candidate = current_blocks[:i] + current_blocks[i+1:]
            candidate = _repair(v, k, t, candidate, time_limit=2.0)
            
            if len(candidate) < len(current_blocks):
                current_blocks = candidate
                current_len = len(current_blocks)
                improved = True
                break
        
        # 2. Try swapping blocks (if removal didn't work)
        if not improved:
            # Try swapping one block with a random one
            for _ in range(5):
                if time.time() - start_time >= time_limit or len(current_blocks) < 2:
                    break
                
                idx_to_swap = random.randint(0, len(current_blocks) - 1)
                original_block = current_blocks[idx_to_swap]
                
                # Pick a random block from the entire space
                new_block = sorted(random.sample(range(v), k))
                
                if new_block == original_block:
                    continue
                
                candidate = [list(b) for b in current_blocks]
                candidate[idx_to_swap] = new_block
                
                # Repair might add blocks, so we check if the *reduced* size is smaller
                candidate = _repair(v, k, t, candidate, time_limit=2.0)
                
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    current_len = len(current_blocks)
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_blocks

def construct(v, k, t):
    """Main entry point: genetic algorithm with custom crossover."""
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute
    data = _get_precomputed(v, k, t)
    
    # Initial population: diverse starting points
    population = []
    
    # 1. Standard greedy
    greedy1 = _greedy_cover(v, k, t, time_limit=10.0)
    population.append(greedy1)
    
    # 2. Randomized Greedy
    for _ in range(4):
        greedy_r = _greedy_cover(v, k, t, time_limit=5.0, randomize=True)
        population.append(greedy_r)
    
    # 3. Random + Repair (to get diverse, potentially worse but structurally different solutions)
    for _ in range(2):
        # Start with a smaller random set to force repair to work hard
        num_rand_blocks = 50
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(num_rand_blocks)]
        rand_repaired = _repair(v, k, t, rand_blocks, time_limit=5.0)
        population.append(rand_repaired)
    
    best_individual = min(population, key=len)
    best_len = len(best_individual)
    
    # Genetic algorithm loop
    pop_size = 10
    population = population[:pop_size]
    
    while time.time() - start_time < time_limit - 20:
        # Sort by size
        population.sort(key=len)
        
        # Elitism: keep top half
        elite_count = max(2, len(population) // 2)
        elite = population[:elite_count]
        
        new_population = list(elite)
        
        # Generate children via crossover
        while len(new_population) < pop_size:
            if time.time() - start_time >= time_limit - 10:
                break
            
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t, time_limit=5.0)
            
            # Mutation: occasionally replace a block with a random one
            if random.random() < 0.2 and child:
                idx = random.randint(0, len(child) - 1)
                child[idx] = sorted(random.sample(range(v), k))
                child = _repair(v, k, t, child, time_limit=3.0)
            
            # Local search
            if time.time() - start_time < time_limit - 10:
                child = _local_search(v, k, t, child, time_limit=4.0)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=len)
        
        if len(current_best) < best_len:
            best_individual = current_best
            best_len = len(best_individual)
            # Early stop if we hit the known lower bound or very close
            if best_len <= 151:
                break
    
    # Final repair and validation
    final_blocks = _repair(v, k, t, best_individual, time_limit=10.0)
    
    # Validate coverage
    all_subsets = data['all_subsets']
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
    
    if len(covered) != len(all_subsets):
        # Fallback to greedy
        final_blocks = _greedy_cover(v, k, t, time_limit=110.0)
    
    return final_blocks
# EVOLVE-BLOCK-END

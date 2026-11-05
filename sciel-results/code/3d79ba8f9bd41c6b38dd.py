"""Genetic algorithm covering design C(19,7,4) with optimized GA and local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

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
    
    # Build coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.int8)
    block_subsets = [] # List of lists of subset indices for each block
    for i, block in enumerate(all_blocks):
        subs = []
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
            subs.append(idx)
        block_subsets.append(subs)
    
    # Store block tuples for easy access
    block_tuples = [tuple(b) for b in all_blocks]
    
    # Map block tuple to index for O(1) lookup
    block_to_idx = {bt: i for i, bt in enumerate(block_tuples)}
    
    data = {
        'all_subsets': all_subsets,
        't_index_map': t_index_map,
        'num_subsets': num_subsets,
        'all_blocks': all_blocks,
        'block_tuples': block_tuples,
        'coverage_matrix': coverage_matrix,
        'num_blocks': num_blocks,
        'block_subsets': block_subsets,
        'block_to_idx': block_to_idx,
    }
    
    _CACHE[key] = data
    return data

def _greedy_cover(v, k, t, time_limit=15.0):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    uncovered = np.ones(num_subsets, dtype=np.int8)
    blocks = []
    
    while uncovered.any():
        gains = coverage_matrix @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        
        blocks.append(list(data['block_tuples'][best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        
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
    block_subsets = data['block_subsets']
    
    # Map blocks to indices, handling duplicates
    block_indices = []
    seen = set()
    for b in blocks:
        bt = tuple(sorted(b))
        if bt in block_to_idx:
            bi = block_to_idx[bt]
            if bi not in seen:
                block_indices.append(bi)
                seen.add(bi)
    
    if not block_indices:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    # Calculate current coverage counts
    cover_count = np.zeros(num_subsets, dtype=np.int8)
    for bi in block_indices:
        for si in block_subsets[bi]:
            cover_count[si] += 1
    
    uncovered = (cover_count == 0)
    
    # Add blocks to cover remaining gaps
    if uncovered.any():
        uncovered_int = uncovered.astype(np.int8)
        while uncovered.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered_int
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            block_indices.append(best_idx)
            seen.add(best_idx)
            
            for si in block_subsets[best_idx]:
                cover_count[si] += 1
            
            uncovered = (cover_count == 0)
            if uncovered.any():
                uncovered_int = uncovered.astype(np.int8)
            else:
                uncovered_int = None
    
    # Reduction phase: iteratively remove redundant blocks
    # A block is redundant if all subsets it covers have count > 1
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_block_indices = []
        for bi in block_indices:
            # Check if any subset covered by bi has count 1
            is_redundant = True
            for si in block_subsets[bi]:
                if cover_count[si] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for si in block_subsets[bi]:
                    cover_count[si] -= 1
                changed = True
            else:
                new_block_indices.append(bi)
        
        block_indices = new_block_indices
    
    # Reconstruct blocks
    result = [list(data['block_tuples'][bi]) for bi in block_indices]
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
        bt = tuple(sorted(b))
        if bt in block_to_idx:
            bi = block_to_idx[bt]
            if bi not in seen:
                seen.add(bi)
                pool_indices.append(bi)
    
    if not pool_indices:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    # Greedy selection from pool
    uncovered = np.ones(num_subsets, dtype=np.int8)
    child_blocks = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for all available blocks
        # To speed up, we can just pick the max from the available subset
        if len(available) < 50:
             # Small pool: check all
             best_idx = -1
             best_gain = -1
             best_pos = -1
             for pos, bi in enumerate(available):
                 gain = int(np.sum(coverage_matrix[bi] & uncovered))
                 if gain > best_gain:
                     best_gain = gain
                     best_idx = bi
                     best_pos = pos
        else:
             # Large pool: use matrix multiplication on the subset of rows
             # This is faster than iterating in Python if pool is large
             sub_matrix = coverage_matrix[available]
             gains = sub_matrix @ uncovered
             best_pos = int(np.argmax(gains))
             best_gain = gains[best_pos]
             best_idx = available[best_pos]
        
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
    current_size = len(current_blocks)
    
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
                current_size = len(candidate)
                improved = True
                break
        
        # 2. Try swapping a block with a random one if it reduces size after repair
        if not improved:
            for _ in range(5):
                if time.time() - start_time >= time_limit or not current_blocks:
                    break
                
                idx_to_remove = random.randint(0, len(current_blocks) - 1)
                candidate = current_blocks[:idx_to_remove] + current_blocks[idx_to_remove+1:]
                
                # Add a random block
                random_block = sorted(random.sample(range(v), k))
                candidate.append(random_block)
                
                candidate = _repair(v, k, t, candidate, time_limit=2.0)
                
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    current_size = len(candidate)
                    improved = True
                    break
            
            if not improved:
                # Try a few random removals without adding (pure reduction attempt)
                for _ in range(3):
                    if time.time() - start_time >= time_limit or not current_blocks:
                        break
                    idx_to_remove = random.randint(0, len(current_blocks) - 1)
                    candidate = current_blocks[:idx_to_remove] + current_blocks[idx_to_remove+1:]
                    candidate = _repair(v, k, t, candidate, time_limit=2.0)
                    
                    if len(candidate) < len(current_blocks):
                        current_blocks = candidate
                        current_size = len(candidate)
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
    greedy1 = _greedy_cover(v, k, t, time_limit=15.0)
    population.append(greedy1)
    
    # 2. Greedy with random tie-breaking (run multiple times)
    for _ in range(3):
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0)
        if greedy_r:
            # Perturb: replace a few blocks with random ones, then repair
            perturbed = [list(b) for b in greedy_r]
            num_perturb = max(1, len(perturbed) // 10)
            for _ in range(num_perturb):
                idx = random.randint(0, len(perturbed) - 1)
                perturbed[idx] = sorted(random.sample(range(v), k))
            perturbed = _repair(v, k, t, perturbed, time_limit=10.0)
            population.append(perturbed)
    
    # 3. Random + repair
    for _ in range(3):
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(80)]
        rand_repaired = _repair(v, k, t, rand_blocks, time_limit=10.0)
        population.append(rand_repaired)
    
    best_individual = min(population, key=len)
    
    # Genetic algorithm loop
    pop_size = 8
    population = population[:pop_size]
    
    while time.time() - start_time < time_limit - 15:
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
            
            child = _crossover(p1, p2, v, k, t, time_limit=8.0)
            
            # Mutation: occasionally replace a block with a random one
            if random.random() < 0.3 and child:
                idx = random.randint(0, len(child) - 1)
                child[idx] = sorted(random.sample(range(v), k))
                child = _repair(v, k, t, child, time_limit=8.0)
            
            # Local search
            if time.time() - start_time < time_limit - 10:
                child = _local_search(v, k, t, child, time_limit=5.0)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=len)
        
        if len(current_best) < len(best_individual):
            best_individual = current_best
            if len(best_individual) <= 151:
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

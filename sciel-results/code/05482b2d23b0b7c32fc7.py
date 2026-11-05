"""Genetic algorithm covering design C(19,7,4) with bitmask-based operations."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

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
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage for each block as a set of subset indices
    block_coverage = []
    for block in all_blocks:
        cov = set()
        for combo in itertools.combinations(block, t):
            cov.add(t_index_map[combo])
        block_coverage.append(cov)
    
    # Convert to numpy arrays for fast operations
    # Use a 2D boolean matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.bool_)
    for i, cov in enumerate(block_coverage):
        for idx in cov:
            coverage_matrix[i, idx] = True
    
    block_tuples = [tuple(b) for b in all_blocks]
    
    # Precompute block index lookup
    block_idx_map = {frozenset(b): i for i, b in enumerate(all_blocks)}
    
    data = {
        'all_subsets': all_subsets,
        't_index_map': t_index_map,
        'num_subsets': num_subsets,
        'all_blocks': all_blocks,
        'block_tuples': block_tuples,
        'coverage_matrix': coverage_matrix,
        'num_blocks': num_blocks,
        'block_idx_map': block_idx_map,
    }
    
    _CACHE[key] = data
    return data

def _block_to_idx(block, data):
    """Convert a block to its index in all_blocks."""
    fs = frozenset(block)
    return data['block_idx_map'].get(fs, -1)

def _greedy_cover(v, k, t, time_limit=15.0, random_seed=None):
    """Fast greedy covering with optional random tie-breaking."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    num_blocks = data['num_blocks']
    
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random
    
    uncovered = np.ones(num_subsets, dtype=np.bool_)
    blocks = []
    used_blocks = set()
    
    while uncovered.any() and time.time() - start_time < time_limit:
        gains = coverage_matrix @ uncovered.astype(np.int8)
        
        # Find max gain
        max_gain = gains.max()
        if max_gain == 0:
            break
        
        # Get all blocks with max gain for tie-breaking
        candidates = np.where(gains == max_gain)[0]
        
        # Filter out already used blocks
        available = [c for c in candidates if c not in used_blocks]
        if not available:
            # If all max-gain blocks are used, take next best
            gains2 = gains.copy()
            for c in candidates:
                if c in used_blocks:
                    gains2[c] = -1
            max_gain2 = gains2.max()
            if max_gain2 <= 0:
                break
            candidates = np.where(gains2 == max_gain2)[0]
            available = [c for c in candidates if c not in used_blocks]
            if not available:
                break
        
        best_idx = rng.choice(available)
        blocks.append(best_idx)
        used_blocks.add(best_idx)
        uncovered[coverage_matrix[best_idx]] = False
        
    # Convert to block lists
    return [list(data['block_tuples'][bi]) for bi in blocks]

def _repair(v, k, t, blocks, time_limit=10.0):
    """Repair a list of blocks to ensure full coverage, then reduce."""
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Build current coverage mask
    total_mask = np.zeros(num_subsets, dtype=np.bool_)
    block_indices = []
    for b in blocks:
        bi = _block_to_idx(b, data)
        if bi >= 0:
            block_indices.append(bi)
            total_mask[coverage_matrix[bi]] = True
    
    uncovered = ~total_mask
    
    # Add blocks to cover remaining gaps
    if uncovered.any():
        while uncovered.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered.astype(np.int8)
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            block_indices.append(best_idx)
            total_mask[coverage_matrix[best_idx]] = True
            uncovered = ~total_mask
    
    # Reduction phase: iteratively remove redundant blocks
    if len(block_indices) <= 1:
        return [list(data['block_tuples'][bi]) for bi in block_indices]
    
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_indices = []
        for i, bi in enumerate(block_indices):
            if len(block_indices) <= 1:
                new_indices.append(bi)
                continue
            
            # Check if removing this block still covers everything
            mask_without = np.zeros(num_subsets, dtype=np.bool_)
            for j, bj in enumerate(block_indices):
                if j != i:
                    mask_without[coverage_matrix[bj]] = True
            
            if mask_without.all():
                changed = True
            else:
                new_indices.append(bi)
        
        block_indices = new_indices
    
    # Remove duplicates
    unique_indices = list(dict.fromkeys(block_indices))
    return [list(data['block_tuples'][bi]) for bi in unique_indices]

def _crossover(parent1, parent2, v, k, t, time_limit=8.0):
    """Custom crossover: merge two coverings by greedy selection from union."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Build union of block indices from both parents
    seen = set()
    pool_indices = []
    for b in parent1 + parent2:
        bi = _block_to_idx(b, data)
        if bi >= 0 and bi not in seen:
            seen.add(bi)
            pool_indices.append(bi)
    
    # Greedy selection from pool
    uncovered = np.ones(num_subsets, dtype=np.bool_)
    child_indices = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        best_gain = -1
        best_pos = -1
        
        for pos, bi in enumerate(available):
            gain = int(np.sum(coverage_matrix[bi] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
        
        if best_gain <= 0:
            break
        
        child_indices.append(available[best_pos])
        uncovered[coverage_matrix[available[best_pos]]] = False
        available.pop(best_pos)
    
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = [list(data['block_tuples'][bi]) for bi in child_indices]
        child_blocks = _repair(v, k, t, child_blocks, time_limit=time_limit)
        # Convert back to indices
        child_indices = []
        for b in child_blocks:
            bi = _block_to_idx(b, data)
            if bi >= 0:
                child_indices.append(bi)
    
    return [list(data['block_tuples'][bi]) for bi in child_indices]

def _local_search(v, k, t, blocks, time_limit=8.0):
    """Local search: try removing or swapping blocks to find smaller coverings."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    num_blocks = data['num_blocks']
    
    current_indices = []
    for b in blocks:
        bi = _block_to_idx(b, data)
        if bi >= 0:
            current_indices.append(bi)
    
    if not current_indices:
        return blocks
    
    best_size = len(current_indices)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try removing each block
        for i in range(len(current_indices)):
            if time.time() - start_time >= time_limit:
                break
            
            candidate = current_indices[:i] + current_indices[i+1:]
            if not candidate:
                continue
            
            # Check if candidate covers everything
            mask = np.zeros(num_subsets, dtype=np.bool_)
            for bi in candidate:
                mask[coverage_matrix[bi]] = True
            
            if mask.all():
                current_indices = candidate
                best_size = len(current_indices)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping one block with another
        if len(current_indices) > 0:
            for i in range(len(current_indices)):
                if time.time() - start_time >= time_limit:
                    break
                
                # Try replacing block i with a random block
                for _ in range(5):
                    new_bi = random.randint(0, num_blocks - 1)
                    if new_bi in current_indices:
                        continue
                    
                    candidate = current_indices[:i] + [new_bi] + current_indices[i+1:]
                    
                    # Check coverage
                    mask = np.zeros(num_subsets, dtype=np.bool_)
                    for bi in candidate:
                        mask[coverage_matrix[bi]] = True
                    
                    if mask.all():
                        current_indices = candidate
                        best_size = len(current_indices)
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            # Try random perturbations: remove a random block and repair
            if len(current_indices) > 1:
                idx_to_remove = random.randint(0, len(current_indices) - 1)
                candidate = current_indices[:idx_to_remove] + current_indices[idx_to_remove+1:]
                candidate_blocks = [list(data['block_tuples'][bi]) for bi in candidate]
                repaired = _repair(v, k, t, candidate_blocks, time_limit=3.0)
                repaired_indices = []
                for b in repaired:
                    bi = _block_to_idx(b, data)
                    if bi >= 0:
                        repaired_indices.append(bi)
                
                if len(repaired_indices) < best_size:
                    current_indices = repaired_indices
                    best_size = len(current_indices)
                    improved = True
            
            if not improved:
                break
    
    return [list(data['block_tuples'][bi]) for bi in current_indices]

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
    
    # 2. Greedy with different random seeds for diversity
    for seed in [42, 123, 456, 789]:
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0, random_seed=seed)
        population.append(greedy_r)
    
    # 3. Random + repair
    for _ in range(3):
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(80)]
        rand_repaired = _repair(v, k, t, rand_blocks, time_limit=10.0)
        population.append(rand_repaired)
    
    best_individual = min(population, key=len)
    
    # Genetic algorithm loop
    pop_size = 10
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

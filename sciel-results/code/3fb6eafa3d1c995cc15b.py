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
    
    # Build coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    # Using int8 for memory efficiency, but we will use boolean views for speed
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.int8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
    
    # Store block tuples for easy access
    block_tuples = [tuple(b) for b in all_blocks]
    
    # Create a fast lookup: frozenset -> index
    # This is crucial for O(1) block identification
    block_to_idx = {frozenset(b): i for i, b in enumerate(all_blocks)}
    
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

def _fast_greedy_cover(v, k, t, time_limit=10.0, randomize=False):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    num_blocks = data['num_blocks']
    block_tuples = data['block_tuples']
    
    uncovered = np.ones(num_subsets, dtype=bool)
    selected_indices = []
    
    # Precompute gains? No, uncovered changes.
    # We need to find the block that covers the most uncovered subsets.
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        # Calculate gains for all blocks
        # gains[i] = number of uncovered subsets covered by block i
        # Using matrix multiplication is fast if uncovered is dense, but here it's sparse initially?
        # Actually, uncovered starts all 1s.
        # gains = coverage_matrix @ uncovered (if uncovered is float)
        # But coverage_matrix is int8. 
        # Let's use a vectorized approach.
        
        # Optimization: If we have many blocks, calculating gains for all is O(N_blocks * N_subsets).
        # N_blocks ~ 50k, N_subsets ~ 3876. 50k * 3.8k = 190M ops. This might be slow in Python loop if not vectorized.
        # np.dot is vectorized.
        
        # Convert uncovered to float for dot product? Or use int?
        # gains = np.sum(coverage_matrix * uncovered[:, None], axis=1) -> This creates a huge intermediate array.
        # Better: gains = coverage_matrix @ uncovered.astype(float)
        
        uncovered_float = uncovered.astype(np.float32)
        gains = coverage_matrix @ uncovered_float
        
        if randomize:
            # Add small noise to break ties randomly
            gains += np.random.uniform(0, 0.1, size=num_blocks)
        
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        
        selected_indices.append(best_idx)
        uncovered[coverage_matrix[best_idx] == 1] = False
        
    return [list(block_tuples[i]) for i in selected_indices]

def _repair_and_reduce(v, k, t, blocks, time_limit=5.0):
    """
    1. Ensure all subsets are covered by adding necessary blocks.
    2. Reduce the set by removing redundant blocks.
    """
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    block_to_idx = data['block_to_idx']
    block_tuples = data['block_tuples']
    
    # Convert input blocks to indices
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(frozenset(b), -1)
        if idx != -1:
            block_indices.append(idx)
    
    # Remove duplicates
    block_indices = list(dict.fromkeys(block_indices))
    
    # 1. Repair: Check coverage
    if not block_indices:
        # Fallback to greedy
        return _fast_greedy_cover(v, k, t, time_limit)
        
    # Build total coverage mask
    # Instead of summing, we can just OR the rows
    # But we need to know what is missing.
    total_mask = np.zeros(num_subsets, dtype=bool)
    for bi in block_indices:
        total_mask |= (coverage_matrix[bi] == 1)
    
    missing = ~total_mask
    
    if missing.any():
        # Greedily add blocks to cover missing
        missing_float = missing.astype(np.float32)
        while missing.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ missing_float
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            block_indices.append(best_idx)
            total_mask |= (coverage_matrix[best_idx] == 1)
            missing = ~total_mask
            missing_float = missing.astype(np.float32)
            
    # 2. Reduction: Remove redundant blocks
    # A block is redundant if all subsets it covers are covered by other blocks.
    # Strategy: Iterate through blocks. For each, check if removing it leaves any subset uncovered.
    # To speed this up, we can count how many blocks cover each subset.
    # If a subset is covered by only 1 block, that block is "critical" and cannot be removed.
    # We only need to check redundancy for blocks that do NOT cover any critical subset.
    
    if len(block_indices) > 1:
        # Count coverage per subset
        # coverage_count[j] = number of blocks in block_indices that cover subset j
        # We can compute this by summing the rows of coverage_matrix for selected blocks.
        # selected_mask = np.zeros(num_blocks, dtype=bool)
        # selected_mask[block_indices] = True
        # coverage_count = np.sum(coverage_matrix[selected_mask], axis=0)
        
        # This is O(N_selected * N_subsets). 
        # If N_selected is ~150, and N_subsets ~4000, it's 600k ops. Very fast.
        
        selected_mask = np.zeros(data['num_blocks'], dtype=bool)
        for bi in block_indices:
            selected_mask[bi] = True
            
        coverage_count = np.sum(coverage_matrix[selected_mask], axis=0)
        
        # Identify critical subsets (covered by exactly 1 block)
        critical_subsets = (coverage_count == 1)
        
        # Identify critical blocks (blocks that cover at least one critical subset)
        # A block is critical if it covers any subset where coverage_count == 1.
        # critical_blocks = np.any(coverage_matrix[:, critical_subsets] == 1, axis=1)
        # But we only care about blocks in our set.
        
        # Let's just try to remove each non-critical block.
        # To find non-critical blocks:
        # A block bi is non-critical if it doesn't cover any critical subset.
        # i.e., sum(coverage_matrix[bi, critical_subsets]) == 0
        
        new_indices = []
        for bi in block_indices:
            # Check if this block is critical
            # If it covers any critical subset, it MUST stay.
            # critical_subsets is a boolean mask of size num_subsets.
            # block_coverage = coverage_matrix[bi]
            # is_critical = np.any(block_coverage[critical_subsets])
            
            # Optimization: 
            # If coverage_count[subset] > 1 for all subsets covered by bi, then bi is not critical.
            # We can check: np.all(coverage_count[coverage_matrix[bi] == 1] > 1)
            
            covered_by_bi = coverage_matrix[bi] == 1
            if np.all(coverage_count[covered_by_bi] > 1):
                # This block is potentially redundant.
                # We need to verify that removing it doesn't uncover anything.
                # Actually, if coverage_count > 1 for all its subsets, then by definition,
                # every subset it covers is covered by at least one OTHER block.
                # So it is redundant!
                continue
            else:
                new_indices.append(bi)
        
        block_indices = new_indices
        
    return [list(block_tuples[i]) for i in block_indices]

def _crossover(parent1, parent2, v, k, t, time_limit=5.0):
    """Custom crossover: merge two coverings by greedy selection from union."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    block_to_idx = data['block_to_idx']
    block_tuples = data['block_tuples']
    
    # Build union of block indices
    seen = set()
    pool_indices = []
    for b in parent1 + parent2:
        idx = block_to_idx.get(frozenset(b), -1)
        if idx != -1 and idx not in seen:
            seen.add(idx)
            pool_indices.append(idx)
    
    if not pool_indices:
        return _fast_greedy_cover(v, k, t, time_limit)
        
    # Greedy selection from pool
    uncovered = np.ones(num_subsets, dtype=bool)
    child_indices = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for available blocks only
        # To avoid full matrix mult, we can iterate if pool is small, or use mask.
        # Pool size is at most len(p1)+len(p2) ~ 300.
        # N_subsets ~ 4000.
        # 300 * 4000 = 1.2M ops. Fast enough.
        
        best_gain = -1
        best_pos = -1
        best_idx = -1
        
        # Vectorized gain calculation for available blocks
        # Create a matrix of available blocks
        # avail_mask = np.zeros(data['num_blocks'], dtype=bool)
        # avail_mask[available] = True
        # avail_matrix = coverage_matrix[avail_mask] # This copies data, might be slow.
        
        # Let's just loop over available indices, it's small.
        for pos, bi in enumerate(available):
            gain = np.sum(coverage_matrix[bi] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = bi
                best_pos = pos
        
        if best_gain == 0:
            break
        
        child_indices.append(best_idx)
        uncovered[coverage_matrix[best_idx] == 1] = False
        available.pop(best_pos)
    
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = [list(block_tuples[i]) for i in child_indices]
        child_blocks = _repair_and_reduce(v, k, t, child_blocks, time_limit)
        return child_blocks
    else:
        # Reduce
        child_blocks = [list(block_tuples[i]) for i in child_indices]
        return _repair_and_reduce(v, k, t, child_blocks, time_limit)

def _local_search(v, k, t, blocks, time_limit=5.0):
    """
    Local search: Try to remove blocks or swap blocks to reduce size.
    """
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    block_to_idx = data['block_to_idx']
    block_tuples = data['block_tuples']
    
    current_indices = []
    for b in blocks:
        idx = block_to_idx.get(frozenset(b), -1)
        if idx != -1:
            current_indices.append(idx)
            
    current_indices = list(dict.fromkeys(current_indices))
    
    if not current_indices:
        return blocks
        
    # Compute coverage counts for current set
    selected_mask = np.zeros(data['num_blocks'], dtype=bool)
    for bi in current_indices:
        selected_mask[bi] = True
    coverage_count = np.sum(coverage_matrix[selected_mask], axis=0)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try removing any non-critical block
        # A block is removable if all subsets it covers have coverage_count > 1.
        for i, bi in enumerate(current_indices):
            covered_by_bi = coverage_matrix[bi] == 1
            if np.all(coverage_count[covered_by_bi] > 1):
                # Remove it
                current_indices.pop(i)
                # Update coverage counts
                coverage_count[covered_by_bi] -= 1
                improved = True
                break # Restart loop to re-evaluate other blocks
                
        if improved:
            continue
            
        # 2. Try swapping: Remove block i, add block j that covers the gaps.
        # This is expensive. We only do a limited search.
        # Find a block that covers many "critical" subsets of the current set?
        # Actually, just try random swaps if stuck?
        
        # Let's try: Pick a random block to remove, see what becomes uncovered.
        # Then try to find a single block that covers all uncovered subsets? Unlikely.
        # Or find a block that covers the most uncovered subsets, add it, and see if total size decreases?
        # Size decreases only if we remove 1 and add 0 (impossible if gap) or remove 2 and add 1.
        
        # Let's stick to removal for speed, as the reduction phase in repair is similar.
        # Maybe try a "2-for-1" swap?
        # Pick two blocks i, j. Remove them. See if one block k can cover all gaps?
        
        # Given time constraints, simple removal is often sufficient after a good greedy/crossover.
        # Let's add a small random perturbation if stuck to escape local minima.
        if random.random() < 0.1 and current_indices:
            # Remove a random block, repair, and keep if smaller
            idx_to_remove = random.randint(0, len(current_indices) - 1)
            candidate_indices = current_indices[:idx_to_remove] + current_indices[idx_to_remove+1:]
            candidate_blocks = [list(block_tuples[i]) for i in candidate_indices]
            repaired = _repair_and_reduce(v, k, t, candidate_blocks, time_limit=2.0)
            if len(repaired) < len(current_indices):
                current_indices = []
                for b in repaired:
                    idx = block_to_idx.get(frozenset(b), -1)
                    if idx != -1:
                        current_indices.append(idx)
                improved = True

    return [list(block_tuples[i]) for i in current_indices]

def construct(v, k, t):
    """Main entry point: genetic algorithm with custom crossover."""
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute
    data = _get_precomputed(v, k, t)
    
    # Initial population
    population = []
    
    # 1. Standard greedy
    greedy1 = _fast_greedy_cover(v, k, t, time_limit=10.0, randomize=False)
    population.append(greedy1)
    
    # 2. Greedy with randomization
    for _ in range(3):
        greedy_r = _fast_greedy_cover(v, k, t, time_limit=10.0, randomize=True)
        population.append(greedy_r)
    
    # 3. Random start + repair
    for _ in range(3):
        # Start with a small random set
        rand_blocks = [list(sorted(random.sample(range(v), k))) for _ in range(50)]
        repaired = _repair_and_reduce(v, k, t, rand_blocks, time_limit=10.0)
        population.append(repaired)
        
    # Sort initial population
    population.sort(key=len)
    
    best_individual = population[0]
    
    # Genetic Algorithm
    pop_size = 10
    # Ensure we have enough individuals
    while len(population) < pop_size:
        # Duplicate best with mutation
        base = random.choice(population)
        mutated = [list(b) for b in base]
        if mutated:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = sorted(random.sample(range(v), k))
        mutated = _repair_and_reduce(v, k, t, mutated, time_limit=5.0)
        population.append(mutated)
        
    population = population[:pop_size]
    
    while time.time() - start_time < time_limit - 10:
        # Sort
        population.sort(key=len)
        
        # Elitism
        elite_count = max(2, pop_size // 3)
        elite = population[:elite_count]
        
        new_population = list(elite)
        
        # Generate children
        while len(new_population) < pop_size and time.time() - start_time < time_limit - 5:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t, time_limit=5.0)
            
            # Local Search
            child = _local_search(v, k, t, child, time_limit=5.0)
            
            new_population.append(child)
            
            if len(child) < len(best_individual):
                best_individual = child
                if len(best_individual) <= 151:
                    break
                    
        population = new_population
        
        if len(best_individual) <= 151:
            break
            
    # Final repair
    final_blocks = _repair_and_reduce(v, k, t, best_individual, time_limit=10.0)
    
    # Validate
    all_subsets = data['all_subsets']
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
            
    if len(covered) != len(all_subsets):
        # Fallback
        final_blocks = _fast_greedy_cover(v, k, t, time_limit=110.0)
        
    return final_blocks
# EVOLVE-BLOCK-END

"""Genetic algorithm covering design C(19,7,4) with efficient bitset operations."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

_CACHE = {}

def _get_precomputed(v, k, t):
    """Precompute coverage data using efficient numpy operations."""
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Coverage matrix: block i covers subset j
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.uint8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            coverage_matrix[i, t_index_map[combo]] = 1
    
    # Block to index mapping using frozenset
    block_frozensets = [frozenset(b) for b in all_blocks]
    block_to_idx = {fs: i for i, fs in enumerate(block_frozensets)}
    
    data = {
        'all_subsets': all_subsets,
        'num_subsets': num_subsets,
        't_index_map': t_index_map,
        'all_blocks': all_blocks,
        'num_blocks': num_blocks,
        'coverage_matrix': coverage_matrix,
        'block_to_idx': block_to_idx,
        'block_frozensets': block_frozensets,
    }
    _CACHE[key] = data
    return data

def _block_idx(block, data):
    """Get block index from block tuple/list."""
    fs = frozenset(block)
    return data['block_to_idx'].get(fs, -1)

def _greedy_cover(data, time_limit=20.0):
    """Greedy set cover using coverage matrix."""
    start = time.time()
    cm = data['coverage_matrix']
    ns = data['num_subsets']
    nb = data['num_blocks']
    
    uncovered = np.ones(ns, dtype=np.uint8)
    selected = []
    
    while uncovered.any() and time.time() - start < time_limit:
        gains = cm @ uncovered
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        selected.append(best)
        uncovered[cm[best] == 1] = 0
    
    return selected

def _repair(data, block_indices, time_limit=15.0):
    """Repair: add missing blocks, then remove redundant ones."""
    if not block_indices:
        return _greedy_cover(data, time_limit)
    
    start = time.time()
    cm = data['coverage_matrix']
    ns = data['num_subsets']
    
    # Deduplicate
    unique_idx = list(dict.fromkeys(block_indices))
    
    # Compute coverage
    total = np.zeros(ns, dtype=np.uint8)
    for bi in unique_idx:
        total |= cm[bi]
    
    uncovered = (total == 0).astype(np.uint8)
    
    # Add blocks to cover gaps
    while uncovered.any() and time.time() - start < time_limit:
        gains = cm @ uncovered
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        unique_idx.append(best)
        total |= cm[best]
        uncovered = (total == 0).astype(np.uint8)
    
    # Reduction: remove redundant blocks
    if len(unique_idx) > 1:
        changed = True
        while changed and time.time() - start < time_limit:
            changed = False
            new_idx = []
            for i, bi in enumerate(unique_idx):
                if len(unique_idx) <= 1:
                    new_idx.append(bi)
                    continue
                # Check if block bi is redundant
                mask_without = np.zeros(ns, dtype=np.uint8)
                for j, bj in enumerate(unique_idx):
                    if j != i:
                        mask_without |= cm[bj]
                if mask_without.all():
                    changed = True
                else:
                    new_idx.append(bi)
            unique_idx = new_idx
    
    return unique_idx

def _crossover(parent1_idx, parent2_idx, data, time_limit=15.0):
    """Crossover: greedy selection from union of parent blocks."""
    start = time.time()
    cm = data['coverage_matrix']
    ns = data['num_subsets']
    
    # Union of blocks
    seen = set()
    pool = []
    for bi in parent1_idx + parent2_idx:
        if bi not in seen:
            seen.add(bi)
            pool.append(bi)
    
    uncovered = np.ones(ns, dtype=np.uint8)
    child = []
    
    while uncovered.any() and pool and time.time() - start < time_limit:
        best_pos = -1
        best_gain = -1
        for pos, bi in enumerate(pool):
            gain = int(np.sum(cm[bi] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
        if best_gain <= 0:
            break
        bi = pool.pop(best_pos)
        child.append(bi)
        uncovered[cm[bi] == 1] = 0
    
    if uncovered.any():
        child = _repair(data, child, time_limit)
    
    return child

def _local_search(data, block_indices, time_limit=10.0):
    """Local search: try single-block removals and swaps."""
    start = time.time()
    current = list(block_indices)
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try removing each block
        for i in range(len(current)):
            if time.time() - start >= time_limit:
                break
            candidate = current[:i] + current[i+1:]
            candidate = _repair(data, candidate, time_limit=3.0)
            if len(candidate) < len(current):
                current = candidate
                improved = True
                break
        
        if not improved:
            # Try swap: remove one, add another
            for i in range(len(current)):
                if time.time() - start >= time_limit:
                    break
                for new_bi in range(data['num_blocks']):
                    if new_bi in current:
                        continue
                    candidate = current[:i] + [new_bi] + current[i+1:]
                    candidate = _repair(data, candidate, time_limit=2.0)
                    if len(candidate) < len(current):
                        current = candidate
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return current

def construct(v, k, t):
    """Main entry: GA with efficient bitset-based operations."""
    start = time.time()
    time_limit = 115.0
    data = _get_precomputed(v, k, t)
    
    # Initial population
    population = []
    
    # Greedy with different random seeds for diversity
    random.seed(42)
    for trial in range(5):
        g = _greedy_cover(data, time_limit=10.0)
        # Perturb and repair
        if g:
            perturbed = list(g)
            n_perturb = max(1, len(perturbed) // 15)
            for _ in range(n_perturb):
                idx = random.randint(0, len(perturbed) - 1)
                perturbed[idx] = random.randint(0, data['num_blocks'] - 1)
            repaired = _repair(data, perturbed, time_limit=10.0)
            population.append(repaired)
        else:
            population.append(g)
    
    # Random initial + repair
    for _ in range(3):
        rand_idx = [random.randint(0, data['num_blocks'] - 1) for _ in range(100)]
        repaired = _repair(data, rand_idx, time_limit=10.0)
        population.append(repaired)
    
    best = min(population, key=len)
    
    # GA loop
    pop_size = 10
    population = population[:pop_size]
    
    while time.time() - start < time_limit - 20:
        population.sort(key=len)
        elite_n = max(3, len(population) // 2)
        elite = population[:elite_n]
        
        new_pop = list(elite)
        
        while len(new_pop) < pop_size:
            if time.time() - start >= time_limit - 15:
                break
            
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, data, time_limit=10.0)
            
            # Mutation
            if random.random() < 0.4 and child:
                n_mut = random.randint(1, 3)
                for _ in range(n_mut):
                    if child:
                        idx = random.randint(0, len(child) - 1)
                        child[idx] = random.randint(0, data['num_blocks'] - 1)
                child = _repair(data, child, time_limit=8.0)
            
            # Local search
            if time.time() - start < time_limit - 15:
                child = _local_search(data, child, time_limit=5.0)
            
            new_pop.append(child)
        
        population = new_pop
        cur_best = min(population, key=len)
        if len(cur_best) < len(best):
            best = cur_best
    
    # Final repair
    final_idx = _repair(data, best, time_limit=10.0)
    
    # Convert to blocks
    final_blocks = [list(data['all_blocks'][bi]) for bi in final_idx]
    
    # Validate
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
    
    if len(covered) != data['num_subsets']:
        fallback = _greedy_cover(data, time_limit=100.0)
        final_blocks = [list(data['all_blocks'][bi]) for bi in fallback]
    
    return final_blocks
# EVOLVE-BLOCK-END

import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
    
    # Precompute triple frequency (how many blocks cover each triple)
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    
    # Precompute inverse frequency for weighted gain
    inv_freq = 1.0 / (triple_freq + 1e-9)
    
    def _local_search_reduce(selected_blocks, deadline):
        """Try to remove redundant blocks from the solution."""
        if not selected_blocks:
            return selected_blocks
        
        # Build coverage count for each triple
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block in selected_blocks:
            for triple in itertools.combinations(block, t):
                coverage_count[triple_to_idx[triple]] += 1
        
        # Try to remove blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Create a copy to iterate over
            for block in list(selected_blocks):
                # Check if this block can be removed
                can_remove = True
                for triple in itertools.combinations(block, t):
                    if coverage_count[triple_to_idx[triple]] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove this block
                    for triple in itertools.combinations(block, t):
                        coverage_count[triple_to_idx[triple]] -= 1
                    selected_blocks.remove(block)
                    changed = True
        return selected_blocks
    
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Deterministic greedy
    uncovered = np.ones(n_triples, dtype=bool)
    selected_blocks = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
        gains = np.sum(uncovered[block_triples], axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_blocks.append(blocks[best_idx])
        uncovered[block_triples[best_idx]] = False
    
    # Local search to reduce
    reduced = _local_search_reduce(selected_blocks, deadline)
    if len(reduced) < best_size:
        best_size = len(reduced)
        best_solution = [list(b) for b in reduced]
    
    # Strategy 2: Randomized greedy with multiple restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 2.0:
                break
            gains = np.sum(uncovered[block_triples], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Random tie-breaking among candidates with high gains
            candidates = np.where(gains >= max_gain * 0.9)[0]
            if len(candidates) > 0:
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
            
            selected_blocks.append(blocks[best_idx])
            uncovered[block_triples[best_idx]] = False
        
        # Local search to reduce
        reduced = _local_search_reduce(selected_blocks, deadline)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_solution = [list(b) for b in reduced]
    
    # Strategy 3: Weighted greedy (prioritize rare triples)
    for restart in range(8):
        if time.time() > deadline - 3.0:
            break
        
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 1.0:
                break
            
            # Weighted gain using vectorized NumPy
            # gains[i] = sum(inv_freq[block_triples[i]] * uncovered[block_triples[i]])
            weighted_mask = inv_freq[block_triples] * uncovered[block_triples].astype(np.float64)
            gains = np.sum(weighted_mask, axis=1)
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            
            selected_blocks.append(blocks[best_idx])
            uncovered[block_triples[best_idx]] = False
        
        # Local search to reduce
        reduced = _local_search_reduce(selected_blocks, deadline)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_solution = [list(b) for b in reduced]
    
    if best_solution is None:
        # Fallback: return empty list (should not happen)
        return []
    
    return best_solution

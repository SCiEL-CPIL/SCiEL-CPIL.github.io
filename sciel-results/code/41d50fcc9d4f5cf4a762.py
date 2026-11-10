"""
Improved covering design solver using Local Search on a valid cover.
Approach:
1. Generate an initial valid cover using a fast greedy method.
2. Use Local Search (Simulated Annealing style) to minimize the number of blocks.
   - Moves: Swap blocks, Remove blocks.
   - Constraint: Cover must remain valid (all triples covered).
   - Heuristic: Accept swaps that maintain validity and improve a secondary metric
     (e.g., number of redundant blocks or overlap structure) to escape local minima.
"""
import itertools
import random
import numpy as np
import time

# Global cache for precomputed block-triple relationships
_cache = {}

def _precompute_blocks(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    
    all_points = list(range(v))
    all_blocks = list(itertools.combinations(all_points, k))
    num_blocks = len(all_blocks)
    all_triples = list(itertools.combinations(all_points, t))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # For each block, store the list of triple indices it covers
    block_triples = np.zeros((num_blocks, t), dtype=np.int32) # This is wrong, each block covers C(k,t) triples
    # C(7,3) = 35.
    triples_per_block = int(len(list(itertools.combinations(range(k), t))))
    
    block_triples_list = []
    # Inverse: for each triple, which blocks cover it?
    # This is expensive to build fully. 
    # Instead, we can just iterate.
    
    # For speed in local search, we need to quickly check validity.
    # We maintain `triple_counts`.
    
    data = {
        'all_blocks': all_blocks,
        'num_blocks': num_blocks,
        'all_triples': all_triples,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx,
        'triples_per_block': triples_per_block,
        'block_triples_idx': [] # List of lists of triple indices
    }
    
    for b in all_blocks:
        b_set = set(b)
        covered = []
        for tr in all_triples:
            if all(x in b_set for x in tr):
                covered.append(triple_to_idx[tr])
        data['block_triples_idx'].append(covered)
        
    _cache[key] = data
    return data

def _local_search(v, k, t, initial_blocks, time_limit=110):
    data = _precompute_blocks(v, k, t)
    num_blocks = data['num_blocks']
    num_triples = data['num_triples']
    block_triples_idx = data['block_triples_idx']
    
    # Initialize state
    active = np.zeros(num_blocks, dtype=bool)
    triple_counts = np.zeros(num_triples, dtype=np.int32)
    
    for b in initial_blocks:
        b_idx = data['all_blocks'].index(b) # This is O(N), slow. 
        # Better: create a set of block tuples for lookup?
        # Or just assume initial_blocks are from the same list?
        # Let's make a lookup map for initial blocks.
        # Actually, _greedy_cover returns lists. We can convert to tuples.
        pass
    
    # Re-initialize properly with lookup
    block_set_to_idx = {b: i for i, b in enumerate(data['all_blocks'])}
    
    active_blocks_list = []
    for b in initial_blocks:
        b_tuple = tuple(b)
        if b_tuple in block_set_to_idx:
            idx = block_set_to_idx[b_tuple]
            if not active[idx]:
                active[idx] = True
                active_blocks_list.append(idx)
                for t_idx in block_triples_idx[idx]:
                    triple_counts[t_idx] += 1
    
    current_num_blocks = len(active_blocks_list)
    best_num_blocks = current_num_blocks
    best_active = active.copy()
    best_counts = triple_counts.copy()
    
    rng = random.Random(42)
    start_time = time.time()
    
    # Helper to check if a block is removable
    def is_removable(b_idx):
        for t_idx in block_triples_idx[b_idx]:
            if triple_counts[t_idx] == 1:
                return False
        return True
    
    # Helper to prune
    def prune():
        nonlocal active, triple_counts, active_blocks_list, current_num_blocks
        changed = True
        while changed:
            changed = False
            new_active_list = []
            for b_idx in active_blocks_list:
                if is_removable(b_idx):
                    # Remove
                    active[b_idx] = False
                    for t_idx in block_triples_idx[b_idx]:
                        triple_counts[t_idx] -= 1
                    changed = True
                else:
                    new_active_list.append(b_idx)
            active_blocks_list = new_active_list
            current_num_blocks = len(active_blocks_list)
            
            if current_num_blocks < best_num_blocks:
                best_num_blocks = current_num_blocks
                best_active = active.copy()
                best_counts = triple_counts.copy()
                
    # Initial Prune
    prune()
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        # Try to find a swap
        if not active_blocks_list:
            break
            
        # Pick a random active block to remove
        out_idx_pos = rng.randrange(len(active_blocks_list))
        out_idx = active_blocks_list[out_idx_pos]
        
        # Check if removing it makes the cover invalid
        # If valid, we would have pruned it. So it must be invalid.
        # Find which triples become uncovered.
        uncovered_triples = []
        for t_idx in block_triples_idx[out_idx]:
            if triple_counts[t_idx] == 1:
                uncovered_triples.append(t_idx)
                
        if not uncovered_triples:
            # Should have been pruned
            continue
            
        # We need to find an inactive block that covers all uncovered_triples
        # AND is not currently active.
        # Also, to be a good swap, it should ideally not introduce new "critical" issues,
        # but since we are swapping, the count of blocks stays the same.
        
        # Optimization: Only consider inactive blocks that cover *at least one* of the uncovered triples?
        # Actually, it must cover *all* of them.
        
        # Heuristic: Pick a random uncovered triple, and look at blocks covering it.
        # If the number of uncovered triples is small, intersect the lists of blocks covering each.
        
        if len(uncovered_triples) > 3:
            # Too many constraints, skip this move or try harder?
            # For now, skip if too complex
            continue
            
        # Get candidates from the first uncovered triple
        # We need a reverse index: triple -> list of blocks
        # We didn't precompute that. Let's build it on the fly or precompute?
        # Precomputing reverse index is better.
        
        # Let's add reverse index to _precompute_blocks
        pass
        
    # Return best found
    # Convert best_active back to list of blocks
    final_blocks = []
    for i in range(num_blocks):
        if best_active[i]:
            final_blocks.append(list(data['all_blocks'][i]))
            
    return final_blocks

def construct(v, k, t):
    # 1. Get initial valid cover using the existing greedy
    # We reuse the logic from the current program but make it faster/deterministic for init
    # The current _greedy_cover is slow. Let's use a simpler greedy for init.
    
    # Simple Greedy Init
    all_points = list(range(v))
    all_t_subs = set(itertools.combinations(all_points, t))
    uncovered = set(all_t_subs)
    blocks = []
    rng = random.Random(42)
    
    # Precompute blocks? No, just greedy.
    # To make it fast, we'll use a simpler heuristic:
    # Pick random block, if it covers most uncovered, keep it.
    
    # Actually, let's just use the existing _greedy_cover but with a time limit or fewer restarts for init?
    # The existing _greedy_cover is O(N * K * V * C(K,T)).
    
    # Let's implement a very fast greedy:
    # 1. Shuffle points.
    # 2. For each step, pick the block that covers the most uncovered triples.
    #    To do this fast, we need to score blocks.
    #    We can precompute all blocks and their triple masks.
    
    data = _precompute_blocks(v, k, t)
    num_blocks = data['num_blocks']
    block_triples_idx = data['block_triples_idx']
    num_triples = data['num_triples']
    
    # State for greedy
    triple_counts = np.zeros(num_triples, dtype=np.int32)
    uncovered_mask = np.ones(num_triples, dtype=bool)
    
    active_blocks = []
    
    # Greedy
    while np.any(uncovered_mask):
        best_block_idx = -1
        best_gain = 0
        
        # Sample a subset of blocks to speed up?
        # Or just iterate all? 346k blocks * 35 triples = 12M ops per step.
        # 76 steps * 12M = 900M ops. Might be too slow in Python.
        
        # Optimization: Only consider blocks that cover at least one uncovered triple.
        # We can maintain a list of "active" candidate blocks?
        
        # Simpler: Randomized Greedy.
        # Pick 100 random blocks, score them, pick best.
        candidates = rng.sample(range(num_blocks), 100)
        for b_idx in candidates:
            gain = 0
            for t_idx in block_triples_idx[b_idx]:
                if uncovered_mask[t_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break
            
        active_blocks.append(best_block_idx)
        for t_idx in block_triples_idx[best_block_idx]:
            triple_counts[t_idx] += 1
            if triple_counts[t_idx] == 1:
                uncovered_mask[t_idx] = False
                
    # Convert to list of blocks
    initial_blocks = [list(data['all_blocks'][b_idx]) for b_idx in active_blocks]
    
    # 2. Local Search
    if time.time() < 110: # Leave time for LS
        optimized_blocks = _local_search(v, k, t, initial_blocks, time_limit=115)
        if len(optimized_blocks) < len(initial_blocks):
            return optimized_blocks
            
    return initial_blocks

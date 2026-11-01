import itertools
import random
import time

# Global cache for precomputation to avoid recomputing on multiple calls if any
_PRECOMPUTED = {}

def _precompute(v, k, t):
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]
        
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Generate all k-blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = [0] * len(all_blocks)
    blocks_by_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            mask |= (1 << t_idx)
            blocks_by_triple[t_idx].append(b_idx)
        block_masks[b_idx] = mask
        
    _PRECOMPUTED[(v, k, t)] = {
        'all_blocks': all_blocks,
        'block_masks': block_masks,
        'blocks_by_triple': blocks_by_triple,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx
    }
    return _PRECOMPUTED[(v, k, t)]

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters (not expected in this task)
        return _simple_greedy(v, k, t)

    start_time = time.time()
    deadline = start_time + 110.0
    
    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    full_mask = (1 << num_triples) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy instances
    run_id = 0
    while time.time() < deadline:
        blocks = _randomized_greedy_bitset(v, k, t, data, random_seed=run_id)
        run_id += 1
        
        # Apply reduction
        blocks = _reduce_blocks(blocks, v, t)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            if best_count <= 76:
                # We found a very good solution. Continue to see if we can beat it.
                pass

    if best_blocks is None:
        best_blocks = _simple_greedy(v, k, t)
        
    # Final reduction
    best_blocks = _reduce_blocks(best_blocks, v, t)
    
    return [list(b) for b in best_blocks]

def _randomized_greedy_bitset(v, k, t, data, random_seed):
    rng = random.Random(random_seed)
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    uncovered_mask = (1 << num_triples) - 1
    blocks = []
    
    while uncovered_mask:
        # Pick a random uncovered triple
        # Find a set bit in uncovered_mask
        # We can pick a random index from 0..num_triples-1 and check if it's set,
        # or extract a random bit. Extracting a random bit from a large int is tricky.
        # Simpler: pick random triple index, if covered, retry? 
        # Better: iterate through bits? No, 2024 bits is small.
        # Let's just pick a random triple index and check if it's in uncovered_mask.
        # If not, pick another. Since coverage is dense, this might be slow if uncovered is small.
        # Alternative: maintain a list of uncovered triple indices? 
        # But updating list is O(N). 
        # Let's use the bitset to find *any* uncovered triple quickly?
        # `uncovered_mask & -uncovered_mask` gives the lowest set bit.
        # We can randomize by rotating? 
        # Let's just pick a random triple index and check.
        # If we are near the end, most triples are covered.
        # To ensure we don't loop forever, we can maintain a list of uncovered indices?
        # Let's stick to bitset check.
        
        while True:
            t_idx = rng.randrange(num_triples)
            if (uncovered_mask >> t_idx) & 1:
                break
        
        candidates = blocks_by_triple[t_idx]
        best_block_idx = None
        best_gain = -1
        
        # Evaluate candidates
        for b_idx in candidates:
            mask = block_masks[b_idx]
            gain = (mask & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and rng.random() < 0.5:
                best_block_idx = b_idx
                
        blocks.append(all_blocks[best_block_idx])
        uncovered_mask &= ~block_masks[best_block_idx]
        
    return blocks

def _reduce_blocks(blocks, v, t):
    if not blocks:
        return []
        
    # Map each triple to list of block indices covering it
    triple_to_blocks = {}
    for i, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            if sub not in triple_to_blocks:
                triple_to_blocks[sub] = []
            triple_to_blocks[sub].append(i)
            
    active = [True] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if not active[i]:
                continue
            redundant = True
            for sub in itertools.combinations(blocks[i], t):
                # Check if any other active block covers this sub
                covered = False
                for j in triple_to_blocks[sub]:
                    if j != i and active[j]:
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                active[i] = False
                changed = True
                
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        best_gain = -1
        remaining_points = [p for p in range(v) if p not in target]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks

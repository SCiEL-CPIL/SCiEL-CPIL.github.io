import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    # Use a safety margin to ensure we return before the 150s limit
    deadline = start_time + 145.0
    
    # Precompute all t-subsets and map them to IDs
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # 1. Generate a randomized greedy covering
        blocks = _greedy_cover(v, k, t, all_subsets, subset_to_id, seed)
        
        # 2. Optimize by removing redundant blocks
        blocks = _optimize(blocks, t, subset_to_id)
        
        # 3. Check for improvement
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
    return best_blocks

def _greedy_cover(v, k, t, all_subsets, subset_to_id, seed):
    rng = random.Random(seed)
    # List of uncovered subset IDs
    uncovered = list(range(len(all_subsets)))
    uncovered_set = set(uncovered)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered subset to anchor the block
        idx = rng.randrange(len(uncovered))
        anchor_subset = all_subsets[uncovered[idx]]
        
        block = list(anchor_subset)
        
        # Greedily add points to maximize coverage of uncovered subsets
        while len(block) < k:
            best_gain = -1
            best_candidates = []
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # Check subsets formed by x and t-1 points from block
                for rest in itertools.combinations(block, t - 1):
                    new_subset = tuple(sorted(rest + (x,)))
                    sid = subset_to_id.get(new_subset)
                    if sid is not None and sid in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(x)
            
            if not best_candidates:
                # If no gain, pick a random point not in block
                candidates = [x for x in range(v) if x not in block]
                x = rng.choice(candidates)
            else:
                x = rng.choice(best_candidates)
            
            block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for s in itertools.combinations(block, t):
            sid = subset_to_id.get(s)
            if sid is not None:
                if sid in uncovered_set:
                    uncovered_set.remove(sid)
                    # Remove from list efficiently (swap with last, pop)
                    # Find index of sid in uncovered list
                    # This is O(N), but N decreases.
                    # For better perf, we could use a different structure, but N=2024 is small.
                    try:
                        u_idx = uncovered.index(sid)
                        uncovered[u_idx] = uncovered[-1]
                        uncovered.pop()
                    except ValueError:
                        pass # Already removed? Should not happen if logic is correct.
        
    return blocks

def _optimize(blocks, t, subset_to_id):
    if not blocks:
        return blocks
        
    # Map each block to the list of subset IDs it covers
    block_ids = []
    coverage = {}
    
    for block in blocks:
        ids = []
        for s in itertools.combinations(block, t):
            sid = subset_to_id.get(s)
            if sid is not None:
                ids.append(sid)
                coverage[sid] = coverage.get(sid, 0) + 1
        block_ids.append(ids)
        
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_ids = []
        
        for i, block in enumerate(blocks):
            # Check if block is redundant: all its subsets are covered by others
            redundant = True
            for sid in block_ids[i]:
                if coverage[sid] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block
                for sid in block_ids[i]:
                    coverage[sid] -= 1
                changed = True
            else:
                new_blocks.append(block)
                new_block_ids.append(block_ids[i])
                
        blocks = new_blocks
        block_ids = new_block_ids
        
    return blocks

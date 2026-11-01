import itertools
import time
import random
from collections import defaultdict

def _greedy_random(v, k, t, seed):
    rng = random.Random(seed)
    all_points = list(range(v))
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    if not uncovered:
        return []

    while uncovered:
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                gain = 0
                if t == 3:
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = tuple(sorted((block[i], block[j], x)))
                            if tri in uncovered:
                                gain += 1
                else:
                    for comb in itertools.combinations(block, t-1):
                        tri = tuple(sorted(list(comb) + [x]))
                        if tri in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                next_point = rng.choice([x for x in all_points if x not in block_set])
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        blocks.append(block)
        
        for tri in itertools.combinations(block, t):
            uncovered.discard(tuple(tri))
            
    return blocks

def _reduce_blocks(v, k, t, blocks, deadline):
    """Try to remove redundant blocks from the covering."""
    if not blocks:
        return blocks
    
    # Build coverage count for each triple
    triple_to_blocks = defaultdict(list)
    for idx, block in enumerate(blocks):
        for tri in itertools.combinations(block, t):
            triple_to_blocks[tuple(sorted(tri))].append(idx)
    
    all_triples = list(triple_to_blocks.keys())
    removed = [False] * len(blocks)
    
    # Iteratively try to remove blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if removed[i]:
                continue
            # Check if block i is the only block covering any triple
            is_redundant = True
            for tri in itertools.combinations(blocks[i], t):
                tri_key = tuple(sorted(tri))
                covering_blocks = [j for j in triple_to_blocks[tri_key] if not removed[j]]
                if len(covering_blocks) == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                removed[i] = True
                changed = True
                # Update triple_to_blocks
                for tri in itertools.combinations(blocks[i], t):
                    tri_key = tuple(sorted(tri))
                    if i in triple_to_blocks[tri_key]:
                        triple_to_blocks[tri_key].remove(i)
    
    # Filter out removed blocks
    result = [blocks[i] for i in range(len(blocks)) if not removed[i]]
    return result

def _local_search_swap(v, k, t, blocks, deadline):
    """Try to improve the covering by swapping points in blocks."""
    if not blocks:
        return blocks
    
    rng = random.Random(int(time.time() * 1000) % 100000)
    all_points = list(range(v))
    
    # Build coverage structure
    triple_count = defaultdict(int)
    for block in blocks:
        for tri in itertools.combinations(block, t):
            triple_count[tuple(sorted(tri))] += 1
    
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    # Try random swaps
    while time.time() < deadline:
        # Pick a random block and try to improve it
        block_idx = rng.randint(0, len(best_blocks) - 1)
        original_block = list(best_blocks[block_idx])
        block_set = set(original_block)
        
        # Try removing one point and adding another
        for remove_pt in original_block:
            new_block_set = block_set - {remove_pt}
            # Try adding each point not in the block
            for add_pt in all_points:
                if add_pt in new_block_set:
                    continue
                
                new_block = sorted(new_block_set | {add_pt})
                
                # Calculate how many new triples this block would cover
                # that aren't covered by other blocks
                gain = 0
                loss = 0
                for tri in itertools.combinations(new_block, t):
                    tri_key = tuple(sorted(tri))
                    if tri_key in triple_count:
                        if triple_count[tri_key] == 1:
                            # This triple is only covered by this block
                            # Check if it's in the new block
                            if all(p in new_block_set for p in tri_key):
                                loss += 1
                    else:
                        gain += 1
                
                # Also count triples in original block that won't be covered
                for tri in itertools.combinations(original_block, t):
                    tri_key = tuple(sorted(tri))
                    if tri_key in triple_count and triple_count[tri_key] == 1:
                        if not all(p in new_block_set for p in tri_key):
                            loss += 1
                
                if gain > loss:
                    # Apply the swap
                    best_blocks[block_idx] = new_block
                    
                    # Update triple_count
                    for tri in itertools.combinations(original_block, t):
                        triple_count[tuple(sorted(tri))] -= 1
                    for tri in itertools.combinations(new_block, t):
                        triple_count[tuple(sorted(tri))] += 1
                    
                    changed = True
                    break
            else:
                continue
            break
    
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Time remaining for this iteration
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        blocks = _greedy_random(v, k, t, seed)
        
        # Reduce blocks
        reduce_deadline = min(deadline - 2, time.time() + remaining * 0.4)
        blocks = _reduce_blocks(v, k, t, blocks, reduce_deadline)
        
        # Local search if time permits
        if time.time() < deadline - 10:
            ls_deadline = min(deadline - 2, time.time() + (deadline - time.time()) * 0.5)
            blocks = _local_search_swap(v, k, t, blocks, ls_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if best_len == 69:
            break
            
        seed += 1
    
    return best_blocks

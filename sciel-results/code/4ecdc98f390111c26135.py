import itertools
import time
import random

def _greedy_fast(v, k, t, seed):
    rng = random.Random(seed)
    all_points = list(range(v))
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    if not uncovered:
        return []

    # Pre-compute pairs for t=3 to speed up gain calculation
    if t == 3:
        # Map point to list of points? No.
        pass

    while uncovered:
        # Pick random uncovered triple
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            # Calculate gain for each candidate
            # Gain = number of uncovered triples completed by adding x
            for x in all_points:
                if x in block_set:
                    continue
                
                gain = 0
                if t == 3:
                    # Check pairs in block
                    # Optimization: pre-compute pairs in block?
                    # Block size is small (max 7).
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = (block[i], block[j], x)
                            # Tuple must be sorted for set lookup? 
                            # itertools.combinations produces sorted tuples.
                            # So we must sort (block[i], block[j], x)
                            a, b, c = sorted((block[i], block[j], x))
                            if (a, b, c) in uncovered:
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

def _remove_redundant(blocks, v, t):
    # Returns True if any block was removed
    # Use a set of covered triples for each block?
    # Better: count coverage.
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    cover_count = [0] * len(all_triples)
    
    for block in blocks:
        for tri in itertools.combinations(block, t):
            cover_count[triple_to_idx[tri]] += 1
            
    changed = False
    i = 0
    while i < len(blocks):
        block = blocks[i]
        redundant = True
        for tri in itertools.combinations(block, t):
            if cover_count[triple_to_idx[tri]] == 1:
                redundant = False
                break
        
        if redundant:
            # Remove block
            for tri in itertools.combinations(block, t):
                cover_count[triple_to_idx[tri]] -= 1
            blocks.pop(i)
            changed = True
            # Do not increment i, check same index again
        else:
            i += 1
            
    return changed

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run greedy restarts
    while time.time() < deadline - 10.0:
        blocks = _greedy_fast(v, k, t, seed)
        
        # Apply redundancy removal
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = _remove_redundant(blocks, v, t)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 69:
            break
            
        seed += 1
        
    return best_blocks

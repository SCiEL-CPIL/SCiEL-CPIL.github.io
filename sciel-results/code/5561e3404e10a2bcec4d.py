import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute triple IDs
    triples = list(itertools.combinations(range(v), t))
    triple_to_id = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # 3D Lookup Table for Triple ID
    # triple_id[a][b][c] valid if a < b < c
    triple_id = [[[-1]*v for _ in range(v)] for __ in range(v)]
    for i, (a, b, c) in enumerate(triples):
        triple_id[a][b][c] = i
        
    # Helper to get ID from sorted points
    def get_id(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return triple_id[a][b][c]

    def greedy_construct(noise=0.0):
        uncovered = set(range(num_triples))
        # Point coverage counts for tie-breaking
        pt_count = [0] * v
        for tid in uncovered:
            a, b, c = triples[tid]
            pt_count[a] += 1
            pt_count[b] += 1
            pt_count[c] += 1
            
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            block = []
            
            while len(block) < k:
                best_x = -1
                best_score = -1
                best_gain = -1
                best_pt_count = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Calculate gain: triples {x, y, z} with y,z in block
                    if len(block) >= t - 1:
                        for i in range(len(block)):
                            for j in range(i+1, len(block)):
                                y = block[i]
                                z = block[j]
                                tid = get_id(x, y, z)
                                if tid != -1 and tid in uncovered:
                                    gain += 1
                    else:
                        gain = 0
                        
                    score = gain
                    if noise > 0:
                        score += random.gauss(0, noise)
                    
                    # Tie-breaking: higher gain is better.
                    # If gain is same, prefer point with higher uncovered count.
                    # If score (with noise) is higher, it's better.
                    # We want to maximize score.
                    # If scores are close, use pt_count as secondary key?
                    # The noise handles randomness.
                    
                    if score > best_score:
                        best_score = score
                        best_x = x
                        best_gain = gain
                        best_pt_count = pt_count[x]
                    elif score == best_score:
                        if pt_count[x] > best_pt_count:
                            best_x = x
                            best_gain = gain
                            best_pt_count = pt_count[x]
                
                if best_x == -1:
                    break
                    
                block.append(best_x)
            
            if not block:
                break
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered and pt_count
            for combo in itertools.combinations(block, t):
                tid = get_id(*combo)
                if tid != -1 and tid in uncovered:
                    uncovered.discard(tid)
                    a, b, c = triples[tid]
                    pt_count[a] -= 1
                    pt_count[b] -= 1
                    pt_count[c] -= 1
        
        return blocks

    def reduce_blocks(blocks):
        if not blocks:
            return blocks
            
        coverage = [0] * num_triples
        block_triples = []
        
        for block in blocks:
            t_ids = []
            for combo in itertools.combinations(block, t):
                tid = get_id(*combo)
                if tid != -1:
                    t_ids.append(tid)
                    coverage[tid] += 1
            block_triples.append(t_ids)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                
                redundant = True
                for tid in block_triples[i]:
                    if coverage[tid] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for tid in block_triples[i]:
                        coverage[tid] -= 1
                    blocks.pop(i)
                    block_triples.pop(i)
                    changed = True
                    break
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # 1. Deterministic
    blocks = greedy_construct(0.0)
    if blocks:
        blocks = reduce_blocks(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # 2. Randomized
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    for noise in noise_levels:
        while time.time() < deadline:
            blocks = greedy_construct(noise)
            if blocks is None:
                break
            blocks = reduce_blocks(blocks)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                
    return best_blocks

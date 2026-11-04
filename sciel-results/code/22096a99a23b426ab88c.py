import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute: for each t-subset index, its elements
    # and for fast membership, use a set of frozensets
    tsub_set = set(all_tsubsets)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_cover(seed=None, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(range(num_tsubsets))
        blocks = []
        
        while uncovered:
            if time.time() > time_limit:
                return None
            
            if seed is not None:
                target_idx = random.choice(list(uncovered))
            else:
                target_idx = min(uncovered)
            target = all_tsubsets[target_idx]
            
            block = list(target)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t-1):
                        new_ts = tuple(sorted(combo + (x,)))
                        if new_ts in tsub_set:
                            idx = tsub_to_idx[new_ts]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
        
        return blocks
    
    def remove_redundant(blocks, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        blocks = list(blocks)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for bi in range(len(blocks)):
                if time.time() > time_limit:
                    break
                redundant = True
                for ts in itertools.combinations(blocks[bi], t):
                    found = False
                    for bj, block_j in enumerate(blocks):
                        if bj == bi:
                            continue
                        if ts in itertools.combinations(block_j, t):
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(bi)
                    changed = True
                    break
        return blocks
    
    # Run multiple greedy attempts
    seeds = [None] + list(range(15))
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        blocks = greedy_cover(seed=seed, time_limit=deadline - 5)
        if blocks:
            blocks = remove_redundant(blocks, time_limit=deadline - 2)
            if blocks and len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        blocks = greedy_cover(seed=None, time_limit=deadline - 1)
        best_blocks = blocks
    
    return best_blocks

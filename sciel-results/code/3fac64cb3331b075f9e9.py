import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    points = range(v)
    all_tsets = list(itertools.combinations(points, t))
    
    # Map each t-set to the list of blocks containing it
    blocks_for_tset = {ts: [] for ts in all_tsets}
    
    # Precompute t-subsets for each block
    block_tsets = {}
    
    for block in itertools.combinations(points, k):
        t_subs = list(itertools.combinations(block, t))
        block_tsets[block] = t_subs
        for t_sub in t_subs:
            blocks_for_tset[t_sub].append(block)
            
    def greedy_cover(seed):
        random.seed(seed)
        uncovered_set = set(all_tsets)
        uncovered_list = list(all_tsets)
        blocks = []
        
        while uncovered_list:
            # Pick a random valid index
            while True:
                idx = random.randint(0, len(uncovered_list) - 1)
                t_set = uncovered_list[idx]
                if t_set in uncovered_set:
                    break
            
            # Remove from list
            uncovered_list[idx] = uncovered_list[-1]
            uncovered_list.pop()
            uncovered_set.remove(t_set)
            
            # Get candidates
            candidates = blocks_for_tset[t_set]
            
            best_block = None
            max_gain = -1
            
            for block in candidates:
                gain = 0
                for t_sub in block_tsets[block]:
                    if t_sub in uncovered_set:
                        gain += 1
                gain += random.random() * 0.1
                if gain > max_gain:
                    max_gain = gain
                    best_block = block
                    
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Remove covered t-sets
            for t_sub in block_tsets[best_block]:
                if t_sub in uncovered_set:
                    uncovered_set.remove(t_sub)
                    
            if time.time() > deadline:
                break
                
        return blocks

    def reduce_cover(blocks):
        if not blocks:
            return []
            
        coverage = {ts: 0 for ts in all_tsets}
        for block in blocks:
            for t_sub in block_tsets[block]:
                coverage[t_sub] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                redundant = True
                for t_sub in block_tsets[block]:
                    if coverage[t_sub] == 1:
                        redundant = False
                        break
                if redundant:
                    for t_sub in block_tsets[block]:
                        coverage[t_sub] -= 1
                    blocks.pop(i)
                    changed = True
                    if time.time() > deadline:
                        return blocks
        return blocks

    best_blocks = None
    seed = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        if len(blocks) > 0:
            reduced = reduce_cover(blocks)
            if best_blocks is None or len(reduced) < len(best_blocks):
                best_blocks = reduced
        seed += 1
        
    if best_blocks is None:
        blocks = greedy_cover(0)
        return [list(b) for b in blocks]
        
    return [list(b) for b in best_blocks]

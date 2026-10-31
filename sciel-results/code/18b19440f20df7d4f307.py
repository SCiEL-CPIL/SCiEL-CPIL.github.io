import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_set = set(all_t_subsets)
    
    def greedy_cover(randomize=False):
        uncovered = set(t_subset_set)
        blocks = []
        
        while uncovered:
            if randomize:
                start_ts = random.choice(list(uncovered))
            else:
                start_ts = min(uncovered)
                
            block = set(start_ts)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1: break
                block.add(best_point)
            
            blocks.append(sorted(block))
            for ts in itertools.combinations(blocks[-1], t):
                uncovered.discard(ts)
                
        return blocks

    def remove_redundant(blocks):
        changed = True
        while changed:
            changed = False
            cover_count = {ts: 0 for ts in t_subset_set}
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    cover_count[ts] += 1
            
            new_blocks = []
            for b in blocks:
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy
    blocks = greedy_cover(False)
    blocks = remove_redundant(blocks)
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = [b[:] for b in blocks]
        
    # 2. Randomized Greedy with Restarts
    while time.time() < deadline:
        blocks = greedy_cover(True)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]

    return best_blocks

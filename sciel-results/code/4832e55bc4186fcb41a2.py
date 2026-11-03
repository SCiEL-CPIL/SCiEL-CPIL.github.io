import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [[] for _ in range(num_blocks)]
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(num_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tsub_to_idx[ts]
            block_tsubs[b_idx].append(ts_idx)
            tsub_blocks[ts_idx].append(b_idx)
            
    # 2. Greedy Construction
    def greedy_cover(seed):
        rng = random.Random(seed)
        count = np.zeros(num_tsubs, dtype=np.int32)
        selected_blocks = []
        uncovered = set(range(num_tsubs))
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            ts_idx = next(iter(uncovered))
            candidates = tsub_blocks[ts_idx]
            
            best_block = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for ts in block_tsubs[b_idx]:
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected_blocks.append(best_block)
            
            for ts in block_tsubs[best_block]:
                count[ts] += 1
                if count[ts] == 1:
                    uncovered.discard(ts)
                    
        return selected_blocks

    best_blocks = None
    best_count = float('inf')
    
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        blocks = greedy_cover(seed)
        if blocks is not None and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            
    if best_blocks is None:
        return []
        
    # 3. Local Search
    count = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in best_blocks:
        for ts in block_tsubs[b_idx]:
            count[ts] += 1
            
    zero_set = set()
    for ts in range(num_tsubs):
        if count[ts] == 0:
            zero_set.add(ts)
            
    rng = random.Random(42)
    no_improve = 0
    max_no_improve = 5000
    current_stamp = 0
    freq = np.zeros(num_blocks, dtype=np.int16)
    
    while time.time() < deadline - 2:
        if no_improve > max_no_improve:
            break
            
        if not best_blocks:
            break
            
        idx = rng.randrange(len(best_blocks))
        b_old = best_blocks[idx]
        
        # Remove b_old
        for ts in block_tsubs[b_old]:
            count[ts] -= 1
            if count[ts] == 0:
                zero_set.add(ts)
                
        if not zero_set:
            best_blocks.pop(idx)
            no_improve = 0
            continue
            
        # Find candidates
        current_stamp += 1
        # Reset freq for used indices? 
        # We can just zero out the ones we touch.
        touched = []
        
        for ts in zero_set:
            for b in tsub_blocks[ts]:
                if freq[b] == 0:
                    touched.append(b)
                freq[b] += 1
                
        target_count = len(zero_set)
        candidates = []
        for b in touched:
            if freq[b] == target_count:
                candidates.append(b)
            freq[b] = 0 # Reset for next iteration
            
        if candidates:
            b_new = candidates[rng.randint(0, len(candidates)-1)]
            
            # Add b_new
            for ts in block_tsubs[b_new]:
                count[ts] += 1
                if count[ts] == 1:
                    zero_set.discard(ts)
            best_blocks.append(b_new)
            
            # Cleanup
            changed = True
            while changed:
                changed = False
                for i in range(len(best_blocks) - 1, -1, -1):
                    b_check = best_blocks[i]
                    redundant = True
                    for ts in block_tsubs[b_check]:
                        if count[ts] == 1:
                            redundant = False
                            break
                    if redundant:
                        for ts in block_tsubs[b_check]:
                            count[ts] -= 1
                            if count[ts] == 0:
                                zero_set.add(ts)
                        best_blocks.pop(i)
                        changed = True
                        no_improve = 0
                        break
        else:
            # Revert
            for ts in block_tsubs[b_old]:
                count[ts] += 1
                if count[ts] == 1:
                    zero_set.discard(ts)
            no_improve += 1
            
    return [list(all_blocks[b]) for b in best_blocks]

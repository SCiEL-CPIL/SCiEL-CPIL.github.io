import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute triples
    all_ts = list(itertools.combinations(range(v), t))
    t_to_id = {ts: i for i, ts in enumerate(all_ts)}
    n_ts = len(all_ts)
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_build(rng):
        cover_count = [0] * n_ts
        blocks = []
        
        # Data structures for fast random uncovered selection
        uncovered_ids = list(range(n_ts))
        id_to_pos = list(range(n_ts))
        is_uncovered = [True] * n_ts
        
        def remove_uncovered(tid):
            pos = id_to_pos[tid]
            if pos == -1:
                return
            last_tid = uncovered_ids[-1]
            uncovered_ids[pos] = last_tid
            id_to_pos[last_tid] = pos
            uncovered_ids.pop()
            id_to_pos[tid] = -1
            is_uncovered[tid] = False
            
        while uncovered_ids:
            # Pick random uncovered
            idx = rng.randrange(len(uncovered_ids))
            tid = uncovered_ids[idx]
            target_ts = all_ts[tid]
            block = list(target_ts)
            
            # Remove target from uncovered list immediately? 
            # No, we only remove when cover_count becomes 1.
            # But target is currently 0, so it is uncovered.
            # We will update cover_count after building the block.
            # So we don't remove it yet.
            
            # Greedily add points
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_ts = tuple(sorted(rest + (x,)))
                        nt_id = t_to_id[new_ts]
                        if cover_count[nt_id] == 0:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    break
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Update cover counts
            for ts in itertools.combinations(block, t):
                tid = t_to_id[ts]
                cover_count[tid] += 1
                if cover_count[tid] == 1:
                    remove_uncovered(tid)
                    
        return blocks, cover_count

    def local_search(blocks, cover_count):
        cc = cover_count[:]
        b_list = list(blocks)
        
        removed_any = True
        while removed_any:
            removed_any = False
            for i in range(len(b_list) - 1, -1, -1):
                block = b_list[i]
                for ts in itertools.combinations(block, t):
                    tid = t_to_id[ts]
                    cc[tid] -= 1
                valid = True
                for ts in itertools.combinations(block, t):
                    tid = t_to_id[ts]
                    if cc[tid] == 0:
                        valid = False
                        break
                
                if valid:
                    b_list.pop(i)
                    removed_any = True
                else:
                    for ts in itertools.combinations(block, t):
                        tid = t_to_id[ts]
                        cc[tid] += 1
                        
        return b_list, cc

    # Run iterations
    base_seed = 42
    for i in range(1000): # Max iterations, limited by time
        if time.time() > deadline - 5.0:
            break
        seed = base_seed + i
        r = random.Random(seed)
        blocks, cc = greedy_build(r)
        if len(blocks) < best_len:
            new_blocks, new_cc = local_search(blocks, cc)
            if len(new_blocks) < best_len:
                best_blocks = new_blocks
                best_len = len(new_blocks)
                
    if best_blocks is None:
        blocks, _ = greedy_build(random.Random(0))
        return blocks
    return best_blocks

import itertools
import random
import time
import math

def construct(v, k, t):
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Precompute all possible blocks and the subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_id = {b: i for i, b in enumerate(all_blocks)}
    n_blocks = len(all_blocks)
    
    # block_subsets[i] is a list of subset IDs covered by block i
    block_subsets = []
    for block in all_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_id[sub])
        block_subsets.append(subs)
        
    # subset_to_blocks[s_id] = list of block IDs containing subset s_id
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_id, subs in enumerate(block_subsets):
        for s_id in subs:
            subset_to_blocks[s_id].append(b_id)

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        # Greedy Construction
        uncovered = bytearray([1] * n_subsets)
        uncovered_ids = list(range(n_subsets))
        id_to_idx = {i: i for i in range(n_subsets)}
        
        blocks = []
        block_set = set()
        
        while uncovered_ids:
            if time.time() > deadline - 5.0:
                break
                
            # Pick random uncovered subset
            idx = random.randrange(len(uncovered_ids))
            t_id = uncovered_ids[idx]
            
            # Get candidate blocks for this subset
            candidates = subset_to_blocks[t_id]
            
            best_gain = -1
            best_cands = []
            
            for b_id in candidates:
                if b_id in block_set:
                    continue
                
                # Calculate gain
                gain = 0
                for s_id in block_subsets[b_id]:
                    if uncovered[s_id]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [b_id]
                elif gain == best_gain:
                    best_cands.append(b_id)
            
            if not best_cands:
                break
            
            chosen_b_id = random.choice(best_cands)
            blocks.append(all_blocks[chosen_b_id])
            block_set.add(chosen_b_id)
            
            # Update uncovered
            for s_id in block_subsets[chosen_b_id]:
                if uncovered[s_id]:
                    uncovered[s_id] = 0
                    sub_idx = id_to_idx[s_id]
                    last_idx = len(uncovered_ids) - 1
                    temp_id = uncovered_ids[sub_idx]
                    uncovered_ids[sub_idx] = uncovered_ids[last_idx]
                    uncovered_ids[last_idx] = temp_id
                    if sub_idx != last_idx:
                        id_to_idx[uncovered_ids[sub_idx]] = sub_idx
                    uncovered_ids.pop()
                    del id_to_idx[s_id]
                    
        if not uncovered_ids:
            # Reduction Phase
            current_b_ids = [block_to_id[b] for b in blocks]
            
            # Compute cover counts
            cover_count = [0] * n_subsets
            for b_id in current_b_ids:
                for s_id in block_subsets[b_id]:
                    cover_count[s_id] += 1
            
            # Iteratively remove redundant blocks
            while time.time() < deadline - 5.0:
                removed_any = False
                for i in range(len(current_b_ids)):
                    b_id = current_b_ids[i]
                    can_remove = True
                    for s_id in block_subsets[b_id]:
                        if cover_count[s_id] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for s_id in block_subsets[b_id]:
                            cover_count[s_id] -= 1
                        current_b_ids.pop(i)
                        removed_any = True
                        break
                if not removed_any:
                    break

            if len(current_b_ids) < best_len:
                best_len = len(current_b_ids)
                best_blocks = [all_blocks[b_id] for b_id in current_b_ids]

    return best_blocks

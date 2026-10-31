import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0

    # Fallback for non-standard parameters
    if v != 19 or k != 7 or t != 4:
        return _greedy_generic(v, k, t)

    # --- Precomputation ---
    # 1. Map 4-subsets to IDs
    map4 = {}
    all_4s = []
    id_counter = 0
    for comb in itertools.combinations(range(v), t):
        map4[comb] = id_counter
        all_4s.append(comb)
        id_counter += 1
    n_4s = id_counter # 3876

    # 2. For each 7-subset (block), get list of 4-subset IDs it covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks) # 50388
    block_4s = []
    for block in all_blocks:
        ids = []
        for sub in itertools.combinations(block, t):
            ids.append(map4[sub])
        block_4s.append(np.array(ids, dtype=np.int32))
    
    # 3. For each 4-subset, get list of block indices covering it
    sub_blocks = [[] for _ in range(n_4s)]
    for b_idx, ids in enumerate(block_4s):
        for sid in ids:
            sub_blocks[sid].append(b_idx)
            
    # --- Greedy Initialization ---
    counts = np.zeros(n_4s, dtype=np.int32)
    current_blocks = []
    
    uncovered_mask = np.ones(n_4s, dtype=bool)
    
    # Precompute block_4s_array once
    block_4s_array = np.array(block_4s, dtype=np.int32)
    
    while np.any(uncovered_mask):
        if time.time() - start_time > time_limit - 15:
            break
            
        scores = uncovered_mask[block_4s_array].sum(axis=1)
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
             break
             
        current_blocks.append(best_idx)
        counts[block_4s_array[best_idx]] += 1
        uncovered_mask = (counts == 0)
        
    # --- Local Search ---
    def is_valid():
        return np.all(counts >= 1)
        
    def try_remove(b_idx):
        counts[block_4s_array[b_idx]] -= 1
        if np.any(counts == 0):
            counts[block_4s_array[b_idx]] += 1
            return False
        current_blocks.remove(b_idx)
        return True
        
    def try_add(b_idx):
        if b_idx in current_blocks:
            return False
        current_blocks.append(b_idx)
        counts[block_4s_array[b_idx]] += 1
        return True
        
    def get_redundancy_scores():
        if not current_blocks:
            return []
        b_4s = block_4s_array[current_blocks]
        c_4s = counts[b_4s]
        red_scores = (c_4s > 1).sum(axis=1)
        return red_scores
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    while time.time() - start_time < time_limit - 10:
        # 1. Try to remove most redundant blocks
        removed_any = False
        red_scores = get_redundancy_scores()
        if len(red_scores) == 0:
            break
            
        order = np.argsort(red_scores)[::-1]
        
        for i in order:
            if time.time() - start_time > time_limit - 10:
                break
            b_idx = current_blocks[i]
            if try_remove(b_idx):
                removed_any = True
                break
                
        if not removed_any:
            # 2. Try swaps - remove one block, add one that covers what was uncovered
            if len(current_blocks) > 0:
                # Pick a block with low redundancy (most critical) to try removing
                # Actually, pick a random block to explore
                b_old = random.choice(current_blocks)
                counts[block_4s_array[b_old]] -= 1
                uncovered_now = (counts == 0)
                
                if np.any(uncovered_now):
                    unc_idx = np.where(uncovered_now)[0]
                    
                    # Find candidate blocks that cover uncovered sets
                    candidates = set()
                    for uid in unc_idx:
                        candidates.update(sub_blocks[uid])
                    
                    if candidates:
                        # Pick the candidate that covers the most uncovered sets
                        best_candidate = -1
                        max_cov = -1
                        for cand in candidates:
                            cov = np.sum(uncovered_now[block_4s_array[cand]])
                            if cov > max_cov:
                                max_cov = cov
                                best_candidate = cand
                                
                        if best_candidate != -1 and best_candidate not in current_blocks:
                            current_blocks.remove(b_old)
                            current_blocks.append(best_candidate)
                            counts[block_4s_array[best_candidate]] += 1
                            
                            if not np.any(counts == 0):
                                pass  # Successful swap
                            else:
                                counts[block_4s_array[best_candidate]] -= 1
                                counts[block_4s_array[b_old]] += 1
                                current_blocks.remove(best_candidate)
                                current_blocks.append(b_old)
                        else:
                            counts[block_4s_array[b_old]] += 1
                    else:
                        counts[block_4s_array[b_old]] += 1
                else:
                    current_blocks.remove(b_old)
                    removed_any = True
                    
        # Update best
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
    # --- Final Polish Pass: Try 2-for-1 swaps (remove 2 blocks, add 1) ---
    # This can escape local minima where no single removal works but removing two together does
    if time.time() - start_time < time_limit - 5:
        # Try all pairs of blocks to see if removing both is valid
        n_cur = len(current_blocks)
        if n_cur > 1:
            # Sort by redundancy (least redundant first = most critical)
            red_scores = get_redundancy_scores()
            order = np.argsort(red_scores)  # Ascending: most critical first
            
            # Try removing pairs of blocks that are both redundant
            for i in range(n_cur):
                if time.time() - start_time > time_limit - 5:
                    break
                b1 = current_blocks[order[i]]
                # Only try if b1 has some redundancy
                if red_scores[order[i]] == 0:
                    continue
                    
                counts[block_4s_array[b1]] -= 1
                
                for j in range(i + 1, n_cur):
                    if time.time() - start_time > time_limit - 5:
                        break
                    b2 = current_blocks[order[j]]
                    if red_scores[order[j]] == 0:
                        continue
                        
                    counts[block_4s_array[b2]] -= 1
                    
                    if not np.any(counts == 0):
                        # Both removed successfully
                        current_blocks.remove(b1)
                        current_blocks.remove(b2)
                        # Try to add a block that covers what was uncovered
                        # Actually, if we removed both and it's still valid, great!
                        # But we should try to add one block back if it helps
                        # For now, just keep both removed
                        break
                    else:
                        # Need to add a block to cover the gap
                        uncovered_now = (counts == 0)
                        unc_idx = np.where(uncovered_now)[0]
                        
                        candidates = set()
                        for uid in unc_idx:
                            candidates.update(sub_blocks[uid])
                        
                        if candidates:
                            best_candidate = -1
                            max_cov = -1
                            for cand in candidates:
                                if cand == b1 or cand == b2:
                                    continue
                                cov = np.sum(uncovered_now[block_4s_array[cand]])
                                if cov > max_cov:
                                    max_cov = cov
                                    best_candidate = cand
                            
                            if best_candidate != -1 and best_candidate not in current_blocks:
                                # 2-for-1 swap: remove b1, b2; add best_candidate
                                current_blocks.remove(b1)
                                current_blocks.remove(b2)
                                current_blocks.append(best_candidate)
                                counts[block_4s_array[best_candidate]] += 1
                                
                                if not np.any(counts == 0):
                                    break  # Success
                                else:
                                    # Revert
                                    counts[block_4s_array[best_candidate]] -= 1
                                    current_blocks.remove(best_candidate)
                                    current_blocks.append(b1)
                                    current_blocks.append(b2)
                                    counts[block_4s_array[b1]] += 1
                                    counts[block_4s_array[b2]] += 1
                                    continue
                        else:
                            # Revert
                            counts[block_4s_array[b1]] += 1
                            counts[block_4s_array[b2]] += 1
                            continue
                else:
                    # Revert b1
                    counts[block_4s_array[b1]] += 1
                    
            # Update best after 2-for-1 pass
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)
                
    # Ensure we return a valid covering
    if not np.all(counts >= 1):
        uncovered_mask = (counts == 0)
        while np.any(uncovered_mask):
            scores = uncovered_mask[block_4s_array].sum(axis=1)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            if best_idx not in current_blocks:
                current_blocks.append(best_idx)
                counts[block_4s_array[best_idx]] += 1
                uncovered_mask = (counts == 0)
            else:
                break

    # Use the best found blocks
    final_blocks = best_blocks
    result = [list(all_blocks[i]) for i in final_blocks]
    return result

def _greedy_generic(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks

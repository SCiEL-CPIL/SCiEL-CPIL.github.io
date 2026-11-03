import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0
    
    points = range(v)
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    block_tsubs = [None] * n_blocks
    block_mask_for_tsub = [0] * n_tsubsets
    block_cover_masks = [0] * n_blocks
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
        
        mask = 0
        bit_block = 1 << bi
        for ts_idx in tsubs:
            block_mask_for_tsub[ts_idx] |= bit_block
            mask |= (1 << ts_idx)
        block_cover_masks[bi] = mask

    def greedy_cover(seed=0):
        rng = random.Random(seed)
        covered_mask = 0
        selected = []
        full_mask = (1 << n_tsubsets) - 1
        
        while covered_mask != full_mask:
            if time.time() > deadline - 10:
                break
            
            best_gain = 0
            best_blocks = []
            
            # Optimization: If we are close to full, we might find it faster
            # Just scan all blocks
            not_covered = ~covered_mask
            
            for bi in range(n_blocks):
                gain = (block_cover_masks[bi] & not_covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [bi]
                elif gain == best_gain and gain > 0:
                    best_blocks.append(bi)
            
            if not best_blocks:
                break
                
            chosen = rng.choice(best_blocks)
            selected.append(chosen)
            covered_mask |= block_cover_masks[chosen]
            
        return selected

    def minimize_cover(block_indices, seed=0):
        rng = random.Random(seed)
        selected = set(block_indices)
        
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            cover_count[block_tsubs[bi]] += 1
        
        best_selected = set(selected)
        best_count = len(selected)
        
        def is_redundant(bi):
            ts_indices = block_tsubs[bi]
            return np.min(cover_count[ts_indices]) > 1

        def remove_redundant():
            nonlocal best_selected, best_count
            removed_any = True
            while removed_any:
                removed_any = False
                for bi in list(selected):
                    if is_redundant(bi):
                        selected.discard(bi)
                        cover_count[block_tsubs[bi]] -= 1
                        removed_any = True
                        if len(selected) < best_count:
                            best_selected = set(selected)
                            best_count = len(selected)

        def get_random_bit(mask):
            if mask == 0:
                return -1
            bl = mask.bit_length()
            while True:
                pos = rng.randint(0, bl - 1)
                if (mask >> pos) & 1:
                    return pos

        def try_swap():
            if not selected:
                return False
            sel_list = list(selected)
            bi = sel_list[rng.randint(0, len(sel_list)-1)]
            
            ts_indices = block_tsubs[bi]
            unique_mask = cover_count[ts_indices] == 1
            unique_tsubs = ts_indices[unique_mask]
            
            if len(unique_tsubs) == 0:
                return False
            
            cand_mask = (1 << n_blocks) - 1
            for u in unique_tsubs:
                cand_mask &= block_mask_for_tsub[u]
                if cand_mask == 0:
                    break
            
            if cand_mask == 0:
                return False
            
            cand_mask &= ~(1 << bi)
            
            if cand_mask == 0:
                return False
            
            cj = get_random_bit(cand_mask)
            if cj == -1:
                return False
            
            selected.discard(bi)
            cover_count[ts_indices] -= 1
            
            cj_ts = block_tsubs[cj]
            selected.add(cj)
            cover_count[cj_ts] += 1
            
            return True

        remove_redundant()
        
        max_steps = 5000
        no_improve = 0
        step = 0
        
        while step < max_steps and time.time() < deadline - 2:
            step += 1
            
            if len(selected) < best_count:
                best_selected = set(selected)
                best_count = len(selected)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 1000:
                    if selected:
                        new_bi = rng.randint(0, n_blocks - 1)
                        if new_bi not in selected:
                            selected.add(new_bi)
                            cover_count[block_tsubs[new_bi]] += 1
                            no_improve = 0
                    else:
                        break
            
            remove_redundant()
            
            if try_swap():
                remove_redundant()
            
            if rng.random() < 0.01:
                new_bi = rng.randint(0, n_blocks - 1)
                if new_bi not in selected:
                    selected.add(new_bi)
                    cover_count[block_tsubs[new_bi]] += 1
                    remove_redundant()

        return [all_blocks[bi] for bi in best_selected]

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 10
    for seed in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        init_blocks = greedy_cover(seed)
        if not init_blocks:
            continue
        
        refined = minimize_cover(init_blocks, seed)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined

    if best_blocks is None:
        best_blocks = greedy_cover(0)

    return best_blocks

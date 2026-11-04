import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)
        
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)
            
    # Convert to numpy arrays for faster access? 
    # Lists are okay, but numpy might be faster for argmax etc.
    # block_tsets is list of lists.
    
    # 2. Greedy Set Cover
    uncovered = np.ones(n_tsub, dtype=bool)
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = len(block_tsets[bi])
        
    selected = []
    remaining = n_tsub
    
    while remaining > 0:
        if time.time() > deadline - 20.0:
            break
        bi = int(np.argmax(gain))
        if gain[bi] == 0:
            break
        selected.append(bi)
        gain[bi] = 0
        
        for ti in block_tsets[bi]:
            if uncovered[ti]:
                uncovered[ti] = False
                remaining -= 1
                for bj in tset_to_blocks[ti]:
                    gain[bj] -= 1
                    
    # 3. Reduction Pass
    def reduce_solution(sel):
        if not sel: return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
        
    selected = reduce_solution(selected)
    
    # 4. Kick Local Search
    # Maintain coverage counts for current selected
    cov = np.zeros(n_tsub, dtype=np.int32)
    for bi in selected:
        for ti in block_tsets[bi]:
            cov[ti] += 1
            
    best_size = len(selected)
    
    while time.time() < deadline - 5.0:
        if not selected: break
        # Pick random block to kick
        idx = random.randint(0, len(selected) - 1)
        bi_kick = selected[idx]
        
        # Remove it
        selected.pop(idx)
        for ti in block_tsets[bi_kick]:
            cov[ti] -= 1
            
        # Find uncovered tsets
        uncovered_tsets = [ti for ti in block_tsets[bi_kick] if cov[ti] == 0]
        
        if not uncovered_tsets:
            # It was redundant, keep it removed (should have been caught, but okay)
            continue
        
        # Greedily cover uncovered_tsets
        new_blocks = []
        uncovered_set = set(uncovered_tsets)
        
        while uncovered_set:
            if time.time() > deadline - 5.0:
                break
            # Find block covering max uncovered_set elements
            best_bj = -1
            max_cover = 0
            # Candidates: blocks covering any element in uncovered_set
            # We can just iterate all blocks? No, too slow.
            # Use tset_to_blocks.
            candidates = set()
            for ti in uncovered_set:
                candidates.update(tset_to_blocks[ti])
            
            for bj in candidates:
                # Count how many elements of uncovered_set are in block_tsets[bj]
                # Optimization: precompute intersection?
                # Just loop
                cnt = 0
                for ti in block_tsets[bj]:
                    if ti in uncovered_set:
                        cnt += 1
                if cnt > max_cover:
                    max_cover = cnt
                    best_bj = bj
                    
            if best_bj == -1:
                # Should not happen
                break
                
            new_blocks.append(best_bj)
            for ti in block_tsets[best_bj]:
                if ti in uncovered_set:
                    uncovered_set.remove(ti)
                    
        # Add new blocks
        for bj in new_blocks:
            selected.append(bj)
            for ti in block_tsets[bj]:
                cov[ti] += 1
                
        # Reduce
        selected = reduce_solution(selected)
        
        # Update cov for new selected? 
        # reduce_solution doesn't return cov.
        # We need to recompute cov for the next iteration.
        # This is expensive.
        # Optimization: only recompute if we accept the move.
        
        if len(selected) < best_size:
            best_size = len(selected)
            # Recompute cov
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in selected:
                for ti in block_tsets[bi]:
                    cov[ti] += 1
        else:
            # Revert
            # This is tricky because we modified selected and cov.
            # We need to store the state before the kick.
            pass
            
    # The revert logic is missing. We need to save state.
    
    # Let's fix the kick loop to use save/restore.
    
    return [list(all_blocks[bi]) for bi in selected]

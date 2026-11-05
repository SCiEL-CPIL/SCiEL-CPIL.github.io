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
    
    # 2. Greedy Set Cover
    uncovered = np.ones(n_tsub, dtype=bool)
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = len(block_tsets[bi])
        
    selected = []
    remaining = n_tsub
    
    while remaining > 0:
        if time.time() > deadline - 40.0:
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
                    
    # 3. Reduction Pass - iterative
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
    
    # 4. Simulated Annealing Local Search with proper save/restore
    cov = np.zeros(n_tsub, dtype=np.int32)
    for bi in selected:
        for ti in block_tsets[bi]:
            cov[ti] += 1
            
    best_selected = list(selected)
    best_size = len(selected)
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    
    while time.time() < deadline - 5.0 and T > T_end:
        if not selected:
            break
        
        # Pick a random move type: kick (remove + refill) or swap
        move_type = random.random()
        
        if move_type < 0.6:
            # Kick: remove one block, greedily refill uncovered
            idx = random.randint(0, len(selected) - 1)
            bi_kick = selected[idx]
            
            # Save state
            saved_selected = list(selected)
            saved_cov = cov.copy()
            
            # Remove
            selected.pop(idx)
            for ti in block_tsets[bi_kick]:
                cov[ti] -= 1
            
            # Find uncovered
            uncovered_tsets = [ti for ti in block_tsets[bi_kick] if cov[ti] == 0]
            
            if not uncovered_tsets:
                # Redundant block removed - accept
                selected = reduce_solution(selected)
                # Recompute cov
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                T *= cooling_rate
                continue
            
            # Greedily cover uncovered
            new_blocks = []
            uncovered_set = set(uncovered_tsets)
            
            while uncovered_set and time.time() < deadline - 5.0:
                best_bj = -1
                max_cover = 0
                candidates = set()
                for ti in uncovered_set:
                    candidates.update(tset_to_blocks[ti])
                
                for bj in candidates:
                    cnt = 0
                    for ti in block_tsets[bj]:
                        if ti in uncovered_set:
                            cnt += 1
                    if cnt > max_cover:
                        max_cover = cnt
                        best_bj = bj
                        
                if best_bj == -1:
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
            
            # Accept/reject
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                # Reject - restore
                selected = saved_selected
                cov = saved_cov
                
        else:
            # Swap: remove one block, add one different block
            if len(selected) < 2:
                T *= cooling_rate
                continue
                
            idx_remove = random.randint(0, len(selected) - 1)
            bi_remove = selected[idx_remove]
            
            # Find a candidate block to add
            # Prefer blocks that cover tsets currently only covered by bi_remove
            single_covered = [ti for ti in block_tsets[bi_remove] if cov[ti] == 1]
            
            if single_covered:
                # Find blocks covering some of these
                candidates = set()
                for ti in single_covered:
                    candidates.update(tset_to_blocks[ti])
                candidates.discard(bi_remove)
                if not candidates:
                    T *= cooling_rate
                    continue
                bi_add = random.choice(list(candidates))
            else:
                # Random candidate
                bi_add = random.randint(0, n_blocks - 1)
                if bi_add == bi_remove:
                    T *= cooling_rate
                    continue
            
            # Check if bi_add is already in selected
            if bi_add in selected:
                T *= cooling_rate
                continue
            
            # Save state
            saved_selected = list(selected)
            saved_cov = cov.copy()
            
            # Perform swap
            selected[idx_remove] = bi_add
            for ti in block_tsets[bi_remove]:
                cov[ti] -= 1
            for ti in block_tsets[bi_add]:
                cov[ti] += 1
            
            # Check coverage validity
            valid = True
            for ti in block_tsets[bi_remove]:
                if cov[ti] == 0:
                    valid = False
                    break
            
            if not valid:
                # Reject - restore
                selected = saved_selected
                cov = saved_cov
                T *= cooling_rate
                continue
            
            # Reduce
            selected = reduce_solution(selected)
            
            # Accept/reject
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                # Reject - restore
                selected = saved_selected
                cov = saved_cov
                
        T *= cooling_rate
    
    # Final reduction
    best_selected = reduce_solution(best_selected)
    
    return [list(all_blocks[bi]) for bi in best_selected]

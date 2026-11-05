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
    
    # Precompute numpy arrays for faster operations
    block_tsets_arr = np.array(block_tsets, dtype=np.int32)
    block_tsets_len = np.array([len(ts) for ts in block_tsets], dtype=np.int32)
    
    # 2. Greedy Set Cover with vectorized gain updates
    def greedy_cover(seed_offset=0):
        rng = random.Random(seed_offset)
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = block_tsets_len.copy()
        
        selected = []
        remaining = n_tsub
        
        while remaining > 0:
            if time.time() > deadline - 45.0:
                break
            # Add small random perturbation for diversification
            if rng.random() < 0.1:
                perturbation = rng.random() * 0.1
                gain = gain + perturbation
                bi = int(np.argmax(gain))
                gain -= perturbation
            else:
                bi = int(np.argmax(gain))
            if gain[bi] == 0:
                break
            selected.append(bi)
            gain[bi] = 0
            
            # Find newly covered tsets using numpy
            bi_tsets = block_tsets_arr[bi]
            newly_mask = uncovered[bi_tsets]
            newly_covered = bi_tsets[newly_mask]
            
            if len(newly_covered) > 0:
                remaining -= len(newly_covered)
                uncovered[newly_covered] = False
                # Vectorized gain decrement
                for ti in newly_covered:
                    blocks_covering = tset_to_blocks[ti]
                    if blocks_covering:
                        gain[blocks_covering] -= 1
                        
        return selected
    
    selected = greedy_cover(42)
    
    # 3. Fast Reduction Pass
    def reduce_solution(sel):
        if not sel: return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            cov[block_tsets_arr[bi]] += 1
                
        new_sel = []
        for bi in sel:
            bi_tsets = block_tsets_arr[bi]
            essential = np.any(cov[bi_tsets] == 1)
            if not essential:
                cov[bi_tsets] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
        
    selected = reduce_solution(selected)
    
    # 4. Aggressive deletion polish: repeatedly try removing each block
    def aggressive_polish(sel, time_limit):
        if not sel:
            return sel
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in sel:
                cov[block_tsets_arr[bi]] += 1
            new_sel = []
            for bi in sel:
                bi_tsets = block_tsets_arr[bi]
                essential = np.any(cov[bi_tsets] == 1)
                if not essential:
                    cov[bi_tsets] -= 1
                    changed = True
                else:
                    new_sel.append(bi)
            sel = new_sel
        return sel
    
    selected = aggressive_polish(selected, deadline - 35.0)
    
    # 5. Simulated Annealing Local Search
    cov = np.zeros(n_tsub, dtype=np.int32)
    for bi in selected:
        cov[block_tsets_arr[bi]] += 1
            
    best_selected = list(selected)
    best_size = len(selected)
    
    # Track selected as a set for O(1) membership
    selected_set = set(selected)
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.997
    
    T = T_start
    
    while time.time() < deadline - 8.0 and T > T_end:
        if not selected:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Kick: remove one block, greedily refill uncovered
            idx = random.randint(0, len(selected) - 1)
            bi_kick = selected[idx]
            
            saved_selected = list(selected)
            saved_set = set(selected)
            saved_cov = cov.copy()
            
            # Remove
            selected.pop(idx)
            selected_set.discard(bi_kick)
            kick_tsets = block_tsets_arr[bi_kick]
            cov[kick_tsets] -= 1
            
            # Find uncovered tsets using numpy
            uncovered_mask = cov[kick_tsets] == 0
            uncovered_tsets = kick_tsets[uncovered_mask]
            
            if len(uncovered_tsets) == 0:
                selected = aggressive_polish(selected, deadline - 8.0)
                selected_set = set(selected)
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_arr[bi]] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                T *= cooling_rate
                continue
            
            # Greedily cover uncovered
            new_blocks = []
            uncovered_set = set(uncovered_tsets.tolist())
            
            while uncovered_set and time.time() < deadline - 8.0:
                best_bj = -1
                max_cover = 0
                candidates = set()
                for ti in uncovered_set:
                    candidates.update(tset_to_blocks[ti])
                
                for bj in candidates:
                    cnt = 0
                    for ti in block_tsets_arr[bj]:
                        if ti in uncovered_set:
                            cnt += 1
                    if cnt > max_cover:
                        max_cover = cnt
                        best_bj = bj
                        
                if best_bj == -1:
                    break
                    
                new_blocks.append(best_bj)
                bj_tsets = block_tsets_arr[best_bj]
                for ti in bj_tsets:
                    if ti in uncovered_set:
                        uncovered_set.discard(ti)
                        
            for bj in new_blocks:
                selected.append(bj)
                selected_set.add(bj)
                cov[block_tsets_arr[bj]] += 1
                    
            selected = aggressive_polish(selected, deadline - 8.0)
            selected_set = set(selected)
            
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_arr[bi]] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                selected_set = saved_set
                cov = saved_cov
                
        else:
            # Swap: remove one block, add one different block
            if len(selected) < 2:
                T *= cooling_rate
                continue
                
            idx_remove = random.randint(0, len(selected) - 1)
            bi_remove = selected[idx_remove]
            
            remove_tsets = block_tsets_arr[bi_remove]
            single_covered_mask = cov[remove_tsets] == 1
            single_covered = remove_tsets[single_covered_mask]
            
            if len(single_covered) > 0:
                candidates = set()
                for ti in single_covered:
                    candidates.update(tset_to_blocks[ti])
                candidates.discard(bi_remove)
                candidates -= selected_set
                if not candidates:
                    T *= cooling_rate
                    continue
                bi_add = random.choice(list(candidates))
            else:
                bi_add = random.randint(0, n_blocks - 1)
                if bi_add == bi_remove or bi_add in selected_set:
                    T *= cooling_rate
                    continue
            
            saved_selected = list(selected)
            saved_set = set(selected)
            saved_cov = cov.copy()
            
            # Perform swap
            selected[idx_remove] = bi_add
            selected_set.discard(bi_remove)
            selected_set.add(bi_add)
            cov[remove_tsets] -= 1
            add_tsets = block_tsets_arr[bi_add]
            cov[add_tsets] += 1
            
            # Check coverage validity
            valid = np.all(cov[remove_tsets] > 0)
            
            if not valid:
                selected = saved_selected
                selected_set = saved_set
                cov = saved_cov
                T *= cooling_rate
                continue
            
            selected = aggressive_polish(selected, deadline - 8.0)
            selected_set = set(selected)
            
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_arr[bi]] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                selected_set = saved_set
                cov = saved_cov
                
        T *= cooling_rate
    
    # 6. Multi-restart with different seeds
    if time.time() < deadline - 25.0:
        for restart_seed in [123, 456, 789, 101]:
            if time.time() > deadline - 20.0:
                break
            restart_selected = greedy_cover(restart_seed)
            restart_selected = reduce_solution(restart_selected)
            restart_selected = aggressive_polish(restart_selected, deadline - 15.0)
            
            if len(restart_selected) < best_size:
                best_size = len(restart_selected)
                best_selected = list(restart_selected)
    
    # Final aggressive polish
    best_selected = aggressive_polish(best_selected, deadline - 2.0)
    
    return [list(all_blocks[bi]) for bi in best_selected]

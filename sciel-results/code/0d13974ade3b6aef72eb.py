import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0  # Tighter safety margin for 120s kill

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
    
    # Precompute block_tsets as numpy arrays for vectorized operations
    block_tsets_np = [np.array(ts, dtype=np.int32) for ts in block_tsets]
    
    # Precompute for faster coverage checks
    # block_tsets_set for O(1) membership
    block_tsets_set = [set(ts) for ts in block_tsets]
    
    # Incidence matrix for fast gain computation (sparse representation)
    # For each block, store the tset indices as a set for quick intersection
    
    # 2. Greedy Set Cover with vectorized gain updates
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.array([len(block_tsets[bi]) for bi in range(n_blocks)], dtype=np.int32)
        
        selected = []
        remaining = n_tsub
        
        while remaining > 0:
            if time.time() > deadline - 55.0:
                break
            
            # Add small noise to break ties for randomization
            if seed > 0:
                scores = gain.astype(np.float64) + rng.random() * 0.01
            else:
                scores = gain.astype(np.float64)
            
            bi = int(np.argmax(scores))
            if scores[bi] < 0.5:
                break
            
            selected.append(bi)
            gain[bi] = 0
            
            # Vectorized gain update
            for ti in block_tsets_np[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    # Update gains for all blocks covering this tset
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1
                    
        return selected
    
    selected = greedy_cover(0)
    
    # 3. Reduction Pass - iterative with O(1) set membership
    def reduce_solution(sel):
        if not sel:
            return []
        
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets_np[bi]:
                cov[ti] += 1
                
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets_np[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets_np[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
        
    selected = reduce_solution(selected)
    
    # Helper: compute coverage array from selected blocks
    def compute_cov(sel):
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets_np[bi]:
                cov[ti] += 1
        return cov
    
    # 4. Simulated Annealing Local Search
    cov = compute_cov(selected)
    
    best_selected = list(selected)
    best_size = len(selected)
    
    T_start = 5.0
    T_end = 0.05
    cooling_rate = 0.997
    T = T_start
    
    # Precompute set of selected blocks for O(1) membership
    selected_set = set(selected)
    
    while time.time() < deadline - 15.0 and T > T_end:
        if not selected:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Kick: remove one block, greedily refill uncovered
            idx = random.randint(0, len(selected) - 1)
            bi_kick = selected[idx]
            
            saved_selected = list(selected)
            saved_cov = cov.copy()
            saved_selected_set = set(selected)
            
            selected.pop(idx)
            selected_set.discard(bi_kick)
            for ti in block_tsets_np[bi_kick]:
                cov[ti] -= 1
            
            uncovered_tsets = [ti for ti in block_tsets_np[bi_kick] if cov[ti] == 0]
            
            if not uncovered_tsets:
                selected = reduce_solution(selected)
                selected_set = set(selected)
                cov = compute_cov(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                T *= cooling_rate
                continue
            
            new_blocks = []
            uncovered_set = set(uncovered_tsets)
            
            while uncovered_set and time.time() < deadline - 15.0:
                # Find best block covering most uncovered tsets
                # Use vectorized approach: for each candidate block, count overlap with uncovered
                best_bj = -1
                max_cover = 0
                
                # Collect candidate blocks
                candidates = set()
                for ti in uncovered_set:
                    candidates.update(tset_to_blocks[ti])
                
                # Filter out already selected blocks
                candidates -= selected_set
                
                if not candidates:
                    break
                    
                # Vectorized gain computation
                uncovered_arr = np.zeros(n_tsub, dtype=bool)
                for ti in uncovered_set:
                    uncovered_arr[ti] = True
                
                for bj in candidates:
                    cnt = np.sum(uncovered_arr[block_tsets_np[bj]])
                    if cnt > max_cover:
                        max_cover = cnt
                        best_bj = bj
                        
                if best_bj == -1:
                    break
                    
                new_blocks.append(best_bj)
                selected_set.add(best_bj)
                for ti in block_tsets_np[best_bj]:
                    if ti in uncovered_set:
                        uncovered_set.remove(ti)
                        
            for bj in new_blocks:
                selected.append(bj)
                for ti in block_tsets_np[bj]:
                    cov[ti] += 1
                    
            selected = reduce_solution(selected)
            selected_set = set(selected)
            
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                cov = compute_cov(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                selected_set = saved_selected_set
                cov = saved_cov
                
        else:
            # Swap: remove one block, add one different block
            if len(selected) < 2:
                T *= cooling_rate
                continue
                
            idx_remove = random.randint(0, len(selected) - 1)
            bi_remove = selected[idx_remove]
            
            single_covered = [ti for ti in block_tsets_np[bi_remove] if cov[ti] == 1]
            
            if single_covered:
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
                # Try to find a block that covers some currently singly-covered tsets
                # or just pick randomly
                bi_add = random.randint(0, n_blocks - 1)
                if bi_add == bi_remove or bi_add in selected_set:
                    T *= cooling_rate
                    continue
            
            saved_selected = list(selected)
            saved_cov = cov.copy()
            saved_selected_set = set(selected)
            
            selected[idx_remove] = bi_add
            selected_set.discard(bi_remove)
            selected_set.add(bi_add)
            for ti in block_tsets_np[bi_remove]:
                cov[ti] -= 1
            for ti in block_tsets_np[bi_add]:
                cov[ti] += 1
            
            valid = True
            for ti in block_tsets_np[bi_remove]:
                if cov[ti] == 0:
                    valid = False
                    break
            
            if not valid:
                selected = saved_selected
                selected_set = saved_selected_set
                cov = saved_cov
                T *= cooling_rate
                continue
            
            selected = reduce_solution(selected)
            selected_set = set(selected)
            
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                cov = compute_cov(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                selected_set = saved_selected_set
                cov = saved_cov
                
        T *= cooling_rate
    
    # 5. Final aggressive deletion polish pass
    best_selected = reduce_solution(best_selected)
    
    # Additional polish: try removing blocks in random order
    rng_polish = random.Random(12345)
    for _ in range(10):
        if time.time() > deadline - 3.0:
            break
        order = list(range(len(best_selected)))
        rng_polish.shuffle(order)
        new_sel = []
        cov_check = np.zeros(n_tsub, dtype=np.int32)
        for bi in best_selected:
            for ti in block_tsets_np[bi]:
                cov_check[ti] += 1
        
        for idx in order:
            bi = best_selected[idx]
            essential = False
            for ti in block_tsets_np[bi]:
                if cov_check[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets_np[bi]:
                    cov_check[ti] -= 1
            else:
                new_sel.append(bi)
        
        if len(new_sel) < len(best_selected):
            best_selected = new_sel
        else:
            break
    
    # Final verification and return
    return [list(all_blocks[bi]) for bi in best_selected]

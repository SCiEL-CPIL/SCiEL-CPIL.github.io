import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # Safety margin
    
    # Precomputation
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

    def time_left():
        return deadline - time.time()

    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            gain[bi] = len(block_tsets[bi])
            
        selected = []
        remaining = n_tsub
        
        while remaining > 0 and time_left() > 10.0:
            # Find max gain, with random tie-breaking
            max_gain = int(np.max(gain))
            if max_gain == 0:
                break
            candidates = [bi for bi in range(n_blocks) if gain[bi] == max_gain]
            bi = rng.choice(candidates)
            selected.append(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        if gain[bj] > 0:
                            gain[bj] -= 1
                            
        return selected

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

    def local_search(sel, seed=0):
        if not sel: return sel
        rng = random.Random(seed)
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        sel_set = set(sel)
        improved = True
        max_rounds = 50
        round_num = 0
        while improved and time_left() > 5.0 and round_num < max_rounds:
            improved = False
            round_num += 1
            # Shuffle the order of checking for diversity
            order = list(range(len(sel)))
            rng.shuffle(order)
            for i in order:
                if time_left() < 5.0: break
                bi_remove = sel[i]
                
                newly_uncovered = []
                for ti in block_tsets[bi_remove]:
                    if cov[ti] == 1:
                        newly_uncovered.append(ti)
                        
                if not newly_uncovered:
                    # Pure deletion - best case
                    sel.pop(i)
                    for ti in block_tsets[bi_remove]:
                        cov[ti] -= 1
                    sel_set.discard(bi_remove)
                    improved = True
                    break
                    
                # Try to find a single block that covers all newly_uncovered
                # and doesn't add too much overhead
                candidates = None
                for ti in newly_uncovered:
                    if candidates is None:
                        candidates = set(tset_to_blocks[ti])
                    else:
                        candidates.intersection_update(tset_to_blocks[ti])
                        
                if not candidates:
                    continue
                    
                candidates.difference_update(sel_set)
                
                if not candidates:
                    continue
                
                # Score each candidate: prefer blocks that cover many already-covered tsets
                # (i.e., have high overlap with current solution)
                best_bj = None
                best_score = -1
                for bj in list(candidates):
                    # Count how many tsets in bj are already covered
                    overlap = 0
                    for ti in block_tsets[bj]:
                        if cov[ti] > 0:
                            overlap += 1
                    score = overlap
                    if score > best_score:
                        best_score = score
                        best_bj = bj
                
                if best_bj is None:
                    continue
                    
                # Only swap if it doesn't increase size (it won't, 1-for-1)
                # and has good overlap
                sel[i] = best_bj
                sel_set.discard(bi_remove)
                sel_set.add(best_bj)
                
                for ti in block_tsets[bi_remove]:
                    cov[ti] -= 1
                for ti in block_tsets[best_bj]:
                    cov[ti] += 1
                    
                improved = True
                break
                
        return sel

    # Main: Use multiple greedy restarts with local search
    best_sol = None
    best_size = float('inf')
    
    # First, get a solid baseline
    base_sol = greedy_cover(seed=42)
    base_sol = reduce_solution(base_sol)
    base_sol = local_search(base_sol, seed=42)
    best_sol = base_sol
    best_size = len(best_sol)
    
    # Now do multiple restarts with different seeds and perturbation
    restart_num = 0
    while time_left() > 15.0:
        restart_num += 1
        
        # Strategy: take current best, remove some blocks, re-cover greedily
        current = list(best_sol)
        
        # Random perturbation: remove 2-5 blocks
        n_remove = random.randint(2, min(5, len(current)))
        if len(current) <= n_remove:
            current = []
        else:
            remove_idx = random.sample(range(len(current)), n_remove)
            for idx in sorted(remove_idx, reverse=True):
                current.pop(idx)
                
        # Re-cover greedily from the perturbed state
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in current:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        uncovered_mask = (cov == 0)
        remaining = int(np.sum(uncovered_mask))
        
        if remaining == 0:
            new_sol = local_search(current, seed=restart_num)
            if len(new_sol) < best_size:
                best_sol = new_sol
                best_size = len(new_sol)
            continue
            
        current_set = set(current)
        
        # Greedy completion with randomization
        rng = random.Random(restart_num)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            if bi in current_set:
                continue
            cnt = 0
            for ti in block_tsets[bi]:
                if uncovered_mask[ti]:
                    cnt += 1
            gain[bi] = cnt
            
        while remaining > 0 and time_left() > 10.0:
            max_gain = int(np.max(gain))
            if max_gain == 0:
                break
            candidates = [bi for bi in range(n_blocks) if gain[bi] == max_gain]
            bi = rng.choice(candidates)
            current.append(bi)
            current_set.add(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        if gain[bj] > 0:
                            gain[bj] -= 1
                            
        new_sol = reduce_solution(current)
        new_sol = local_search(new_sol, seed=restart_num)
        
        if len(new_sol) < best_size:
            best_sol = new_sol
            best_size = len(new_sol)
            
    # Final polish: one more local search pass
    best_sol = local_search(best_sol, seed=999)
    best_sol = reduce_solution(best_sol)
    
    return [list(all_blocks[bi]) for bi in best_sol]

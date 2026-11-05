import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 135.0  # Safety margin before 150s kill
    
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

    # Helper to check time
    def time_left():
        return deadline - time.time()

    # 2. Greedy Set Cover with incremental gain updates
    def greedy_cover():
        uncovered = np.ones(n_tsub, dtype=bool)
        # gain[bi] = number of currently uncovered tsets covered by block bi
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            gain[bi] = len(block_tsets[bi])
            
        selected = []
        remaining = n_tsub
        
        while remaining > 0 and time_left() > 10.0:
            bi = int(np.argmax(gain))
            if gain[bi] == 0:
                break
            selected.append(bi)
            gain[bi] = 0 # Mark as used
            
            # Update gains for other blocks
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    # For every block covering this tset, its gain decreases by 1
                    for bj in tset_to_blocks[ti]:
                        if gain[bj] > 0: # Optimization: only if not already 0
                            gain[bj] -= 1
                            
        return selected

    # 3. Reduction Pass (Deletion)
    def reduce_solution(sel):
        if not sel: return []
        # Count coverage
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

    # 4. Local Search (1-Swap)
    def local_search(sel):
        if not sel: return sel
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        sel_set = set(sel)
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            # Try removing each block
            for i in range(len(sel)):
                if time_left() < 5.0: break
                bi_remove = sel[i]
                
                # Check if removing bi_remove uncovers any tset
                newly_uncovered = []
                for ti in block_tsets[bi_remove]:
                    if cov[ti] == 1:
                        newly_uncovered.append(ti)
                        
                if not newly_uncovered:
                    # Redundant
                    sel.pop(i)
                    for ti in block_tsets[bi_remove]:
                        cov[ti] -= 1
                    sel_set.discard(bi_remove)
                    improved = True
                    break # Restart loop as indices changed
                    
                # Find a block to add that covers all newly_uncovered
                # Candidates must cover all ti in newly_uncovered
                # Intersect tset_to_blocks for each ti in newly_uncovered
                # Then filter out blocks already in sel_set
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
                    
                # Pick a candidate. Prefer one that covers many other tsets (high "synergy")
                # Or just pick the first one?
                # Let's pick the one with max coverage of currently covered tsets (to maximize redundancy for future removals)
                best_bj = -1
                best_score = -1
                for bj in candidates:
                    # Score: how many tsets in bj are already covered by others?
                    # Actually, we want to add a block that is "good".
                    # A good block covers many tsets.
                    # Since all blocks cover same number of tsets, maybe pick one that shares many tsets with existing blocks?
                    # This creates dense clusters, which might be easier to reduce later?
                    # Or pick one that is "isolated"?
                    # Let's just pick the first valid one for speed, or random.
                    # Randomization helps escape local minima.
                    best_bj = random.choice(list(candidates))
                    break # Just take one
                    
                if best_bj != -1:
                    # Perform swap
                    sel[i] = best_bj
                    sel_set.discard(bi_remove)
                    sel_set.add(best_bj)
                    
                    for ti in block_tsets[bi_remove]:
                        cov[ti] -= 1
                    for ti in block_tsets[best_bj]:
                        cov[ti] += 1
                        
                    improved = True
                    break # Restart
                    
        return sel

    # 5. Main Loop: Greedy + Local Search + Perturbation
    best_sol = greedy_cover()
    best_sol = reduce_solution(best_sol)
    best_sol = local_search(best_sol)
    best_size = len(best_sol)
    
    while time_left() > 15.0:
        # Perturb: remove a few random blocks and re-greedy
        current = list(best_sol)
        if not current:
            break
            
        # Remove 2-4 random blocks
        n_remove = random.randint(2, 4)
        if len(current) <= n_remove:
            current = []
        else:
            remove_idx = random.sample(range(len(current)), n_remove)
            for idx in sorted(remove_idx, reverse=True):
                current.pop(idx)
                
        # Re-greedy fill
        # We need to run greedy on the remaining uncovered tsets
        # But our greedy_cover starts from scratch.
        # We can modify it to start from a partial solution.
        
        # For simplicity and speed, let's just run a "fill" greedy
        # Calculate current coverage
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in current:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        uncovered_mask = (cov == 0)
        remaining = np.sum(uncovered_mask)
        
        if remaining == 0:
            # We removed blocks but didn't uncover anything? (Shouldn't happen if we removed essential ones)
            # Or we removed redundant ones.
            new_sol = local_search(current)
            if len(new_sol) < best_size:
                best_sol = new_sol
                best_size = len(best_sol)
            continue
            
        # Greedy fill
        # We need to select blocks to cover uncovered_mask.
        # We can reuse the logic but initialize gain based on uncovered_mask.
        
        # To avoid rewriting greedy, let's just append new blocks to current.
        # We need to know which blocks are already in current to avoid duplicates.
        current_set = set(current)
        
        # Calculate gain for all blocks NOT in current_set
        # gain[bi] = count of ti in block_tsets[bi] such that uncovered_mask[ti] is True
        
        # This is expensive to compute from scratch every time.
        # However, n_blocks is 75k. n_tsub is 11k.
        # 75k * 56 ops = 4.2M ops. Fast enough.
        
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
            bi = int(np.argmax(gain))
            if gain[bi] == 0:
                break
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
        new_sol = local_search(new_sol)
        
        if len(new_sol) < best_size:
            best_sol = new_sol
            best_size = len(new_sol)
            
    return [list(all_blocks[bi]) for bi in best_sol]
